import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# 1. GEOMETRIA - pontfelho -> range image
# =============================================================================
#
# A lidar nyers kimenete egy rendezetlen (N, 3) pontfelho. A konvolucio viszont
# racsot var. A megoldas: a lidar NEM veletlenszeruen szor pontokat, hanem
# szabalyos sugarracsban paszaz (64 elevacios szog x 1024 azimut). Ha ezt a
# racsot allitjuk helyre, semmi nem vesz el.
# =============================================================================

def pcd2range(pcd, size, fov, depth_range):
    """
    XYZ pontfelho -> range image (gombi projekcio).

    A harom terbeli koordinatabol KETTO a cella cime lesz, a harmadik
    (a tavolsag) pedig a cella tartalma:

        x, y  ->  azimut (yaw)    ->  OSZLOP  (0..1023)
        z     ->  elevacio (pitch)->  SOR     (0..63)
        r     ->  a CELLA ERTEKE

    pcd         : (N, 3) float, a szenzor koordinatarendszereben
    size        : (H, W), nalunk (64, 1024)
    fov         : (fov_up, fov_down) fokban, nalunk (10, -25)
    depth_range : (min, max) meterben, nalunk (1.0, 50.0)

    return: (H, W) float32; ahol nincs pont, ott -1
    """
    # Fokbol radianba - a numpy trigonometrikus fuggvenyei radiant varnak.
    fov_up = fov[0] / 180.0 * np.pi
    fov_down = fov[1] / 180.0 * np.pi
    fov_range = abs(fov_down) + abs(fov_up)     # a teljes fuggoleges latoszog

    # Minden pont tavolsaga az origotol: r = sqrt(x^2 + y^2 + z^2)
    depth = np.linalg.norm(pcd, 2, axis=1)

    # A hatotavon kivuli pontok eldobasa. Az also hatar (1 m) azert kell, mert
    # azon belul csak az ego auto sajat karosszeriaja van.
    mask = np.logical_and(depth > depth_range[0], depth < depth_range[1])
    depth, pcd = depth[mask], pcd[mask]

    scan_x, scan_y, scan_z = pcd[:, 0], pcd[:, 1], pcd[:, 2]

    # -------------------------------------------------------------------
    # DESCARTES -> GOMBI KOORDINATAK
    # -------------------------------------------------------------------
    #
    # A gombi koordinatak harom szama: (r, yaw, pitch)
    #
    #   r     = sqrt(x^2 + y^2 + z^2)     tavolsag (mar kiszamoltuk fent)
    #   yaw   = atan2(y, x)               vizszintes szog, [-pi, +pi]
    #   pitch = arcsin(z / r)             fuggoleges szog, [-pi/2, +pi/2]
    #
    # MIERT arctan2(y, x) es nem arctan(y/x)?
    #   - az arctan(y/x) nullaval osztana, ha x = 0 (pont oldalt levo pont)
    #   - az arctan csak [-pi/2, pi/2]-t ad, tehat nem tudna megkulonboztetni
    #     az "elore" es a "hatra" iranyt: y/x ugyanaz (1, 1) es (-1, -1)
    #     eseten is!
    #   Az arctan2 kulon nezi x es y elojelet, ezert mind a negy negyedet
    #   helyesen kezeli, es teljes [-pi, +pi] tartomanyt ad.
    #
    # MIERT arcsin(z / r) a pitch?
    #   Derekszogu haromszog: az atfogo r, a fuggoleges befogo z. A definicio
    #   szerint sin(pitch) = z / r, tehat pitch = arcsin(z / r).
    #   Peldak: z = 0   -> arcsin(0) = 0 fok   (vizszintes)
    #           z = r   -> arcsin(1) = 90 fok  (pont folotted)
    #
    # MIERT van MINUSZ a yaw elott?
    #   Mert a kovetkezo lepesben novekvo oszlopindexet akarunk balrol jobbra.
    #   Az elojelvaltas megforditja a korbejaras iranyat.
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # -------------------------------------------------------------------
    # SZOGEK -> [0, 1] -> CELLAINDEX
    # -------------------------------------------------------------------
    #
    # A yaw a [-pi, +pi] tartomanyban van. Ezt kell [0, 1]-be vinni:
    #
    #     yaw / pi          -> [-1, +1]        (osztas a maximummal)
    #     (yaw/pi + 1)      -> [ 0,  2]        (eltolas)
    #     0.5 * (yaw/pi + 1)-> [ 0,  1]        (osszenyomas)
    #
    # A pitch a [fov_down, fov_up] tartomanyban van (nalunk -25..+10 fok):
    #
    #     pitch + |fov_down|            -> [0, fov_range]
    #     (...) / fov_range             -> [0, 1]
    #     1 - (...)                     -> [1, 0]  <- MEGFORDITVA!
    #
    # Miert fordul meg? Mert a kepen a 0. sor van FELUL, a fizikai vilagban
    # viszont a nagy pitch (felfele nezes) tartozik oda. A kivonas ezt a ket
    # ellentetes iranyt hangolja ossze.
    proj_x = 0.5 * (yaw / np.pi + 1.0)                          # [0, 1]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov_range          # [0, 1]

    # A [0,1] arany felszorzasa a tenyleges kepmeretre.
    proj_x *= size[1]                                            # [0, W]
    proj_y *= size[0]                                            # [0, H]

    # Lefele kerekites es a hataron belulre szoritas (a float pontatlansag
    # miatt a legszelso ertek kilophetne).
    proj_x = np.maximum(0, np.minimum(size[1] - 1, np.floor(proj_x))).astype(np.int32)
    proj_y = np.maximum(0, np.minimum(size[0] - 1, np.floor(proj_y))).astype(np.int32)

    # TAKARAS KEZELESE: tobb pont eshet ugyanabba a cellaba. Csokkeno tavolsag
    # szerint irjuk be oket, igy a KOZELEBBI irja felul a tavolabbit - ami
    # fizikailag helyes, mert az takar.
    order = np.argsort(depth)[::-1]
    proj_x, proj_y, depth = proj_x[order], proj_y[order], depth[order]

    proj_range = np.full(size, -1, dtype=np.float32)
    proj_range[proj_y, proj_x] = depth
    return proj_range


def process_scan(range_img, depth_scale=6.0):
    """
    Nyers range image (meterben) -> normalizalt tensor [-1, 1].

    Harom lepes:
      1. log2(d + 1)        a kozeli tartomany felbontasat noveli
      2. / depth_scale      -> [0, 1]
      3. * 2 - 1            -> [-1, 1], mert a decoder tanh-ja ide kepez

    MIERT LOG: a kozeli tartomany a fontos. 5 es 10 meter kozott sokkal nagyobb
    a kulonbseg vezetes szempontjabol, mint 45 es 50 kozott:

        5 m -> -0.138        30 m -> 0.651
       10 m ->  0.153        50 m -> 0.891
       15 m ->  0.333

    depth_scale = 6.0, mert log2(50+1) = 5.67 - ez a biztonsagos felso hatar.

    return: (1, H, W) float32, [-1, 1]
    """
    # Ahol nincs pont (-1), oda 0 kerul. A log2(0+1) = 0, tehat ezek a cellak
    # a normalizalas utan -1-re esnek: "nagyon kozeli" helyett "ures".
    range_img = np.where(range_img < 0, 0, range_img)

    # A +1 azert kell, mert log2(0) minusz vegtelen lenne. A +0.0001 egy
    # tovabbi biztonsagi rahagyas a float pontatlansag ellen.
    range_img = np.log2(range_img + 0.0001 + 1)

    range_img = range_img / depth_scale
    range_img = range_img * 2.0 - 1.0
    range_img = np.clip(range_img, -1, 1)

    # (H, W) -> (1, H, W): a halo csatorna-dimenziot var.
    return np.expand_dims(range_img, axis=0).astype(np.float32)


def points_to_range_image(xyz, size=(64, 1024), fov=(10.0, -25.0),
                          depth_range=(1.0, 50.0), depth_scale=6.0):
    """A ket fenti lepes egyben: nyers XYZ -> (1, 64, 1024) float32, [-1, 1]."""
    return process_scan(pcd2range(xyz, size, fov, depth_range), depth_scale)


# =============================================================================
# 2. EPITOKOCKAK
# =============================================================================

class CircularConv2d(nn.Conv2d):
    """
    Konvolucio, ami vizszintesen KORKOROSEN padel.

    A padding azt jelenti, hogy a kep szelen kitoldjuk a kepet, hogy a
    konvolucio ablaka ne logjon ki. A kerdes, MIVEL toldjuk ki:

      vizszintesen (azimut): KORKOROSEN. Az 1023. oszlop mellett fizikailag a
          0. oszlop van - az azimut korbeer! (Meressel: az "elore" irany az
          512. oszlop, a kep ket szele pedig a "hatra" irany.)

      fuggolegesen (elevacio): konstanssal. A legfelso sor folott nincs semmi,
          ott nincs mit korbeerni.

    FIGYELEM: ez csak 360 fokos range image-nel helyes. Felulnezeti (BEV)
    kepnel hibas lenne - ott a bal szel az autotol balra, a jobb szel az
    autotol jobbra eso terulet, 100 meter van kozottuk.
    """

    def __init__(self, *args, **kwargs):
        # A padding-et kivesszuk a kwargs-bol, mert mi magunk vegezzuk el a
        # forward()-ban - az nn.Conv2d sajat padding-je nem tud korkoros lenni
        # kulon-kulon a ket tengelyen.
        if 'padding' in kwargs:
            self.is_pad = True
            if isinstance(kwargs['padding'], int):
                # Egyetlen szam: mind a negy oldalra ugyanannyi.
                h1 = h2 = v1 = v2 = kwargs['padding']
            elif isinstance(kwargs['padding'], tuple):
                # Negyes: (bal, jobb, fent, lent)
                h1, h2, v1, v2 = kwargs['padding']
            else:
                raise NotImplementedError

            # Az F.pad (bal, jobb, fent, lent) sorrendet var. Szetvalasztjuk a
            # ket tengelyt, mert mas modban padelunk.
            self.h_pad, self.v_pad = (h1, h2, 0, 0), (0, 0, v1, v2)
            del kwargs['padding']
        else:
            self.is_pad = False

        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if self.is_pad:
            if sum(self.h_pad) > 0:
                x = F.pad(x, self.h_pad, mode="circular")
            if sum(self.v_pad) > 0:
                x = F.pad(x, self.v_pad, mode="constant")
        # _conv_forward: az nn.Conv2d belso metodusa, ami a tenyleges
        # konvoluciot vegzi a mar padelt bemeneten.
        return self._conv_forward(x, self.weight, self.bias)


def knn(x, k):
    """
    k legkozelebbi szomszed a FEATURE-terben (nem a 3D terben).

    x      : (B, C, N)   N darab pont, mindegyik C hosszu vektor
    return : (B, N, k)   indexek

    ---------------------------------------------------------------------
    A MATEMATIKA RESZLETESEN
    ---------------------------------------------------------------------

    Ket pont (a es b, mindketto C hosszu vektor) negyzetes euklideszi
    tavolsaga definicio szerint:

        ||a - b||^2 = SUM_i (a_i - b_i)^2

    Bontsuk ki a negyzetet tagonkent:

        SUM_i (a_i^2 - 2*a_i*b_i + b_i^2)
      = SUM_i a_i^2  -  2 * SUM_i a_i*b_i  +  SUM_i b_i^2
      = ||a||^2      -  2 * (a . b)        +  ||b||^2

    Ez a HAROM TAG, es pontosan ezeket szamolja ki a kod harom sora:

        xx                  ->  ||a||^2      (minden pont hossznegyzete)
        inner               ->  -2 * (a . b) (a skalarszorzat matrix)
        xx.transpose(2, 1)  ->  ||b||^2      (ugyanaz, masik tengelyen)

    MIERT EZ A FORMA, es nem a definicio szerinti kivonas?

    Mert igy az OSSZES pontpar tavolsaga EGYETLEN matrixszorzassal adodik.
    Nezzuk a kozepso tagot: x^T * x, ahol x a (C, N) matrix. Az eredmeny
    (N, N) meretu, es az [i, j] eleme pont x_i . x_j, vagyis az i-edik es
    j-edik pont skalarszorzata. Egy hivas, N^2 skalarszorzat.

    A naiv megoldas (ket egymasba agyazott ciklus, pontparonkent kivonas)
    ugyanezt szamolna ki, de tobb nagysagrenddel lassabban - a GPU a nagy
    matrixszorzasra van optimalizalva, nem a ciklusokra.

    A BROADCASTING:
        xx alakja                 : (B, 1, N)
        inner alakja              : (B, N, N)
        xx.transpose(2,1) alakja  : (B, N, 1)

    Amikor ezeket osszeadjuk, a PyTorch automatikusan "kiteregeti" az 1-es
    tengelyeket N-re (broadcasting). Igy a (B, N, N) eredmeny [i, j] eleme:

        -||x_i||^2 + 2*(x_i . x_j) - ||x_j||^2  =  -||x_i - x_j||^2

    MIERT NEGATIV az egesz?

    Figyeld meg, hogy a fenti eredmeny a tavolsag MINUSZ EGYSZERESE. Ez
    szandekos: a topk() a LEGNAGYOBB k erteket adja vissza, mi viszont a
    LEGKISEBB tavolsagokat keressuk. A negalas megforditja a sorrendet, igy
    a legkisebb tavolsagbol lesz a legnagyobb ertek.

    KOLTSEG: N x N matrix batch-enkent.
        N = 2048  (a Stem utan)  -> 4.2 millio elem   -> megy
        N = 65536 (a nyers kep)  -> 4.3 MILLIARD elem -> kezelhetetlen
    Ezert kell a Stem downsampling.

    MEGJEGYZES: minden pont elso szomszedja onmaga, mert ||a - a||^2 = 0, ami
    a negalas utan a legnagyobb ertek. Ez szandekos - a kozeppont sajat
    feature-je is kell az EdgeConv-hoz.
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    # A [1] azert kell, mert a topk KET dolgot ad vissza: (ertekek, indexek).
    return pairwise_distance.topk(k=k, dim=-1)[1]


def get_graph_feature(x, k=20):
    """
    El-feature-ok epitese: [szomszed - kozep || kozep].

    EZ AZ EGESZ GRAF-MEGKOZELITES MAGJA (DGCNN / EdgeConv).

    ---------------------------------------------------------------------
    A MATEMATIKA
    ---------------------------------------------------------------------

    Jelolje x_i az i-edik pont feature-vektorat, es N(i) az i-edik pont k
    darab legkozelebbi szomszedjanak halmazat. Az EdgeConv minden (i, j)
    elre - ahol j eleme N(i) - kiszamol egy el-feature-t:

        e_ij = h( x_j - x_i  ||  x_i )

    ahol  ||  az osszefuzes (konkatenacio), h pedig egy tanulhato fuggveny
    (nalunk 1x1 konvolucio + BatchNorm + LeakyReLU).

    Vagyis a bemenet 2C hosszu: az elso C a KULONBSEG, a masodik C a
    KOZEPPONT sajat feature-je.

    Aztan a GraphLayer ezeket osszevonja a szomszedok felett:

        x'_i = MAX over j in N(i) of  e_ij

    ---------------------------------------------------------------------
    MIERT A KULONBSEG (x_j - x_i), ES NEM CSAK x_j?
    ---------------------------------------------------------------------

    Mert a RELATIV geometria a lenyeg. Vegyunk egy konkret peldat:

        egy auto 10 meterre:  a pontjai kb. 10.0, 10.1, 10.2 tavolsagra
        ugyanaz 30 meterre :  a pontjai kb. 30.0, 30.1, 30.2 tavolsagra

    Az ABSZOLUT ertekek teljesen masok (10 vs 30), de a KULONBSEGEK
    azonosak (0.1, 0.2). Tehat a kulonbseggel a halo egyetlen mintazatot
    tanul meg az "auto" alakzatra, nem kulon-kulon minden tavolsagra.

    Ez az ELTOLAS-INVARIANCIA: f(x + c) ugyanazt adja, mint f(x).

    De a kozeppont abszolut erteke is kell, mert a KONTEXTUS szamit: egy
    3 meterre levo akadaly mas jelentesu, mint egy 40 meterre levo.
    Ezert kapja meg a halo mindkettot.

    x      : (B, C, N)
    return : (B, 2C, N, k)
    """
    batch_size, num_dims, num_points = x.size()
    idx = knn(x, k=k)

    # INDEX-ELTOLAS TRUKK:
    # A batch minden mintajanak 0..N-1 indexei vannak. Lentebb az egeszet egy
    # nagy (B*N, C) tablava lapitjuk, ahol a 2. minta 5. pontja a
    # 2*N + 5 poziciora kerul. Az idx_base pontosan ezt az eltolast adja hozza.
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)

    # (B, C, N) -> (B, N, C). A contiguous() azert kell, mert a transpose csak
    # a "nezetet" valtoztatja meg, a memoriabeli sorrendet nem - a kovetkezo
    # view() viszont osszefuggo memoriat var.
    x = x.transpose(2, 1).contiguous()

    # A szomszedok kigyujtese a lapos tablabol, majd vissza (B, N, k, C) alakra.
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)

    # A kozeppontot k-szor megismeteljuk, hogy minden szomszed melle jusson.
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # Osszefuzes a csatorna-tengelyen -> 2C, majd (B, 2C, N, k) alakra rendezes,
    # mert a kovetkezo Conv2d csatorna-elso sorrendet var.
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class GraphLayer(nn.Module):
    """
    EdgeConv blokk: kNN -> el-feature -> 1x1 konvolucio -> max-pool.

    A graf DINAMIKUS: minden reteg ujraszamolja a kNN-t a SAJAT bemeneten,
    tehat a szomszedsagi viszonyok retegrol retegre valtoznak.
    """

    def __init__(self, channels, k=20):
        super().__init__()
        self.k = k
        # kernel_size=1: pontonkent es szomszedonkent fuggetlen linearis reteg,
        # nincs terbeli keveredes. A bemenet 2C (el-feature), a kimenet C.
        #
        # bias=False, mert utana BatchNorm jon, aminek sajat eltolasa (beta)
        # van - ket eltolas egymas utan felesleges.
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        # x: (B, C, N)
        graph_feat = get_graph_feature(x, k=self.k)   # (B, 2C, N, k)
        out = self.conv(graph_feat)                   # (B, C,  N, k)

        # MAX-POOL a szomszedok folott (dim=-1): a k szomszedbol csinal egyet.
        #
        #     x'_i = MAX over j in N(i) of e_ij      (elemenkent, csatornankent)
        #
        # MIERT MAX ES NEM ATLAG? Ket okbol:
        #
        # 1. PERMUTACIO-INVARIANCIA
        #    A kNN nem garantal sorrendet: ugyanaz a pontfelho mashogy
        #    indexelve mas sorrendben adhatja vissza a szomszedokat. Egy
        #    fuggveny akkor permutacio-invarians, ha
        #
        #        f(a, b, c) = f(b, c, a) = f(c, a, b) = ...
        #
        #    A max ilyen (a maximum nem fugg a sorrendtol). Az atlag is ilyen
        #    lenne, tehat ez onmagaban nem dontene el a kerdest. Viszont:
        #
        # 2. A MAX A "LEGEROSEBB JELET" EMELI KI
        #    Tegyuk fel, hogy a 20 szomszedbol EGY jelzi, hogy "itt egy el
        #    van" (nagy aktivacio), a tobbi 19 pedig sima felszin (kicsi).
        #
        #        atlag:  (19 * 0.1 + 1 * 5.0) / 20 = 0.34   <- majdnem eltunt
        #        max  :  max(0.1, ..., 5.0)       = 5.0     <- megmaradt
        #
        #    Range image-nel pont az elek (egy auto szele, egy fal vege)
        #    hordozzak az informaciot, ezert a max a helyes valasztas.
        #
        # A [0] azert kell, mert a max() (ertekek, indexek) part ad vissza.
        return out.max(dim=-1, keepdim=False)[0]      # (B, C, N)


class PositionalEncoding2D(nn.Module):
    """
    2D abszolut szinuszos poziciokodolas.

    MI A BAJ, AMIT MEGOLD: az encoderben a kepet "kilapitjuk" 2048 ponttá.
    Ezzel elveszne, hogy melyik pont HOL volt a kepen - a graf-reteg csak egy
    rendezetlen halmazt latna.

    A MEGOLDAS: minden poziciohoz hozzaadunk egy jellegzetes szinusz-koszinusz
    mintazatot, kulonbozo frekvenciakon. Olyan, mint egy binaris szam, csak
    folytonos: minden poziciónak egyedi "ujjlenyomata" lesz, es a kozeli
    poziciok ujjlenyomata hasonlo.

    ---------------------------------------------------------------------
    A MATEMATIKA RESZLETESEN
    ---------------------------------------------------------------------

    A keplet (a Transformer paperbol, "Attention is All You Need"):

        PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    ahol  pos = a pozicio (0, 1, 2, ... W-1)
          i   = a csatorna indexe
          d   = a csatornak szama

    A kodban a nevezot elore kiszamoljuk reciprokkent:

        inv_freq = 1 / 10000^(2i/d)

    tehat a szorzas (pos * inv_freq) ugyanaz, mint az osztas a keplettel.

    MIERT EZ A KEPLET? Harom tulajdonsag miatt:

    1. MINDEN POZICIONAK EGYEDI MINTAZATA VAN.
       Az inv_freq egy geometriai sorozat 1-tol 1/10000-ig. Minden csatorna
       mas frekvencian "rezeg":

           i = 0       -> inv_freq = 1        -> gyors rezges
           i = d/2     -> inv_freq = 1/100    -> kozepes
           i = d       -> inv_freq = 1/10000  -> nagyon lassu

       Ez pont olyan, mint egy binaris szam: a magas frekvenciak a also
       biteket kodoljak (finom felbontas), az alacsonyak a felsoket (durva
       felbontas). Csak itt folytonos, nem diszkret.

    2. A KOZELI POZICIOK KODJA HASONLO.
       A szinusz folytonos, tehat sin(5*f) es sin(6*f) kozel van egymashoz.
       Ez fontos: a halo igy tudja, hogy a 5. es 6. oszlop szomszedos.

    3. MIERT KELL SIN ES COS IS?
       Mert egyedul a szinusz nem egyertelmu: sin(30 fok) = sin(150 fok).
       A koszinusz viszont ezeket megkulonbozteti (cos(30) != cos(150)).
       A ket fuggveny EGYUTT egyertelmuen meghatarozza a szoget - ez ugyanaz
       az elv, mint az egysegkoron a (cos, sin) koordinatapar.

       Ezert osztjuk a csatornakat negy reszre: sin(x), cos(x), sin(y), cos(y).

    MERT PROBLEMA (tanitatlan halon): ennek a kodolasnak az amplitudoja 0.590,
    a tenyleges feature-e viszont csak 0.035 - vagyis 17x-esen elnyomja. Emiatt
    a kNN gyakorlatilag csak a poziciot "latja", es a valasztott szomszedok
    atlagos terbeli tavolsaga 1.8 cella (kodolas nelkul 36.1 lenne). Tanitas
    soran ez az arany eltolodhat; ha nem, a kodolas leskalazasa javithat.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # A csatornakat negy reszre osztjuk: sin(x), cos(x), sin(y), cos(y).
        # A ceil felfele kerekit, hogy 4-gyel nem oszthato csatornaszamnal is
        # jusson mindegyiknek - a felesleget a forward() vegen levagjuk.
        channels = int(np.ceil(channels / 4) * 2)

        # inv_freq = 1 / 10000^(2i/d), geometriai sorozat 1-tol 1/10000-ig.
        # A torch.arange(0, channels, 2) a 2i-t adja: 0, 2, 4, 6, ...
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))

    def forward(self, tensor):
        B, C, H, W = tensor.shape
        pos_x = torch.arange(W, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(H, device=tensor.device).type(self.inv_freq.type())

        # einsum("i,j->ij", a, b): kulso szorzat - a minden elemehez b minden
        # eleme. Eredmeny: (len(a), len(b)).
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)

        # sin es cos egyutt: ezek egyutt egyertelmuen meghatarozzak a szoget.
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(0).repeat(H, 1, 1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1).repeat(1, W, 1)
        emb = torch.cat((emb_x, emb_y), dim=-1).permute(2, 0, 1).unsqueeze(0)

        # A [:, :C] levagas azert kell, mert a felfele kerekites (ceil) miatt
        # tobb csatornat generaltunk, mint amennyi kell.
        return tensor + emb[:, :C, :, :].to(tensor.device)


class Stem(nn.Module):
    """
    Bemenet downsamplingje: (B, C, 64, 1024) -> (B, D, 16, 128).

    MIERT KELL: a graf-reteg minden pontra egy N x N tavolsagmatrixot szamol.
    N = 64*1024 = 65536 eseten ez 4.3 MILLIARD elem retegenkent -
    kezelhetetlen. N = 16*128 = 2048 eseten 4.2 millio, ami mar megy.

    MIERT ASZIMMETRIKUS (fuggolegesen /4, vizszintesen /8): mert a bemenet
    maga is az - 64 elevacios sor all szemben 1024 azimut oszloppal.

    ---------------------------------------------------------------------
    A KONVOLUCIO MERETKEPLETE
    ---------------------------------------------------------------------

        out = floor((in + 2*padding - kernel) / stride) + 1

    Levezetes: a kernel kozeppontja az elso ervenyes poziciotol az utolsoig
    csuszik. A padelt kep hossza (in + 2*padding); ebbol a kernel meg eppen
    (in + 2*padding - kernel + 1) kulonbozo helyre fer be. Ha stride-onkent
    lepunk, ezek kozul minden stride-adikat vesszuk - innen az osztas.

    Behelyettesitve a mi ertekeinkkel (kernel=3, padding=1):

        stride = 2:  out = (in + 2 - 3)/2 + 1 = (in - 1)/2 + 1 ~ in/2
        stride = 1:  out = (in + 2 - 3)/1 + 1 = in              (valtozatlan)

    Vagyis a (2,2) stride felez mindket tengelyen, az (1,2) csak
    vizszintesen. A meret alakulasa:

        (64, 1024) --(2,2)--> (32, 512) --(2,2)--> (16, 256) --(1,2)--> (16, 128)
           |                                                              |
           +---- fuggolegesen /4, vizszintesen /8 ------------------------+
    """

    def __init__(self, in_dim=1, out_dim=64):
        super().__init__()
        self.conv1 = CircularConv2d(in_dim, out_dim // 4, kernel_size=3, stride=(2, 2), padding=1)
        self.conv2 = CircularConv2d(out_dim // 4, out_dim // 2, kernel_size=3, stride=(2, 2), padding=1)
        self.conv3 = CircularConv2d(out_dim // 2, out_dim, kernel_size=3, stride=(1, 2), padding=1)

    def forward(self, x):
        # LeakyReLU es nem ReLU: a ReLU a negativ ertekeket pontosan 0-ra vagja,
        # es ott a gradiens is 0 - az ilyen neuron "meghalhat", soha tobbe nem
        # tanul. A LeakyReLU kis meredeksege (0.2) mindig hagy gradienst.
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        return x


# =============================================================================
# 3. ENCODER - graf-alapu
# =============================================================================

class Encoder(nn.Module):
    """
    Range image -> latens.

    EZ A REPO LENYEGI UJDONSAGA: a range image-et pont-halmazkent kezeli, es
    graf-konvolucioval tomoriti.

    Adatut:
        (B, 1, 64, 1024)
          -> Stem              -> (B, 64, 16, 128)
          -> PositionalEncoding
          -> flatten           -> (B, 64, 2048)     "pontfelho" alak
          -> GraphLayer x4     -> (B, 64, 2048)     dinamikus kNN minden retegben
          -> Conv1d projekcio  -> (B, 16, 2048)
          -> reshape           -> (B, 16, 16, 128)  <- a LATENS
    """

    def __init__(self, in_channels=1, z_channels=16, ch=64, k=20, **kwargs):
        super().__init__()
        self.k = k
        self.z_channels = z_channels

        self.stem = Stem(in_channels, ch)
        self.pos_enc = PositionalEncoding2D(ch)

        # Negy graf-reteg egymas utan. Mindegyik ujraszamolja a kNN-t, tehat a
        # szomszedsagok retegrol retegre valtozhatnak (dinamikus graf).
        self.layer1 = GraphLayer(ch, k)
        self.layer2 = GraphLayer(ch, k)
        self.layer3 = GraphLayer(ch, k)
        self.layer4 = GraphLayer(ch, k)

        # Conv1d kernel_size=1: pontonkenti linearis reteg, ami ch csatornabol
        # z_channels-t csinal. Ez a tenyleges szuk keresztmetszet.
        self.proj = nn.Conv1d(ch, z_channels, 1)

    def forward(self, x):
        # 1. Downsampling: (B, 1, 64, 1024) -> (B, 64, 16, 128)
        h = self.stem(x)

        # 2. Poziciokodolas hozzaadasa - a kovetkezo lepes elott KELL, mert
        #    utana mar nincs terbeli informacio.
        h = self.pos_enc(h)

        B, D, H, W = h.shape
        N = H * W                       # 16 * 128 = 2048 "pont"

        # 3. Kilapitas pont-halmaz alakra: (B, 64, 16, 128) -> (B, 64, 2048)
        h = h.view(B, D, N)

        # 4. Hierarchikus graf-kodolas
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)

        # 5. Vetites a latens dimenziora: (B, 64, 2048) -> (B, 16, 2048)
        z = self.proj(h)

        # 6. Vissza 2D alakra, hogy a konvolucios decoder tudjon vele dolgozni.
        return z.view(B, -1, H, W)      # (B, 16, 16, 128)


# =============================================================================
# 4. DECODER - ResNet + felskalazas
# =============================================================================
#
# A decoder NEM graf-alapu, es CSAK A TANITASHOZ kell: o adja a rekonstrukcios
# loss masik felet. Ha kesz a tanitas es az encodert feature extractorkent
# hasznalod, a decoder eldobhato.
# =============================================================================

def nonlinearity(x):
    """
    Swish (SiLU): x * sigmoid(x).

    Hasonlit a ReLU-ra, de folytonosan derivalhato - ettol simabb a tanulas.
    Peldak: swish(-2) = -0.238, swish(0) = 0, swish(2) = 1.762
    """
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    """
    GroupNorm: a csatornakat csoportokba osztja, es csoportonkent normalizal
    (nulla atlag, egyes szoras), majd egy tanulhato skalaval es eltolassal
    visszaallitja a szabadsagot:  y = (x - atlag)/szoras * gamma + beta

    MIERT GroupNorm es nem BatchNorm: a BatchNorm a batch osszes mintaja folott
    atlagol, tehat a viselkedese fugg a batch merettol, es maskepp mukodik
    tanitaskor mint kiertekeleskor. A GroupNorm egy mintan belul dolgozik,
    ezert batch-merettol fuggetlen es kiszamithatobb.
    """
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


# A felskalazashoz tartozo kernel- es padding-meretek stride-onkent.
# A kernel a stride ketszerese + 1 korul van, hogy a bilinearis interpolacio
# utani simitas atfogja az uj pixeleket.
UPSAMPLE_STRIDE2KERNEL = {(1, 2): (1, 5), (2, 2): (3, 3)}
UPSAMPLE_STRIDE2PAD = {(1, 2): (2, 2, 0, 0), (2, 2): (1, 1, 1, 1)}

# A ResnetBlock kernel-merethez tartozo padding, hogy a meret ne valtozzon.
UNIFORM_KERNEL2PAD = {(3, 3): (1, 1, 1, 1), (1, 4): (1, 2, 0, 0)}


class Upsample(nn.Module):
    """
    Felskalazas: bilinearis interpolacio + konvolucio.

    Az interpolacio "kitolti" a hianyzo pixeleket a szomszedokbol atlagolva -
    ettol viszont elmosodott lesz. A rakovetkezo konvolucio elesiti.
    """

    def __init__(self, in_channels, stride):
        super().__init__()
        self.stride = stride
        k = UPSAMPLE_STRIDE2KERNEL[stride]
        p = UPSAMPLE_STRIDE2PAD[stride]
        self.conv = CircularConv2d(in_channels, in_channels, kernel_size=k, padding=p)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.stride, mode='bilinear', align_corners=True)
        return self.conv(x)


class ResnetBlock(nn.Module):
    """
    Ket konvolucio + SKIP CONNECTION.

    A lenyeg a forward() vegen levo `return x + h`: a blokk NEM a kimenetet
    tanulja meg, hanem a VALTOZTATAST, amit a bemenethez hozza kell adni.

    MIERT JO EZ: mely haloknal a gradiens visszafele haladva egyre kisebb lesz
    ("eltuno gradiens"), es a korai retegek nem tanulnak. A +x egy
    "gyorsitosavot" ad: a gradiens ezen az uton valtozatlanul jut vissza.

    Sorrend: norm -> aktivacio -> konvolucio (ez a "pre-activation" ResNet).
    """

    def __init__(self, *, in_channels, out_channels=None, kernel_size=(3, 3), dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        pad = UNIFORM_KERNEL2PAD[kernel_size]

        self.norm1 = Normalize(in_channels)
        self.conv1 = CircularConv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=1, padding=pad)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CircularConv2d(out_channels, out_channels, kernel_size=kernel_size,
                                    stride=1, padding=pad)

        # Ha valtozik a csatornaszam, a skip aghoz is kell egy vetites -
        # kulonben az x + h osszeadas nem menne (mas alaku tenzorok).
        # 1x1 konvolucio: pontonkenti linearis reteg, terbeli keveredes nelkul.
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                          stride=1, padding=0)

    def forward(self, x):
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h        # <- a skip connection


class Decoder(nn.Module):
    """
    Latens -> range image.  (B, 16, 16, 128) -> (B, 1, 64, 1024)

    A felepites tobb "szintbol" all; minden szinten nehany ResnetBlock fut, es
    a szintek kozott egy Upsample duplazza a felbontast.

    A STRIDES INDEXELES CSAPDAJA (ez az eredeti kod legkellemetlenebb resze):

        stride = strides[i_level - 1] if i_level > 0 else None

    Ket dolog egyszerre:
      - az i_level = 0 szinten NINCS upsample
      - a listat EGGYEL ELTOLVA olvassa

    Kovetkezmeny: len(ch_mult) - 1 darab upsample fut, es a strides UTOLSO
    eleme SOHA nem kerul felhasznalasra (dummy).

    A mi konfiguracionkkal:
        ch_mult = (1, 2, 4, 4)
        strides = ((2,2), (2,2), (1,2), (1,1))
                                          ^^^^^ dummy

    A tenyleges felskalazas (forditott sorrendben, i_level = 3, 2, 1):
        (16,128) --(1,2)--> (16,256) --(2,2)--> (32,512) --(2,2)--> (64,1024)

    Ha elrontod, a kimenet nem 64x1024 lesz, es a loss shape hibaval elszall.
    """

    def __init__(self, *, ch, out_ch, ch_mult, strides, num_res_blocks,
                 z_channels, dropout=0.0, tanh_out=True, **kwargs):
        super().__init__()
        # A stride-hoz tartozo kernel-meret a ResnetBlock-okban.
        stride2kernel = {(2, 2): (3, 3), (1, 2): (1, 4)}

        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.tanh_out = tanh_out

        # A legmelyebb szint csatornaszama - innen indul a decoder.
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # A latenst felvisszuk a munkacsatorna-szamra.
        self.conv_in = CircularConv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # "Kozepso" blokkok a legkisebb felbontason, meg felskalazas elott.
        self.mid_block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid_block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # A szintek felepitese. Forditva megyunk (a legmelyebbtol a legfelso
        # felbontasig), es insert(0, ...)-tal rakjuk oket a listaba, hogy a
        # sorrend a vegen novekvo legyen.
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            stride = tuple(strides[i_level - 1]) if i_level > 0 else None
            kernel = stride2kernel[stride] if stride is not None else (1, 4)

            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         kernel_size=kernel, dropout=dropout))
                block_in = block_out

            level = nn.Module()
            level.block = block
            if stride is not None:
                level.upsample = Upsample(block_in, stride)
            self.up.insert(0, level)

        self.norm_out = Normalize(block_in)
        # Az utolso konvolucio viszi vissza 1 csatornara (a range ertekre).
        self.conv_out = CircularConv2d(block_in, out_ch, kernel_size=(1, 4),
                                       stride=1, padding=(1, 2, 0, 0))

    def forward(self, z):
        h = self.conv_in(z)

        h = self.mid_block_1(h)
        h = self.mid_block_2(h)

        # Felfele menet: minden szinten a blokkok, majd (a 0. kivetelevel)
        # egy felskalazas.
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        # tanh: a kimenetet [-1, 1]-be szoritja. Ez azert kell, mert a bemeneti
        # range image is ebben a tartomanyban van. Enelkul a halonak kulon meg
        # kellene tanulnia, hogy "ne menj 1 fole es -1 ala".
        if self.tanh_out:
            h = torch.tanh(h)
        return h


# =============================================================================
# 5. A TELJES MODELL
# =============================================================================

class LidarAE(nn.Module):
    """
    Range image autoencoder graf-alapu encoderrel.

    DETERMINISZTIKUS - nincs VAE, nincs mintavetel, nincs KL-tag.
    Ugyanaz a bemenet mindig ugyanazt a latenst adja.

    Hasznalat tanitashoz:
        model = LidarAE(**DDCONFIG)
        x_rec = model(x)
        loss = F.l1_loss(x, x_rec)

    Hasznalat RL-ben (csak az encoder kell):
        z = model.encode(x)        # (B, 16, 16, 128)
    """

    def __init__(self, ddconfig=None, learning_rate=1e-4, **kwargs):
        super().__init__()
        # A ddconfig egy dict a halo alakjaval. Megadhato kulcsszavakent is.
        cfg = dict(ddconfig) if ddconfig is not None else {}
        cfg.update(kwargs)

        self.learning_rate = learning_rate
        self.encoder = Encoder(**cfg)
        self.decoder = Decoder(**cfg)

    def encode(self, x):
        """
        Range image -> latens.  EZT hasznalja majd az RL.

        x      : (B, 1, 64, 1024) float, [-1, 1]
        return : (B, 16, 16, 128)

        FIGYELEM: a latens 16*16*128 = 32768 ertek. Ez laposítva nagysagrendekkel
        tobb, mint a kamera AE 64 elemu latense, es az SB3 MultiInputPolicy NEM
        normalizal - igy elnyomna a tobbi observationt (steer, throttle,
        waypointok). Az RL-be kotes elott csokkenteni kell.
        """
        return self.encoder(x)

    def decode(self, z):
        """Latens -> range image. RL futaskor NEM kell."""
        return self.decoder(z)

    def forward(self, x):
        """
        Teljes kor: kep -> latens -> rekonstrualt kep.

        A forward()-ot sosem hivod kozvetlenul! A modult hivod fuggvenykent
        (`model(x)`), mert az futtatja a PyTorch beakasztott hookjait is.
        """
        return self.decode(self.encode(x))

    def configure_optimizers(self):
        """
        Az optimalizalo: ez frissiti a sulyokat a gradiensek alapjan.

        Adam: adaptiv optimalizalo, ami parameterenkent kulon lepeskozt tart
        nyilvan, es figyelembe veszi a korabbi gradiensek atlagat is
        (momentum). Ezert kevesebb hangolast igenyel, mint a sima SGD.

        betas=(0.5, 0.9): mennyire simitsa a multbeli gradienseket. Az elso a
        gradiens atlagara, a masodik a negyzetere vonatkozik.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9))

    def init_from_ckpt(self, path):
        """
        Sulyok betoltese checkpointbol (tanitas folytatasahoz).

        map_location="cpu": eloszor a rendszermemoriaba toltjuk, onnan megy a
        GPU-ra. Igy akkor is mukodik, ha a checkpoint mas GPU-n keszult.
        """
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing "
              f"and {len(unexpected)} unexpected keys")
        if missing:
            print(f"Missing Keys: {missing}")
        if unexpected:
            print(f"Unexpected Keys: {unexpected}")


# =============================================================================
# 6. ALAPERTELMEZETT KONFIGURACIO
# =============================================================================

DDCONFIG = dict(
    in_channels=1,      # egy csatorna: a tavolsag (remission nelkul)
    out_ch=1,
    z_channels=16,      # a latens csatornaszama
    ch=64,              # a munkacsatorna-szam (encoder es decoder)
    ch_mult=(1, 2, 4, 4),
    strides=((2, 2), (2, 2), (1, 2), (1, 1)),   # az utolso dummy, lasd Decoder
    num_res_blocks=1,
    dropout=0.0,
    tanh_out=True,      # a bemenet [-1, 1], a kimenet is oda keruljon
    k=20,               # hany szomszed a graf-retegekben
)


# =============================================================================
# Onteszt: futtasd kozvetlenul (python lidar_ae.py), hogy ellenorizd az alakokat
# =============================================================================

if __name__ == "__main__":
    print("Alakok ellenorzese...")

    # Hamis pontfelho: egy "utca" ket fallal.
    rng = np.random.default_rng(0)
    elev = np.linspace(10, -25, 64) * np.pi / 180
    azim = np.linspace(-np.pi, np.pi, 1024, endpoint=False)
    E, A = np.meshgrid(elev, azim, indexing='ij')
    d = np.minimum(np.full_like(E, 60.0), 8.0 / np.maximum(np.abs(np.sin(A)), 1e-3))
    down = E < -0.02
    d[down] = np.minimum(d[down], 2.4 / np.abs(np.sin(E[down])))
    xyz = np.stack([d * np.cos(E) * np.cos(A),
                    d * np.cos(E) * np.sin(A),
                    d * np.sin(E)], -1).reshape(-1, 3).astype(np.float32)
    xyz = xyz[np.linalg.norm(xyz, axis=1) < 50]

    ri = points_to_range_image(xyz)
    print(f"  pontfelho    : {xyz.shape}")
    print(f"  range image  : {ri.shape}, kitoltottseg "
          f"{100 * (ri > -0.999).mean():.1f}%")

    model = LidarAE(DDCONFIG)
    x = torch.from_numpy(ri).unsqueeze(0)       # (1, 1, 64, 1024)
    with torch.no_grad():
        z = model.encode(x)
        x_rec = model(x)

    print(f"  bemenet      : {tuple(x.shape)}")
    print(f"  latens       : {tuple(z.shape)}  = {z.numel()} ertek")
    print(f"  rekonstrukcio: {tuple(x_rec.shape)}")
    assert x_rec.shape == x.shape, "ALAK ELTERES!"

    # Egy tanito lepes, hogy a backward is ellenorizve legyen.
    opt = model.configure_optimizers()
    xb = torch.from_numpy(np.stack([ri] * 2))
    loss = F.l1_loss(xb, model(xb))
    opt.zero_grad()
    loss.backward()
    opt.step()

    n = sum(p.numel() for p in model.parameters())
    print(f"  tanito lepes : OK, loss = {float(loss):.4f}")
    print(f"  parameterek  : {n / 1e6:.1f}M")
    print("OK")
