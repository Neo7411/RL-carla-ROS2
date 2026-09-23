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
# szabalyos sugarracsban paszaz (64 elevacios szog x 512 azimut). Ha ezt a
# racsot allitjuk helyre, semmi nem vesz el.
# =============================================================================

def pcd2range(pcd, size, fov, depth_range):
    """
    XYZ pontfelho -> range image (gombi projekcio).

    A harom terbeli koordinatabol KETTO a cella cime lesz, a harmadik
    (a tavolsag) pedig a cella tartalma:

        x, y  ->  azimut (yaw)    ->  OSZLOP  (0..255)
        z     ->  elevacio (pitch)->  SOR     (0..63)
        r     ->  a CELLA ERTEKE

    pcd         : (N, 3) float, a szenzor koordinatarendszereben
    size        : (H, W), nalunk (44, 256)
    fov         : (fov_up, fov_down) fokban, nalunk (-0.9, -25) - MERVE
    depth_range : (min, max) meterben, nalunk (1.0, 50.0) - MERVE

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

    # TAKARAS KEZELESE: tobb pont eshet ugyanabba a cellaba. Cellankent a
    # KOZELEBBI pont marad - ami fizikailag helyes, mert az takar. A
    # minimum.at cellankent minimumot vesz, rendezes nelkul (a korabbi
    # argsort + "az utolso iras nyer" ugyanezt adta, ~1.7x lassabban).
    flat = np.full(size[0] * size[1], np.inf, dtype=np.float32)
    np.minimum.at(flat, proj_y * size[1] + proj_x, depth.astype(np.float32, copy=False))

    # Ahol nem esett pont, ott -1.
    flat[np.isinf(flat)] = -1
    return flat.reshape(size)


def process_scan(range_img, depth_scale=5.68):
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

    depth_scale = 5.68, mert log2(50+1) = 5.672 - a mi hatotavunk.

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


def points_to_range_image(xyz, size=(44, 256), fov=(-0.9, -25.0),
                          depth_range=(1.0, 50.0), depth_scale=5.68):
    """A ket fenti lepes egyben: nyers XYZ -> (1, 44, 256) float32, [-1, 1].

    A HALO az eredeti TopoLiDM config szerint van (ch=64, ch_mult (1,2,2,4),
    strides [[1,2],[2,2],[2,2]], nrb=2, lr 4.5e-6), a GEOMETRIA viszont a MI
    szenzorunkhoz - a kettot nem szabad osszekeverni.

    A TopoLiDM KITTI-re (Velodyne HDL-64E) van hangolva: fov [3,-25],
    depth_range [1,56], scale 5.84. A mi CARLA-lidarunk MASIK szenzor, es a
    parametereit az ADATBOL mertuk ki (40 frame, 1.1M pont):

        elevacio    -25.00 .. +10.00 fok, pontosan 64 diszkret szinttel
        tavolsag      5.67 .. 50.00 m  (p99: 48.9)

    Ezert fov = (10, -25) es depth_range = (1, 50). A TopoLiDM [3,-25]-e
    levagna a felso 7 fokot - az adat ~5%-at, vagyis a magas targyakat
    (teherautok, tablak, fak koronaja).

    depth_scale = 5.68, mert log2(50+1) = 5.672 - igy a legtavolabbi pont
    pont 1.0-ra normalodik. (A TopoLiDM 5.84-e a sajat 56 m-es hatotavabol
    jon: log2(57) = 5.833.)

    A FELBONTAS A MI SZENZORUNKHOZ VAN MERVE, nem a KITTI-hez.

    A TopoLiDM 64x1024-et hasznal, de az a KITTI HDL-64E-re valo (~120 000
    pont/frame). A mi lidarunk 28 368 pontot ad - a SURUSEG 24%-a -, tehat
    ugyanaz a racs nalunk 60%-ban URES lenne, mig a KITTI-n teljesen tele van.

    Ket dolgot valtoztattunk, mindkettot MERES alapjan:

    1. AZ EG LEVAGASA. A +10 fokos sor csak 19.6%-ban telik meg, a -3.3 fok
       alattiak 77-78%-ban - a kep felso harmada az eg, ott nincs mit
       eltalalni. A fov_up -0.9 fokra vagva a sorok 64 -> 44.

    2. KISEBB RACS. Kevesebb cella, surubben kitoltve.

    MERVE (30 frame), es a "geom. hiba" a lenyeg - minden EREDETI pont
    atlagos tavolsaga a legkozelebbi MEGTARTOTT ponttol:

        H x W   fov_up   kitoltott   megtartott pont   geom. hiba
        64x512   +10.0      60.7%         71.9%          0.363 m
        48x384    +1.2      81.5%         60.3%          0.312 m
        48x320    +1.2      85.9%         52.9%          0.310 m
        44x256    -0.9      90.2%         42.5%          0.310 m   <- ez
        44x192    -0.9      92.9%         32.9%          0.306 m

    A 44x256 a pontok 57%-at eldobja, MEGIS kisebb a geometriai hibaja
    (0.310 m), mint a 64x512-e (0.363 m). Ennek az az oka, hogy a range
    image cellankent a LEGKOZELEBBI pontot tartja meg (takaras-kezeles) -
    amit eldob, az mogotte van, ugyanazon a feluleten. A <20 m-es, vezeteshez
    lenyeges pontoknal ugyanez az arany, nincs szelektiv veszteseg.

    A racs merete a halo stemjehez is illeszkedik: a 3 lepcso ((1,2),(2,2),
    (2,2)) miatt a magassagnak 4-gyel, a szelessegnek 8-cal oszthatonak kell
    lennie. 44/4 = 11, 256/8 = 32, tehat a latens (16, 11, 32).
    """
    return process_scan(pcd2range(xyz, size, fov, depth_range), depth_scale)


def range_to_points(range_img, fov=(-0.9, -25.0), depth_scale=5.68,
                    depth_range=(1.0, 50.0)):
    """A points_to_range_image inverze: (1,H,W) [-1,1] -> (N,3) XYZ.

    Megjelenitesre: a range image lapos csik, felulnezetbe vetitve viszont
    ranezesre ertelmezheto. A cellan beluli pozicio nem jon vissza (a cella
    kozepet vesszuk), ez ~0.2 m hiba. Az ures cellak kiesnek.
    """
    r = np.asarray(range_img, dtype=np.float32)
    if r.ndim == 3:
        r = r[0]
    h, w = r.shape

    fov_down = fov[1] / 180.0 * np.pi
    fov_range = abs(fov_down) + abs(fov[0] / 180.0 * np.pi)

    # process_scan inverze: [-1,1] -> meter
    depth = np.exp2((r + 1.0) * 0.5 * depth_scale) - 1.0
    rows, cols = np.nonzero(depth > depth_range[0])

    # pcd2range inverze: cellaindex -> szog (+0.5 = a cella kozepe)
    yaw = (2.0 * (cols + 0.5) / w - 1.0) * np.pi
    pitch = (1.0 - (rows + 0.5) / h) * fov_range - abs(fov_down)

    # gombi -> Descartes (a minusz a pcd2range yaw-elojelet forditja vissza)
    dep = depth[rows, cols]
    xy = dep * np.cos(pitch)
    return np.stack([xy * np.cos(-yaw), xy * np.sin(-yaw),
                     dep * np.sin(pitch)], axis=1).astype(np.float32)


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


class GraphLayer(nn.Module):
    """
    EdgeConv blokk: kNN -> el-feature -> 1x1 konvolucio -> max-pool.

    EZ AZ EGESZ GRAF-MEGKOZELITES MAGJA (DGCNN / EdgeConv). Jelolje x_i az
    i-edik pont jellemzovektorat, N(i) pedig a k legkozelebbi szomszedjat a
    JELLEMZOTERBEN (nem a 3D terben). Minden (i, j) elre keszul egy
    el-jellemzo, majd a szomszedok folott maximumot veszunk:

        e_ij  = h( x_j - x_i  ||  x_i )
        x'_i  = MAX over j in N(i) of e_ij

    MIERT A KULONBSEG (x_j - x_i) ES A KOZEPPONT IS?
    A relativ geometria a lenyeg: egy auto 10 m-re a pontjait 10.0, 10.1,
    10.2 tavolsagra adja, ugyanaz 30 m-re 30.0, 30.1, 30.2-re. Az abszolut
    ertekek masok, a KULONBSEGEK azonosak - igy a halo egyetlen mintazatot
    tanul az "auto" alakra, nem tavolsagonkent kulon. De a kozeppont abszolut
    erteke is kell, mert a kontextus szamit: egy 3 m-re levo akadaly mas
    jelentesu, mint egy 40 m-re levo.

    MIERT MAX ES NEM ATLAG? Mindketto permutacio-invarians (a kNN nem garantal
    sorrendet, tehat ez kotelezo), de a max a legerosebb jelet emeli ki: ha 20
    szomszedbol egy jelzi, hogy "itt el van" (5.0) es 19 sima felszin (0.1),
    az atlag 0.34 - majdnem eltunt -, a max 5.0. Range image-nel pont az elek
    hordozzak az informaciot.

    A graf DINAMIKUS: minden reteg ujraszamolja a kNN-t a SAJAT bemeneten,
    tehat a szomszedsagi viszonyok retegrol retegre valtoznak.
    """

    def __init__(self, channels, k=20):
        super().__init__()
        self.k = k
        # kernel_size=1: pontonkent es szomszedonkent fuggetlen linearis reteg,
        # nincs terbeli keveredes. A bemenet 2C (el-jellemzo), a kimenet C.
        # bias=False, mert utana BatchNorm jon a sajat eltolasaval.
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def knn(self, x):
        """k legkozelebbi szomszed a jellemzoterben. (B, C, N) -> (B, N, k)

        ||a - b||^2 = ||a||^2 - 2*(a.b) + ||b||^2 alapjan, mert igy az OSSZES
        pontpar tavolsaga egyetlen matrixszorzassal adodik - a GPU erre van
        optimalizalva, nem a pontparonkenti ciklusra. Az eredmeny a tavolsag
        NEGALTJA, mert a topk a legnagyobbakat adja, nekunk meg a legkisebb
        tavolsagok kellenek.

        Koltseg: N x N matrix batchenkent. A Stem utan N = 2048 (4.2 millio
        elem) meg belefer; a nyers kepen N = 65536 lenne (4.3 milliard).

        Minden pont elso szomszedja onmaga (a tavolsaga 0) - ez szandekos, a
        kozeppont sajat jellemzoje is kell az el-jellemzohoz.
        """
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        return (-xx - inner - xx.transpose(2, 1)).topk(k=self.k, dim=-1)[1]

    def forward(self, x):
        B, C, N = x.shape
        idx = self.knn(x)

        # Index-eltolas: az egeszet egy (B*N, C) tablava lapitjuk, ahol a
        # b-edik minta i-edik pontja a b*N + i poziciora kerul.
        idx = (idx + torch.arange(B, device=x.device).view(-1, 1, 1) * N).view(-1)

        x = x.transpose(2, 1).contiguous()                    # (B, N, C)
        neighbors = x.view(B * N, C)[idx].view(B, N, self.k, C)
        center = x.view(B, N, 1, C).expand(-1, -1, self.k, -1)

        # [szomszed - kozep || kozep] -> (B, 2C, N, k) a Conv2d-nek.
        edge = torch.cat((neighbors - center, center), dim=3)
        out = self.conv(edge.permute(0, 3, 1, 2).contiguous())

        # Max-pool a szomszedok folott: a k szomszedbol egy vektor.
        return out.max(dim=-1)[0]                             # (B, C, N)


class PositionalEncoding2D(nn.Module):
    """
    2D abszolut szinuszos poziciokodolas, TANULHATO sullyal.

    MI A BAJ, AMIT MEGOLD: az encoderben a kepet "kilapitjuk" 2048 pontta.
    Ezzel elveszne, hogy melyik pont HOL volt a kepen - a graf-reteg csak egy
    rendezetlen halmazt latna.

    A MEGOLDAS: minden poziciohoz hozzaadunk egy jellegzetes szinusz-koszinusz
    mintazatot, kulonbozo frekvenciakon (a Transformer paper keplete):

        PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Sin es cos egyutt kell, mert egyedul a szinusz nem egyertelmu
    (sin(30 fok) = sin(150 fok)); a ket fuggveny egyutt viszont mar igen -
    ugyanaz az elv, mint az egysegkoron a (cos, sin) par. Ezert osztjuk a
    csatornakat negy reszre: sin(x), cos(x), sin(y), cos(y).

    MIERT TANULHATO A SULY - EZ KIMERT HIBA VOLT:
    A kodolas amplitudoja fixen ~0.59, a Stem kimeneteinek szorasa viszont
    csak ~0.056, tehat a kodolas 10x-esen ELNYOMTA a tenyleges jellemzoket.
    Emiatt a kNN gyakorlatilag csak a poziciot latta:

        a valasztott szomszedok atlagos terbeli tavolsaga
            kodolas nelkul  : 29.6 cella
            fix kodolassal  :  1.8 cella      (a veletlen ~31.4 lenne)

    Vagyis a "dinamikus graf" egy fix 5x5-os ablakka fajult, es a negy
    EdgeConv reteg egy dragan szamolt konvoluciot vegzett - pont az veszett
    el, amiert a graf-megkozelites egyaltalan erdekes (hogy a tavoli, de
    HASONLO ALAKU reszeket is osszekosse).

    A `weight` kezdoerteke ezert 0.03: ezzel a kodolas/jellemzo arany ~0.3,
    ahol a kNN mar a jellemzoket is latja, de a poziciot sem hagyja figyelmen
    kivul. Mivel a Stem tanul, ez az arany menet kozben elcsuszna - a
    tanulhato suly engedi a halonak, hogy maga allitsa be.
    """

    def __init__(self, channels, init_weight=0.03):
        super().__init__()
        # A csatornakat negy reszre osztjuk: sin(x), cos(x), sin(y), cos(y).
        # A ceil felfele kerekit, hogy 4-gyel nem oszthato csatornaszamnal is
        # jusson mindegyiknek - a felesleget a forward() vegen levagjuk.
        c = int(np.ceil(channels / 4) * 2)

        # inv_freq = 1 / 10000^(2i/d), geometriai sorozat 1-tol 1/10000-ig:
        # minden csatorna mas frekvencian "rezeg", mint egy folytonos binaris
        # szam. A magas frekvenciak a finom, az alacsonyak a durva felbontast
        # kodoljak, es a kozeli poziciok kodja hasonlo marad.
        inv_freq = 1.0 / (10000 ** (torch.arange(0, c, 2).float() / c))

        # register_buffer: a modell allapotanak resze (menti/tolti a
        # checkpoint, koveti a .to(device)-ot), de NEM tanulhato parameter.
        # A regi kod sima attributumkent tartotta, ami CPU-n ragadt volna.
        self.register_buffer("inv_freq", inv_freq)
        self.weight = nn.Parameter(torch.tensor(float(init_weight)))

    def forward(self, tensor):
        C, H, W = tensor.shape[1:]
        pos_x = torch.arange(W, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(H, device=tensor.device, dtype=self.inv_freq.dtype)

        # Kulso szorzat: minden poziciohoz minden frekvencia.
        sin_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)

        emb_x = torch.cat((sin_x.sin(), sin_x.cos()), -1).unsqueeze(0).expand(H, -1, -1)
        emb_y = torch.cat((sin_y.sin(), sin_y.cos()), -1).unsqueeze(1).expand(-1, W, -1)
        # A [:C] levagas a ceil-bol eredo tobbletcsatornakat dobja el.
        emb = torch.cat((emb_x, emb_y), -1).permute(2, 0, 1).unsqueeze(0)[:, :C]

        return tensor + self.weight * emb


# =============================================================================
# 3. ENCODER - graf-alapu
# =============================================================================

class Encoder(nn.Module):
    """
    Range image -> latens.

    EZ A REPO LENYEGI UJDONSAGA: a range image-et pont-halmazkent kezeli, es
    graf-konvolucioval tomoriti.

    Adatut:
        (B, 1, 44, 256)
          -> stem (3 konvolucio) -> (B, ch, 11, 32)
          -> PositionalEncoding
          -> flatten             -> (B, 64, 2048)   "pontfelho" alak
          -> GraphLayer x4       -> (B, 64, 2048)   dinamikus kNN retegenkent
          -> Conv1d projekcio    -> (B, 16, 2048)
          -> reshape             -> (B, 16, 16, 128)  <- a LATENS

    MIERT KELL A STEM DOWNSAMPLING: a graf-reteg N x N tavolsagmatrixot szamol.
    A nyers kepen N = 44*256 = 11264, ami 127 MILLIO elem retegenkent -
    kezelhetetlen. A stem utan N = 16*128 = 2048, vagyis 4.2 millio.

    A konvolucio meretkeplete out = floor((in + 2*pad - kernel)/stride) + 1
    szerint (kernel=3, pad=1) a (2,2) stride felez mindket tengelyen:

        (44,256) --(1,2)--> (44,128) --(2,2)--> (22,64) --(2,2)--> (11,32)
    """

    def __init__(self, in_channels=1, z_channels=16, ch=64, k=20, **kwargs):
        super().__init__()
        # LeakyReLU es nem ReLU: a ReLU a negativ ertekeket 0-ra vagja, es ott
        # a gradiens is 0 - az ilyen neuron "meghalhat". A LeakyReLU kis
        # meredeksege mindig hagy gradienst.
        # A stem lepcsoi az EREDETI TopoLiDM strides-aival egyeznek:
        # [[1,2],[2,2],[2,2]] - az elso csak VIZSZINTESEN felez, mert a range
        # image szeles es lapos (64 sor, 1024 oszlop).
        #
        #     (64,1024) --(1,2)--> (64,512) --(2,2)--> (32,256) --(2,2)--> (16,128)
        #
        # A dekoder ugyanezeket forditva jatssza vissza, igy a kimenet alakja
        # pontosan a bemenete.
        self.stem = nn.Sequential(
            CircularConv2d(in_channels, ch // 4, 3, stride=(1, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            CircularConv2d(ch // 4, ch // 2, 3, stride=(2, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            CircularConv2d(ch // 2, ch, 3, stride=(2, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pos_enc = PositionalEncoding2D(ch)

        # Negy graf-reteg egymas utan. Mindegyik ujraszamolja a kNN-t, tehat a
        # szomszedsagok retegrol retegre valtozhatnak (dinamikus graf).
        self.layers = nn.ModuleList([GraphLayer(ch, k) for _ in range(4)])

        # Conv1d kernel_size=1: pontonkenti linearis reteg ch -> z_channels.
        self.proj = nn.Conv1d(ch, z_channels, 1)

    def forward(self, x):
        # A poziciokodolas a kilapitas ELOTT kell: utana mar nincs terbeli
        # informacio, a graf-reteg csak egy rendezetlen halmazt latna.
        h = self.pos_enc(self.stem(x))

        B, D, H, W = h.shape
        h = h.view(B, D, H * W)         # (B, 64, 2048) "pontfelho" alak

        for layer in self.layers:
            h = layer(h)

        # Vissza 2D alakra, hogy a konvolucios decoder tudjon vele dolgozni.
        return self.proj(h).view(B, -1, H, W)       # (B, 16, 16, 128)


# =============================================================================
# 4. DECODER - ResNet + felskalazas
# =============================================================================
#
# A decoder NEM graf-alapu, es CSAK A TANITASHOZ kell: o adja a rekonstrukcios
# loss masik felet. Ha kesz a tanitas es az encodert feature extractorkent
# hasznalod, a decoder eldobhato.
# =============================================================================

# Kernel es padding a felskalazashoz, illetve a ResnetBlock-hoz. Az upsample
# kernelje a stride ketszerese korul van, hogy a bilinearis interpolacio utani
# simitas atfogja az uj pixeleket; a ResnetBlock paddingje meretartó.
UPSAMPLE_KERNEL_PAD = {(1, 2): ((1, 5), (2, 2, 0, 0)), (2, 2): ((3, 3), (1, 1, 1, 1))}
RESNET_KERNEL2PAD = {(3, 3): (1, 1, 1, 1), (1, 4): (1, 2, 0, 0)}


def norm(channels, num_groups=32):
    """GroupNorm: csoportonkent nulla atlagra es egyes szorasra normal, majd
    tanulhato gamma/beta parral visszaadja a szabadsagot.

    MIERT NEM BatchNorm: az a batch osszes mintaja folott atlagol, tehat fugg a
    batch merettol, es maskepp mukodik tanitaskor mint kiertekeleskor. A
    GroupNorm egy mintan belul dolgozik, ezert kiszamithatobb.

    A csoportszamot leszoritjuk, hogy (a) ossza a csatornaszamot, es (b)
    csoportonkent legalabb 4 csatorna maradjon. Enelkul kis `ch` eseten
    csoportonkent egyetlen csatorna jutna, ami InstanceNorma fajulna - ott
    nincs mit atlagolni a csatornak kozott.
    """
    while channels % num_groups or channels // num_groups < 4:
        num_groups //= 2
        if num_groups == 1:
            break
    return nn.GroupNorm(num_groups, channels, eps=1e-6, affine=True)


def swish(x):
    """x * sigmoid(x). Mint a ReLU, de folytonosan derivalhato - simabb tanulas."""
    return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):
    """
    Ket konvolucio + SKIP CONNECTION, pre-activation sorrendben
    (norm -> aktivacio -> konvolucio).

    A lenyeg a forward() vegen levo `x + h`: a blokk nem a kimenetet tanulja
    meg, hanem a VALTOZTATAST, amit a bemenethez hozza kell adni. Mely haloknal
    a gradiens visszafele haladva egyre kisebb lesz ("eltuno gradiens"); a +x
    egy gyorsitosavot ad, amin valtozatlanul jut vissza.
    """

    def __init__(self, in_channels, out_channels=None, kernel_size=(3, 3), dropout=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        pad = RESNET_KERNEL2PAD[kernel_size]

        self.block = nn.Sequential(
            norm(in_channels),
            nn.SiLU(),
            CircularConv2d(in_channels, out_channels, kernel_size, stride=1, padding=pad),
            norm(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            CircularConv2d(out_channels, out_channels, kernel_size, stride=1, padding=pad),
        )
        # Csatornavaltasnal a skip aghoz is kell egy 1x1 vetites, kulonben az
        # osszeadas nem menne (mas alaku tenzorok).
        self.shortcut = (nn.Conv2d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class Upsample(nn.Module):
    """Bilinearis interpolacio + konvolucio: az interpolacio kitolti a hianyzo
    pixeleket (de elmosodottan), a konvolucio utana elesiti."""

    def __init__(self, channels, stride):
        super().__init__()
        self.stride = stride
        kernel, pad = UPSAMPLE_KERNEL_PAD[stride]
        self.conv = CircularConv2d(channels, channels, kernel, padding=pad)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.stride, mode="bilinear", align_corners=True)
        return self.conv(x)


class Decoder(nn.Module):
    """
    Latens -> range image.  (B, 16, 16, 128) -> (B, 1, 64, 512)

    Szintekbol all: minden szinten nehany ResnetBlock fut, a szintek kozott egy
    Upsample noveli a felbontast. A legmelyebb szinttol a legfelsoig haladunk,
    tehat a `ch_mult` es a `strides` is VISSZAFELE olvasodik.

    A mi konfiguracionkkal (ch_mult = (1, 2, 4), strides eleje):
        (16,128) --(2,2)--> (32,256) --(2,2)--> (64,512)

    A legfelso szint utan nincs felskalazas, ezert a `strides` utolso eleme
    (az (1,1) a DDCONFIG-ban) sosem kerul felhasznalasra - dummy.
    """

    def __init__(self, *, ch, out_ch, ch_mult, strides, num_res_blocks,
                 z_channels, dropout=0.0, tanh_out=True, **kwargs):
        super().__init__()
        stride2kernel = {(2, 2): (3, 3), (1, 2): (1, 4)}
        self.tanh_out = tanh_out

        block_in = ch * ch_mult[-1]

        # A latenst felvisszuk a munkacsatorna-szamra, majd ket blokk fut a
        # legkisebb felbontason, meg felskalazas elott.
        self.conv_in = CircularConv2d(z_channels, block_in, 3, stride=1, padding=1)
        self.mid = nn.Sequential(
            ResnetBlock(block_in, dropout=dropout),
            ResnetBlock(block_in, dropout=dropout),
        )

        # A szintek a vegrehajtas sorrendjeben (legmelyebbtol felfele): igy nem
        # kell sem forditott indexeles, sem insert(0, ...).
        levels = []
        for i_level in reversed(range(len(ch_mult))):
            # A szint UTAN kovetkezo felskalazas; a legfelso szint utan nincs.
            stride = tuple(strides[i_level - 1]) if i_level > 0 else None
            kernel = stride2kernel[stride] if stride is not None else (1, 4)

            block_out = ch * ch_mult[i_level]
            stage = [ResnetBlock(block_in if i == 0 else block_out, block_out,
                                 kernel_size=kernel, dropout=dropout)
                     for i in range(num_res_blocks + 1)]
            block_in = block_out

            if stride is not None:
                stage.append(Upsample(block_in, stride))
            levels.append(nn.Sequential(*stage))

        self.up = nn.Sequential(*levels)

        self.norm_out = norm(block_in)
        # Az utolso konvolucio viszi vissza 1 csatornara (a range ertekre).
        self.conv_out = CircularConv2d(block_in, out_ch, kernel_size=(1, 4),
                                       stride=1, padding=(1, 2, 0, 0))

    def forward(self, z):
        h = self.up(self.mid(self.conv_in(z)))
        h = self.conv_out(swish(self.norm_out(h)))
        # tanh: a bemeneti range image is [-1, 1]-ben van, igy a halonak nem
        # kell kulon megtanulnia, hogy ne menjen a tartomanyon kivulre.
        return torch.tanh(h) if self.tanh_out else h


# =============================================================================
# 5. A TELJES MODELL
# =============================================================================

class LidarAE(nn.Module):
    """
    Range image autoencoder graf-alapu encoderrel.

    DETERMINISZTIKUS - nincs VAE, nincs mintavetel, nincs KL-tag.
    Ugyanaz a bemenet mindig ugyanazt a latenst adja.

    Hasznalat tanitashoz:
        model = LidarAE(DDCONFIG)
        loss = model.loss(x)

    Hasznalat RL-ben (csak az encoder kell):
        z = model.encode(x)        # (B, z_channels, 16, 128) jellemzoterkep

    A kimenet TERBELI jellemzoterkep, nem vektor. Az RL sajat CNN
    feature-extractorral dolgozza fel (SB3 BaseFeaturesExtractor) - igy az
    extractor a JUTALOMRA optimalizal, nem a rekonstrukciora.
    """

    def __init__(self, ddconfig=None, learning_rate=1e-4,
                 empty_weight=0.25, **kwargs):
        super().__init__()
        # A ddconfig egy dict a halo alakjaval. Megadhato kulcsszavakent is.
        cfg = dict(ddconfig) if ddconfig is not None else {}
        cfg.update(kwargs)

        self.learning_rate = learning_rate
        # Az URES (-1) cellak sulya a lossban. 1.0 = nincs sulyozas.
        self.empty_weight = empty_weight
        self.encoder = Encoder(**cfg)
        self.decoder = Decoder(**cfg)

        # NINCS BOTTLENECK. A graf-encoder terbeli kimenete MAGA a latens.
        #
        # Korabban egy Linear-par (majd szelesseg-conv) vitte le 256-ra.
        # MERVE (1200 lepes, 600 train frame, L1 a foglalt teruleten):
        #
        #   szukites          L1 foglalt   param
        #   Linear -> 128       0.1061     18.0M
        #   Linear -> 2048      0.1318    135.5M
        #   szelesseg-conv      0.1064      3.4M
        #   NINCS               0.0751      1.2M   <- ez
        #
        # A szukites nelkuli ag FELEANNYI hibat ad, nyolcad annyi
        # parameterbol. Ugyanez jott ki a bev_ae-nel is: a latens MERETE nem
        # szamit, a TERBELI SZERKEZET elvesztese a problema.
        #
        # Az RL ezt a jellemzoterkepet kapja, es sajat CNN feature-extractorral
        # dolgozza fel - igy az extractor a JUTALOMRA optimalizal.
        z_ch = cfg["z_channels"]
        self.spatial = (z_ch, 16, 128)
        self.latent_shape = self.spatial

    def encode(self, x):
        """
        Range image -> TERBELI jellemzoterkep.  EZT hasznalja majd az RL.

        x      : (B, 1, 64, 512) float, [-1, 1]
        return : (B, z_channels, 16, 128)

        Nincs vektorra lapitas - lasd az __init__ tablazatat.
        """
        return self.encoder(x)

    def decode(self, z):
        """Jellemzoterkep -> range image. RL futaskor NEM kell."""
        return self.decoder(z)

    def forward(self, x):
        """Teljes kor: kep -> latens -> rekonstrualt kep."""
        return self.decode(self.encode(x))

    def loss(self, x, parts=False):
        """L1 a range image-en.

        MIERT NEM MSE: a kep nagy resze URES (-1), a tobbi valodi tavolsag -
        ket modusz kozott eles hatarokkal. Az MSE ilyenkor a "biztonsagos
        kozepet" jutalmazza, ezert ELMOSSA a targyak hataret. Az L1 optimuma
        a median, ami ketmoduszu adatnal az egyik valodi modusz - eles marad.

        MIERT NINCS SULYOZAS az ures teruletekre: kimerve. A kep 35%-a ures,
        es felmerult, hogy azok elnyomjak a valodi tavolsagok hibajat (a
        bev_ae-nel pont ez volt a baj). Itt viszont a sulyozas NETTO rontott
        (300 lepes, L1 foglalt / L1 ures):

            empty_weight 1.0 (nincs)   0.162 / 0.642
            empty_weight 0.5           0.104 / 0.780
            empty_weight 0.2           0.084 / 0.931

        A foglalt hiba javul, de az ures annyira romlik, hogy a rekonstrukcio
        hasznalhatatlan lesz - a modell nem tudja, hol VEGZODIK egy targy,
        pedig az RL-nek pont az szamit.

        parts=True : (loss, dict) - kulon az ures es a foglalt teruleten.
            Egyetlen szam elrejti, ha a modell csak az ures/nem-ures hatart
            tanulta meg, a geometriat nem.
        """
        rec = self(x)

        # SULYOZOTT L1. A kep 35%-a URES (-1), es a sulyozatlan L1-ben ez adta
        # a loss ~85%-at: 0.35 * 0.633 = 0.222 a 0.259-bol. Merve, 4 epoch:
        #
        #   epoch   teljes    l1_occupied   l1_empty
        #     1     0.2660      0.060        0.646
        #     2     0.2588      0.056        0.633
        #     3     0.2544      0.054        0.624
        #     4     0.2503      0.055        0.611
        #
        # Az l1_occupied a 3. epochra beallt (0.054), es a 4.-nel vissza is
        # ment - vagyis a halo mar csak az URES teruleteket csiszolta, 12
        # perc/epoch aron. A gradiens rossz helyre ment.
        #
        # A dekoder tanh-ja sosem ad pontos -1-et, tehat az ures cellakon
        # marad egy le nem vihato reziduum - ugyanaz, mint a bev_ae-nel a
        # Sigmoid. Ott a maszkolas hozta a 0.82 -> 0.98-as ugrast.
        occ = (x > -0.999).float()
        w = self.empty_weight + (1.0 - self.empty_weight) * occ
        total = ((rec - x).abs() * w).sum() / w.sum()
        if not parts:
            return total
        with torch.no_grad():
            occ = x > -0.999
            err = (rec - x).abs()
            occupied = float(err[occ].mean()) if bool(occ.any()) else 0.0
            empty = float(err[~occ].mean()) if bool((~occ).any()) else 0.0
        return total, {"l1": float(total), "l1_occupied": occupied,
                       "l1_empty": empty, "occupied_frac": float(occ.float().mean())}

    def configure_optimizers(self):
        """Adam. A betas=(0.5, 0.9) a szokasosnal gyorsabban felejti a multbeli
        gradienseket - generativ modelleknel bevett."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                betas=(0.5, 0.9))

    def init_from_ckpt(self, path):
        """Sulyok betoltese checkpointbol (tanitas folytatasahoz)."""
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
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

# A `ch` ES a `num_res_blocks` a DECODERT szabja meg, es MERVE az a draga resz,
# nem a graf. Egy tanito lepesbol (batch 8, fwd+bwd) 106 ms az encoder es
# 615 ms a decoder: a legfelso szintje a teljes 64x512 felbontason dolgozik,
# ami mintankent 4.19M elem blokkonkent.
#
# A KONFIGURACIO AZ EREDETI TopoLiDM REPOBOL VALO (IRMVLab/TopoLiDM,
# configs/autoencoder/kitti/autoencoder_c2_p4_topo.yaml):
#
#     base_learning_rate: 4.5e-6      <- EZ A LENYEG
#     z_channels: 16,  in_channels: 1,  out_ch: 1
#     ch: 64,  ch_mult: [1,2,2,4],  strides: [[1,2],[2,2],[2,2]]
#     num_res_blocks: 2,  attn_levels: [],  dropout: 0.0
#     batch_size: 32
#     dataset: size [64,1024], fov [3,-25], depth_range [1,56], scale 5.84
#
# Korabban sajat kezzel hangolt, kisebb valtozatot hasznaltunk (ch=16,
# ch_mult (1,4,8), nrb=0, 64x512-es kep, lr=1e-4). Az gyorsabb volt
# (10.7 vs ~25 perc/epoch), de a loss oszcillalt es alig csokkent - az LR
# huszonketszerese volt az eredetinek.
#
# A sajat meresek a KIS halon (mar nem ervenyesek, csak referenciakent):
#     ch=32, mult(1,2,4)    47.2 f/s   12.4 perc/epoch   l1_occ 0.1006
#     ch=16, mult(1,2,4)    84.1 f/s    6.9 perc/epoch   l1_occ 0.1154
#     ch=16, mult(1,4,8)    56.0 f/s   10.4 perc/epoch   l1_occ 0.0988
#     ch=8,  mult(1,2,4)   128.2 f/s    4.6 perc/epoch   l1_occ 0.1278
DDCONFIG = dict(
    in_channels=1,      # egy csatorna: a tavolsag (remission nelkul)
    out_ch=1,
    z_channels=16,      # a latens csatornaszama
    ch=32,              # a munkacsatorna-szam (aranyosan a felezett kephez)
    ch_mult=(1, 2, 2, 4),
    # Az ELSO lepes csak VIZSZINTESEN felez (1024 -> 512), mert a range image
    # szeles es lapos: 64 sor, 1024 oszlop. Csak utana jon a ketiranyu felezes.
    strides=((1, 2), (2, 2), (2, 2)),
    num_res_blocks=2,
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
    azim = np.linspace(-np.pi, np.pi, 512, endpoint=False)
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
    x = torch.from_numpy(ri).unsqueeze(0)       # (1, 1, 64, 512)
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
    loss, parts = model.loss(xb, parts=True)
    opt.zero_grad()
    loss.backward()
    opt.step()

    n = sum(p.numel() for p in model.parameters())
    print(f"  tanito lepes : OK, loss = {float(loss):.4f}")
    print(f"  bontas       : foglalt {parts['l1_occupied']:.4f}  "
          f"ures {parts['l1_empty']:.4f}  "
          f"(a kep {100 * parts['occupied_frac']:.0f}%-a foglalt)")
    print(f"  parameterek  : {n / 1e6:.1f}M")
    print("OK")
