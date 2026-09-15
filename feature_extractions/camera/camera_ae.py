import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class CameraAutoEncoder(L.LightningModule):
    def __init__(self, latent_dim: int = 256, base_channels: int = 32, lr: float = 1e-3,
                 latent_scale: float = 4.0):
        super().__init__()

        self.save_hyperparameters()

        # ------------------------------------------------------------------
        # ENCODER: (3, 80, 160) -> (latent_dim,)
        # ------------------------------------------------------------------
        # Minden Conv2d(kernel=4, stride=2, padding=1) blokk PONTOSAN felezi a
        # terbeli meretet. A keplet:  out = floor((in + 2*pad - kernel)/stride) + 1
        # behelyettesitve: (in + 2 - 4)/2 + 1 = in/2
        #
        # Ezert kovethetjuk vegig fejben a meretet:
        #   bemenet:      (3,   80, 160)
        #   conv1 utan:   (C,   40,  80)
        #   conv2 utan:   (2C,  20,  40)
        #   conv3 utan:   (4C,  10,  20)
        #   conv4 utan:   (8C,   5,  10)
        #
        # A csatornaszam kozben duplazodik. Ez a szokasos CNN "trade": terbeli
        # felbontast vesztunk, cserebe egyre absztraktabb jellemzoket nyerunk.
        # Az elso retegek eleket es szineket latnak, az utolsok mar olyasmit,
        # hogy "ut", "jarda", "auto".
        C = base_channels
        self.encoder = nn.Sequential(
            # (3, 80, 160) -> (C, 40, 80)
            nn.Conv2d(3, C, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.LeakyReLU(0.2, inplace=True),

            # (C, 40, 80) -> (2C, 20, 40)
            nn.Conv2d(C, C * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (2C, 20, 40) -> (4C, 10, 20)
            nn.Conv2d(C * 2, C * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (4C, 10, 20) -> (8C, 5, 10)
            nn.Conv2d(C * 4, C * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Az utolso conv utani alak. Ezt eltaroljuk, mert a decodernek pontosan
        # ebbe kell visszaalakitania a lapos latent vektort.
        self.enc_out_shape = (C * 8, 5, 10)
        flat_dim = C * 8 * 5 * 10  # base_channels=32 eseten: 256*5*10 = 12800

        # A conv stack kimenete meg egy 3D "kockat" ad (csatorna x magassag x
        # szelesseg). Kilapitjuk egy vektorra, es egy linearis reteggel
        # levisszuk latent_dim-re. Ez a tenyleges szuk keresztmetszet.
        self.fc_encode = nn.Linear(flat_dim, latent_dim)

        # ------------------------------------------------------------------
        # DECODER: (latent_dim,) -> (3, 80, 160)
        # ------------------------------------------------------------------
        # Pontosan az encoder tukorkepe. Eloszor egy Linear visszafujja a
        # latent vektort flat_dim meretre, aztan ConvTranspose2d retegek
        # (ugyanazokkal a kernel/stride/padding ertekekkel) minden lepesben
        # DUPLAZZAK a terbeli meretet.
        self.fc_decode = nn.Linear(latent_dim, flat_dim)

        self.decoder = nn.Sequential(
            # (8C, 5, 10) -> (4C, 10, 20)
            nn.ConvTranspose2d(C * 8, C * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C * 4),
            nn.ReLU(inplace=True),

            # (4C, 10, 20) -> (2C, 20, 40)
            nn.ConvTranspose2d(C * 4, C * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C * 2),
            nn.ReLU(inplace=True),

            # (2C, 20, 40) -> (C, 40, 80)
            nn.ConvTranspose2d(C * 2, C, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),

            # (C, 40, 80) -> (3, 80, 160)
            nn.ConvTranspose2d(C, 3, kernel_size=4, stride=2, padding=1, bias=False),
            # Sigmoid: a kimenetet [0, 1]-be szoritja. Ez azert kell, mert a
            # bemeneti kepeket is [0, 1]-re normalizaljuk (uint8 / 255).
            # Ha itt nem lenne aktivacio, a halo tetszoleges szamot adhatna
            # vissza, es feleslegesen kene megtanulnia, hogy "0 ala es 1 fole
            # ne menj".
            nn.Sigmoid(),
        )

    # ----------------------------------------------------------------------
    # A halo hasznalata
    # ----------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Kep -> latent vektor.

        EZT fogja hivni az RL kod: a CARLA-tol jott kepbol csinal egy
        latent_dim hosszu vektort, amit beteszunk az SAC observation-be.

        x     : (B, 3, 80, 160) float tensor, [0, 1] tartomanyban
        return: (B, latent_dim), -latent_scale .. +latent_scale tartomanyban
        """
        h = self.encoder(x)
        # flatten(1): a 0. dim (batch) marad, minden mast egy dimenzioba huz.
        # (B, 8C, 5, 10) -> (B, 8C*5*10)
        h = h.flatten(1)

        # tanh + skala: a latentet -latent_scale..+latent_scale koze szoritja.
        #
        # Ez NEM kozmetika. A latent az RL-ben egy Dict observation resze, es az
        # SB3 MultiInputPolicy nem normalizal - csak osszefuzi a jeleket egy
        # vektorra. Korlat nelkul a latent -121..144 kozott mozgott, mikozben a
        # tobbi jel 0..1 (steer, throttle), +-3.14 (angle) es +-50 (waypoints)
        # nagysagrendu volt: a latent egyszeruen elnyomta az osszes tobbit, es
        # az agens nem tanult (a jutalom monoton csokkent).
        #
        # A regi VAE-nal ez azert nem volt gond, mert ott a KL-tag tartotta a
        # latentet +-4 korul. Itt nincs ilyen kenyszer, ezert kell explicit.
        return self.hparams.latent_scale * torch.tanh(self.fc_encode(h))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Latent vektor -> kep.

        Tanitas kozben kell (hogy legyen mit osszehasonlitani a bemenettel),
        es akkor, ha kivancsi vagy, mit "lat" a halo egy adott latentbol.
        Az RL futasakor NEM hasznaljuk - ott csak az encode kell.

        z     : (B, latent_dim)
        return: (B, 3, 80, 160), [0, 1] tartomanyban
        """
        h = self.fc_decode(z)
        # A Linear egy lapos vektort ad; vissza kell hajtogatni 3D alakra,
        # hogy a ConvTranspose retegek tudjanak vele dolgozni.
        # A -1 azt jelenti: "a batch meretet szamold ki magad".
        h = h.view(-1, *self.enc_out_shape)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Teljes kor: kep -> latent -> rekonstrualt kep.

        A forward() a PyTorch modulok "fo" metodusa. Sosem hivod kozvetlenul
        (`model.forward(x)`), hanem a modult hivod fuggvenykent: `model(x)`.
        A kulonbseg nem kozmetikai - a `model(x)` alak futtatja a beakasztott
        hookokat is, amikre a Lightning es a profiler epul.
        """
        return self.decode(self.encode(x))

    # ----------------------------------------------------------------------
    # Lightning tanitasi logika
    # ----------------------------------------------------------------------
    #
    # Innentol jon az, amit sima PyTorchban kezzel kellene megirni. Sima
    # torchban egy epoch igy nezne ki:
    #
    #     model.train()
    #     for x, _ in train_loader:
    #         x = x.to(device)              # <- kezi eszkozkezeles
    #         optimizer.zero_grad()         # <- kezi gradiens torles
    #         out = model(x)
    #         loss = F.mse_loss(out, x)
    #         loss.backward()               # <- kezi backprop
    #         optimizer.step()              # <- kezi sulyfrissites
    #
    # A Lightning ebbol pontosan azt a ket sort kerdezi meg toled, ami
    # tenyleg a TE dontesed (mi a loss), a tobbit o intezi. A .to(device),
    # a zero_grad(), a backward(), a step(), a model.train()/eval() valtas,
    # a checkpoint mentes, a progress bar, a TensorBoard logolas - mind
    # automatikus. Ezert rovidebb es ezert nehezebb elrontani.

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        """
        A kozos resz train/val/test-hez. Mindharom ugyanazt csinalja, csak
        mas neven logol, ezert nem irjuk le haromszor.

        A `batch` innen jon: a DataLoader ad egy (kepek, cimkek) part, mert
        ImageFolder datasetet hasznalunk. A cimke minket nem erdekel (nincs
        is ertelmes cimke), ezert `_`-ba dobjuk.
        """
        x, _ = batch
        x_hat = self(x)

        # MSE = atlagos negyzetes elteres pixelenkent. Ez a rekonstrukcios
        # hiba: mennyire ter el a visszaepitett kep az eredetitol.
        # Alternativa lenne az L1 (abszolut elteres) - az elesebb kepeket ad,
        # az MSE hajlamos elmosni. Kezdesnek az MSE a szokasos.
        loss = F.mse_loss(x_hat, x)

        # A self.log intezi a TensorBoard/CSV irast es az epoch-szintu
        # atlagolast is. prog_bar=True: kiirja a progress barra is.
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        """Egy tanito batch. A visszaadott loss-bol a Lightning hivja a backward()-ot."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """
        Egy validacios batch. Itt a Lightning automatikusan kikapcsolja a
        gradiensszamitast (torch.no_grad) es eval modba teszi a modellt -
        ez utobbi fontos, mert a BatchNorm maskepp viselkedik tanitas es
        kiertekeles kozben.
        """
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        """Egy teszt batch. Ugyanaz, mint a validacio, csak a vegso merestre."""
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        """
        Itt mondod meg, mivel frissuljenek a sulyok.

        A self.parameters() minden tanithato tensort visszaad a modellbol
        (az osszes conv kernel, linear suly, batchnorm skala). Az Adam egy
        adaptiv optimizer: parameterenkent kulon lepeskozt tart nyilvan,
        ezert jellemzoen kevesebb hangolast igenyel, mint a sima SGD.
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
