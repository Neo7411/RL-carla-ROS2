# 04 — `vae/models.py` — a Variational Autoencoder

## Mi az a VAE és miért kell?

**A probléma:** a dashcam képe 160×80×3 = **38 400 szám**. Ha ezt közvetlenül beadnád az RL
ágensnek, két baj lenne:
1. Óriási hálózat kellene, ami lassan tanul.
2. Az RL amúgy is mintaszegény — a képfeldolgozás megtanulása RL jutalomjelből
   rendkívül nehéz.

**A megoldás:** válaszd szét a feladatot.
- Egy **VAE** megtanulja *előre, felügyelet nélkül* (unsupervised), hogyan lehet a képet
  **64 számmal** leírni. Ehhez csak képek kellenek, jutalom nem.
- Az RL ágens csak azt tanulja meg, hogy ebből a 64 számból (+ pár mérőszám) hogyan
  vezessen.

Ez a "World Models" (Ha & Schmidhuber, 2018) megközelítés lényege.

```
   dashcam kép                latent vektor            RL ágens
   160×80×3                   64 szám                  akció
   ┌──────────┐   encoder    ┌────┐    SAC háló      ┌─────────┐
   │ ▓▓▒▒░░   │ ──────────►  │ z  │ ──────────────►  │ steer   │
   │  ══════  │  (fagyasztva)└────┘                  │ throttle│
   └──────────┘                 │                    └─────────┘
                                │ decoder (csak megjelenítéshez)
                                ▼
                          ┌──────────┐
                          │ ▓▓▒▒░░   │  ← "amit a VAE megértett"
                          └──────────┘
```

> **Fontos:** ebben a projektben a VAE **fix, előre betanított** (`vae/model/best.tar`), és
> a tanítás során **nem frissül**. A tanító script (ami a `best.tar`-t készítette) nincs
> ebben a repóban.

## Miért "variational" (variációs)?

Egy sima autoencoder egy pontot ad vissza a latent térben. A VAE **eloszlást** ad:
egy `μ` (átlag) és egy `σ` (szórás) vektort, és a latent vektort ebből **mintavételezi**.

Ennek az az előnye, hogy a latent tér **sima és folytonos** lesz: hasonló képek közel
kerülnek egymáshoz, és a köztes pontok is értelmes képeknek felelnek meg. Ez sokkal jobb
bemenet egy neurális hálónak, mint egy "lyukacsos" latent tér.

---

## A kód: `vae/models.py`

### Konstruktor — encoder (11–20. sor)

```python
image_size = (80, 160)

self.encoder = nn.Sequential(
    nn.Conv2d(3,   32,  4, stride=2), nn.ReLU(),
    nn.Conv2d(32,  64,  4, stride=2), nn.ReLU(),
    nn.Conv2d(64,  128, 4, stride=2), nn.ReLU(),
    nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(),
)
```

Négy konvolúciós réteg, mindegyik **felezi a felbontást** (`stride=2`) és **duplázza a
csatornaszámot**. Ez a klasszikus "piramis" minta: térbeli felbontást cserélünk
jellemző-gazdagságra.

**A tényleges méretek lépésenként** (kiszámolva a `(H - k)/s + 1` képlettel):

| Réteg | Kimenet (C × H × W) | Elemszám |
|-------|--------------------|----------|
| bemenet | 3 × 80 × 160 | 38 400 |
| conv1 (3→32, k=4, s=2) | 32 × 39 × 79 | 98 592 |
| conv2 (32→64) | 64 × 18 × 38 | 43 776 |
| conv3 (64→128) | 128 × 8 × 18 | 18 432 |
| conv4 (128→256) | **256 × 3 × 8** | 6 144 |
| flatten | 6 144 | |
| **latent** | **64** | ← 600-szoros tömörítés! |

### `_calculate_spatial_size` (62–76. sor)

```python
def _calculate_spatial_size(self, image_size, conv_layers):
    H, W = image_size
    size_hist = [(H, W)]
    for layer in conv_layers:
        if layer.__class__.__name__ != 'Conv2d':
            continue
        conv = layer
        H = int((H + 2*conv.padding[0] - conv.dilation[0]*(conv.kernel_size[0]-1) - 1) / conv.stride[0] + 1)
        W = int((W + 2*conv.padding[1] - conv.dilation[1]*(conv.kernel_size[1]-1) - 1) / conv.stride[1] + 1)
        size_hist.append((H, W))
    return (H, W), size_hist
```

Ez a PyTorch hivatalos kimeneti méret képlete. **Szép megoldás:** a kód automatikusan
kiszámolja a lapos (flatten) réteg méretét, ahelyett hogy be lenne drótozva egy szám.
Ha megváltoztatod a képméretet vagy a rétegeket, a háló magától alkalmazkodik.

### Latent rétegek (22–28. sor)

```python
(self.encoded_H, self.encoded_W), size_hist = self._calculate_spatial_size(image_size, self.encoder)
# → encoded_H = 3, encoded_W = 8

self.mean   = nn.Linear(3 * 8 * 256, 64)     # 6144 → 64
self.logstd = nn.Linear(3 * 8 * 256, 64)     # 6144 → 64

self.latent = nn.Linear(64, 3 * 8 * 256)     # 64 → 6144 (visszafelé)
```

Két **külön** lineáris réteg ugyanabból a jellemzővektorból: az egyik az átlagot, a másik a
szórás logaritmusát adja. Így tanulja meg a háló, mennyire "bizonytalan" egy adott kép
kódolásában.

### Decoder (29–38. sor)

```python
self.decoder = nn.Sequential(
    nn.ConvTranspose2d(256, 128, 4, stride=2), nn.ReLU(),
    nn.ConvTranspose2d(128, 64,  4, stride=2), nn.ReLU(),
    nn.ConvTranspose2d(64,  32,  5, stride=2), nn.ReLU(),   # ← figyelj: k=5!
    nn.ConvTranspose2d(32,  3,   4, stride=2), nn.Sigmoid(),
)
```

Az encoder tükörképe. **Miért `5` a harmadik kernel?** Mert az encoder `int()`-tel lefelé
kerekít, és a transzponált konvolúció `(H-1)*s + k` képlete nem adná vissza pontosan az
eredeti méretet. Az 5-ös kernel korrigálja ezt:

| Réteg | Kimenet |
|-------|---------|
| bemenet (latent-ből átformálva) | 256 × 3 × 8 |
| deconv1 (k=4, s=2) | 128 × 8 × 18 |
| deconv2 (k=4, s=2) | 64 × 18 × 38 |
| deconv3 (**k=5**, s=2) | 32 × 39 × 79 |
| deconv4 (k=4, s=2) | 3 × **80 × 160** ✓ |

**A `Sigmoid()` a végén:** a kimenetet 0–1 közé szorítja, ami megfelel a `ToTensor()`-rel
normalizált bemeneti képeknek.

---

## A metódusok

### `encode(x)` (40–43. sor)
```python
def encode(self, x):
    x = self.encoder(x)              # (1, 3, 80, 160) → (1, 256, 3, 8)
    x = x.flatten(start_dim=1)       # → (1, 6144)   [a batch dimenzió megmarad]
    return self.mean(x), self.logstd(x)   # → két (1, 64)-es tenzor
```

### `decode(z)` (45–49. sor)
```python
def decode(self, z):
    z = self.latent(z)                                    # (1, 64) → (1, 6144)
    z = z.view(-1, 256, self.encoded_H, self.encoded_W)   # → (1, 256, 3, 8)
    z = self.decoder(z)                                   # → (1, 3, 80, 160)
    return z
```

### `reparameterize(mu, logvar)` (51–54. sor) ★
```python
def reparameterize(self, mu, logvar):
    sigma = logvar.exp()
    eps = torch.randn_like(sigma)
    return eps.mul(sigma).add_(mu)      # z = mu + sigma * eps
```

**Ez a "reparameterization trick".** A probléma: nem lehet visszaterjeszteni a gradienst egy
mintavételezésen keresztül. A megoldás: a véletlenszerűséget kihúzzuk egy külön `eps`
változóba, ami nem függ a paraméterektől. Így `z = μ + σ·ε` már differenciálható `μ` és
`σ` szerint.

> ⚠ **Elnevezési következetlenség:** a paraméter neve `logvar` (log-variancia), de a
> réteg neve `self.logstd` (log-szórás), és a kód `logvar.exp()`-et számol.
> - Ha ez tényleg **log-szórás**, akkor `sigma = exp(logstd)` — helyes.
> - Ha **log-variancia** lenne, `sigma = exp(0.5 * logvar)` kellene.
>
> A rétegnév alapján az első az igaz, tehát a kód **konzisztens önmagával** (ugyanígy volt
> a betanításkor is), csak a paraméternév félrevezető. Ne "javítsd meg", mert akkor nem
> illeszkedne a betanított súlyokhoz!

### `forward(x, ...)` (56–60. sor)
```python
def forward(self, x, encode=False, mean=False):
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    x = self.decode(z)
    return x, mu, logvar
```

A teljes átfutás. A `(rekonstrukció, mu, logvar)` hármas azért kell, mert a VAE
veszteségfüggvénye **két tagból** áll:
```
Loss = rekonstrukciós hiba (MSE vagy BCE)  +  KL-divergencia(N(mu, sigma) ‖ N(0, 1))
```
A KL tag "húzza" a latent eloszlást a standard normálishoz — ez teszi a latent teret simává.

> Az `encode` és `mean` paraméterek **nincsenek használva** a törzsben — feltehetően egy
> régebbi verzió maradványai, ahol lehetett kérni csak a kódolást vagy a determinisztikus
> átlagot.

---

## A `vae/model/` mappa tartalma

| Fájl | Mi ez |
|------|-------|
| `best.tar` | ★ a legjobb validációs hibájú checkpoint — **ezt tölti be a projekt** |
| `checkpoint.tar` | az utolsó checkpoint |
| `progress.csv` | a VAE tanítás vesztesége epochonként |
| `loss.png` | veszteséggörbe ábra |
| `compare_gt_and_pred_plot.png` | eredeti vs. rekonstruált képek összehasonlítása |
| `plot_vae.png` | latent tér vizualizáció |

**Nézd meg a `compare_gt_and_pred_plot.png`-t** — ez mutatja meg legjobban, mit "lát" az
ágens valójában. Ha a rekonstrukción nem látszanak élesen a sávhatárok, akkor az ágens
sem tudja őket használni.

## A `VAE_MODEL = "vae_64_augmentation"` config érték

A `config.py`-ban szerepel, de **a kód sehol nem használja** — a `train.py` fixen a
`'./vae/model'` mappát tölti be. Ez a név csak dokumentációs címke a mentett `config.json`-ban
(valószínűleg azt jelöli: 64 dimenziós latent, adataugmentációval tanítva).

---

## Gyakorlati tudnivalók / gyanús pontok

1. **A VAE CPU-n fut.** A `load_vae` nem hívja a `.to('cuda')`-t, tehát a képkódolás
   minden lépésben a processzoron történik. Ez lassítja a tanítást. GPU-ra téve:
   ```python
   model = model.to('cuda')
   # majd a preprocess_frame után: frame = frame.to('cuda')
   ```

2. **Nincs `.eval()` hívás.** Ennél a hálónál nincs Dropout vagy BatchNorm, tehát
   gyakorlatilag nincs különbség — de jó gyakorlat lenne meghívni.

3. **A latent véletlenszerű.** A `reparameterize` mintavételez, tehát ugyanaz a kép
   kétszer más állapotot ad. Lásd a megjegyzést a
   [03-wrappers-rewards-state.md](03-wrappers-rewards-state.md#encode_stateenv-67102-sor--minden-lépésben-fut)-ben.

4. **`torch.load` biztonsági figyelmeztetés.** Újabb PyTorch verziókban a
   `torch.load(...)` alapértelmezés szerint `weights_only=True`-t használ, ami törhet
   ezzel a checkpoint formátummal. Ha hibát kapsz, próbáld:
   ```python
   state = torch.load(model_dir, weights_only=False)
   ```

---

➡️ Következő: [05-navigation.md](05-navigation.md)
