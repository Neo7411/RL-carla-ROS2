# LiDAR autoencoder — működési dokumentáció

Ez a mappa egy LiDAR feature extractort tanít: a CARLA lidarjának **range
image-ét** tömöríti egy latens reprezentációvá, amit később az RL agent
observation-jébe teszünk.

A tanítás **élő CARLA adaton** fut: az autó autopilotban körbemegy a pályán, a
lidar képei egy körkörös bufferbe gyűlnek, és a háló menet közben tanul
belőlük. Nincs külön adatgyűjtő fázis.

Az encoder **gráf-alapú** (EdgeConv), a TopoLiDM paper nyomán — ez a repo
lényegi újdonsága.

---

## 1. Gyors indítás

```bash
# 1. CARLA szerver (külön terminálban)
./CarlaUE4.sh

# 2. Tanítás
conda activate carla_rl
cd feature_extractions/lidar
python train.py
```

**Billentyűk:**

| gomb | hatás |
|---|---|
| `P` | autopilot be/ki |
| `T` | tanítás szünet/indítás |
| `C` / `Shift+C` | időjárás előre/hátra |
| `BACKSPACE` | új autó, új pozíció |
| `S` | checkpoint mentése most |
| `ESC` | kilépés (mentéssel) |

A checkpoint a `lidar_ae.ckpt` fájlba kerül, 1000 lépésenként és kilépéskor
automatikusan. Újraindításkor onnan folytatja.

### Amit a képernyőn látsz

Minden a **jobb felső sarokban**, egy oszlopban:

```
┌──────────────────────────────────────────────────────────┐
│ ┌──────────┐                        ┌─────────────────┐  │
│ │ HUD infó │                        │ Felülnézet      │  │
│ │ Tanitas  │                        │ (csak nézni)    │  │
│ │ Lepes    │   spectator kamera     │                 │  │
│ │ Loss     │   (háttér)             │   [autó középen]│  │
│ │ Buffer   │                        │                 │  │
│ └──────────┘                        └─────────────────┘  │
│                                     ┌─────────────────┐  │
│                                     │ Range (bemenet) │  │
│                                     └─────────────────┘  │
│                                     ┌─────────────────┐  │
│                                     │ AE rekonstrukció│  │
│                                     │    (decoded)    │  │
│                                     └─────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

1. **Felülnézet (BEV)** — ezen **nem** tanul a háló, csak neked mutatja meg, mi
   van az autó körül (a range image ránézésre nehezen olvasható).
2. **Range image (bemenet)** — ezen tanul.
3. **AE rekonstrukció (decoded)** — amit a háló visszaad belőle.

A 2. és 3. panel összehasonlítása mutatja a tanítás haladását: ahogy tanul, a
kettőnek egyre jobban kell hasonlítania.

A range image valódi aránya 16:1 (64×1024), ami 360 px széles panelen 22 px
magas csík lenne — azon semmit nem lehet kivenni. Ezért **függőlegesen nyújtva**
rajzoljuk (90 px). A torzítás csak megjelenítés; a háló a valódi arányú adatot
kapja.

A range image színezése: **piros = közeli**, **kék = távoli**, fekete = üres
cella. Szürkeárnyalat helyett azért színskála, mert így szabad szemmel is
megítélhető, hogy a rekonstrukció eltalálja-e a *távolságot*, nem csak azt, hogy
van-e ott egyáltalán valami.

---

## 2. Miért range image és nem BEV — a legfontosabb döntés

Ez a projekt korábban BEV (felülnézeti) képen tanult. **Mérésekkel kiderült,
hogy az rossz választás**, és ezt érdemes érteni, mert a kísértés nagy
visszatérni rá (a BEV sokkal olvashatóbb ránézésre).

Két okból rossz: a BEV **ritka** (a gráf-encoder sűrű bemenetet igényel), és
**információt dob el** (a raszterizálás elveszti a pontos távolságot).

### A range image nem „kép" a szokásos értelemben

- **64 sor** = elevációs szögek, +10°-tól −25°-ig (a lidar 64 csatornája)
- **1024 oszlop** = **teljes 360° azimut**, körbe az autó körül (0,35°/oszlop)
- **a cella értéke = a TÁVOLSÁG**, nem egy raszterezett folt

Vagyis minden cella **egy valódi lézersugár mérése** — ez a pontfelhő tömör,
szinte veszteségmentes reprezentációja, nem információvesztő vetítés.

Hol van melyik irány (mérve):

| irány | oszlop |
|---|---|
| hátra | 0 |
| balra | 256 |
| **előre** | **512** (a kép közepe) |
| jobbra | 768 |

A kép bal és jobb széle fizikailag **ugyanaz a pont** (hátul) — pontosan ezért
helyes a `CircularConv2d` körkörös paddingja.

### A döntő szám: sűrűség

A gráf-encoder k-NN-je sűrű, tartalmas bemenetet igényel. Mért értékek:

| bemenet | kitöltöttség | tartalmas pont a Stem után |
|---|---|---|
| BEV kép (256×256) | 3,3% | 117 / 1024 = **11%** |
| **range image (64×1024)** | **97,1%** | 996 / 2048 = **97%** |

A BEV azért ritka, mert felülnézetből az útfelület nagy része üres — a lidar
nem fentről néz. A range image viszont a szenzor saját nézőpontját őrzi meg,
ahol **minden kilőtt sugárnak van mérése** (talaj, fal, autó).

### ⚠️ Nyitott kérdés: a pozíció-kódolás dominanciája

Mérés **tanítatlan** hálón, a Stem kimenetén:

| | amplitúdó (std) |
|---|---|
| tényleges feature | 0,035 |
| `PositionalEncoding2D` | **0,590** |

A pozíció-kódolás ~17×-esen elnyomja a tartalmat, ezért a k-NN gyakorlatilag
csak a pozíciót „látja": a választott szomszédok átlagos térbeli távolsága
**1,8 cella** (pozíció-kódolás nélkül 36,1 lenne). Ez azt jelenti, hogy az
EdgeConv ilyenkor egy drága 3×3-as konvolúcióhoz hasonlóan viselkedik.

**Ez az eredeti TopoLiDM kód tulajdonsága, nem a range image-é** — BEV-en
ugyanez mérhető (0,583 vs 0,041).

Amit ez a mérés **nem** dönt el: tanítás során a Stem feature-einek amplitúdója
nő, ami eltolhatja ezt az arányt. Hosszabb tanítás után érdemes újramérni. Ha a
dominancia megmarad, a `PositionalEncoding2D` leskálázása (vagy tanulható
súllyal való ellátása) érdemi javulást hozhat — ezt viszont **még nem mértem
végig**, ezért nincs beépítve.

### Loss: a ritkaság csapdája (BEV-en volt, range image-en nincs)

BEV-en a sima L1 loss mellett a hálónak **megérte mindent üresre állítani**:

| „megoldás" | teljes L1 |
|---|---|
| „mindig −1-et adok" (semmit nem tanul) | 0,0428 |
| a háló az átlagot adja | 0,0827 |

Ezért kellett ott súlyozott loss. **Range image-en ez nem probléma** (97%
kitöltöttség), ezért itt sima L1 van — ahogy az eredeti TopoAutoencoder is
használja.

**L1 és nem MSE**: az L1 élesebb kimenetet ad, az MSE elmosná az objektumok
körvonalát — range image-nél pont az élek (egy autó széle, egy fal vége)
hordozzák az információt.

---

## 3. Az adatút

```
CARLA ray_cast lidar (360°, 50 m, 64 csatorna, 40 Hz, 2 621 440 pont/mp)
        │
        │  (x, y, z, intensity) négyesek, ~65 536 pont/fordulat
        ▼
Lidar.process_lidar_input()               carla_env/wrappers.py
        │
        ├─── on_recv_points ──► (N,3) nyers XYZ ──┐
        │                                          │
        └─── on_recv_image ───► BEV kép ──► HUD    │  (csak nézni)
                                                   │
        ┌──────────────────────────────────────────┘
        ▼
points_to_range_image()                   train.py
        │  pcd2range()    gömbi vetítés, a cellákba a távolság (-1 ahol üres)
        │  process_scan() log2 skálázás + normalizálás [-1, 1]-be
        ▼
range image (1, 64, 1024) float32, [-1, 1]
        │
        ├──────────────► replay buffer (512 kép)
        │                        │
        │                        ▼
        │                  Trainer._loop()  ← külön szálon, GPU-n
        ▼                        │
    Encoder (gráf)               ▼
        │                  gradiens lépés (L1 loss)
        ▼
  latens (16, 16, 128)
        │
        ▼
    Decoder (ResNet)
        │
        ▼
rekonstrukció (1, 64, 1024)
```

### 3.1 A szenzor beállításai

`carla_env/wrappers.py`, `Lidar.__init__`:

| paraméter | érték | miért |
|---|---|---|
| `horizontal_fov` | **360°** | a `CircularConv2d` a kép két szélét összeragasztja — ez csak körbeérő azimutnál helyes |
| `channels` | 64 | = a range image 64 sora |
| `points_per_second` | **2 621 440** | 64 × 1024 × 40 Hz — egy teljes fordulathoz 65 536 pont kell, kevesebbnél a kép kilyukadna |
| `range` | 50 m | |
| `upper_fov` / `lower_fov` | +10° / −25° | = a range image `FOV` paramétere |
| `rotation_frequency` | **40** | sűrűbb mintavétel |

**Ezeknek egyezniük kell a `train.py` `RANGE_SIZE` / `FOV` / `DEPTH_RANGE`
értékeivel**, különben a vetítés rossz cellákba szórja a pontokat.

### ⚠️ A `points_per_second` és a `rotation_frequency` együtt jár

A `points_per_second` a **teljes** pontkibocsátás, ami elosztódik a fordulatok
között. Egy teljes range image-hez fordulatonként 65 536 pont kell:

```
points_per_second = 64 * 1024 * rotation_frequency
```

Ha csak a frekvenciát emeled a pontszám nélkül, a kép arányosan kilyukad —
mérve:

| beállítás | kitöltöttség |
|---|---|
| 40 Hz + **2 621 440** pont/mp | **96,2%** |
| 40 Hz + 1 310 720 pont/mp (a régi érték) | 48,3% — fele üres |

### 3.2 Két koordinátarendszer-buktató

```python
xyz = points[:, :3].copy()
xyz[:, 1] = -xyz[:, 1]
```

- **`.copy()`**: a CARLA szerver újrahasznosítja a buffert, tehát másolás nélkül
  a tartalom bármikor felülíródhat alattunk — épp feldolgozás közben is.
- **`y` előjelváltás**: a CARLA y tengelye balra pozitív (bal kezes rendszer), a
  `pcd2range` gömbi vetítése viszont jobb kezes rendszert vár. Enélkül a range
  image vízszintesen tükrözve lenne.

### 3.3 A log skálázás

`process_scan()` a nyers távolságot így normalizálja:

```
log2(d + 1) / DEPTH_SCALE * 2 - 1        DEPTH_SCALE = 6.0
```

Miért log: **a közeli tartomány a fontos**. 5 és 10 méter között sokkal nagyobb
a különbség vezetés szempontjából, mint 45 és 50 között. A log2 pont ezt a
felbontást adja a közelbe:

| távolság | normalizált érték |
|---|---|
| 5 m | −0,138 |
| 10 m | 0,153 |
| 15 m | 0,333 |
| 30 m | 0,651 |
| 50 m | 0,891 |

`DEPTH_SCALE = 6.0`, mert `log2(50+1) = 5,67` — ez a biztonságos felső határ.

### 3.4 Miért `[-1, 1]`

A decoder utolsó rétege `tanh` (`tanh_out=True`), ami pont ebbe a tartományba
képez. `[0, 1]` bemenettel a háló sosem tudná kihasználni a kimeneti tartomány
alsó felét.

---

## 4. A háló

### 4.1 Encoder — gráf-alapú (a lényeg)

`AE/encoder.py`, `AE/layers.py`

```
(B, 1, 64, 1024)
  → Stem              → (B, 64, 16, 128)     3 conv: /4 függőleges, /8 vízszintes
  → PositionalEncoding
  → flatten           → (B, 64, 2048)        "pontfelhő" alak
  → GraphLayer × 4    → h1..h4               EdgeConv, k=20 szomszéd
  → Conv1d proj       → (B, 16, 2048)
  → reshape           → (B, 16, 16, 128)     ← a LATENS
```

**Stem**: 64×1024 = 65 536 „pontra" a k-NN kezelhetetlen lenne (N×N
távolságmátrix rétegenként). 16×128 = 2048 pont már megy. Az aszimmetria
(függőlegesen /4, vízszintesen /8) azért helyes, mert a bemenet maga is
aszimmetrikus: 64 elevációs sor áll szemben 1024 azimut oszloppal.

**PositionalEncoding2D**: a `flatten` után a gráf-rétegek elvesztenék a térbeli
információt — nem tudnák, melyik „pont" hol volt a képen. Ez injektálja vissza.

**GraphLayer** (EdgeConv, DGCNN nyomán): minden pontra megkeresi a
*feature-térben* legközelebbi k=20 szomszédot, majd a
`[szomszéd − közép ‖ közép]` él-feature-ökön 1×1 konvolúciót futtat, végül
max-poolt a szomszédok fölött. A max-pool teszi permutáció-invariánssá.

A gráf **dinamikus**: minden réteg újraszámolja a k-NN-t a saját bemenetén.

### 4.2 Decoder — sima ResNet

`AE/decoder.py`. Nem gráf-alapú, ResNet blokkok + `Upsample`:

```
(B, 16, 16, 128) → (B, 1, 64, 1024)
   (16,128) →(1,2)→ (16,256) →(2,2)→ (32,512) →(2,2)→ (64,1024)
```

A decoder **csak a tanításhoz kell** — ő adja a rekonstrukciós loss másik felét.
Ha kész a tanítás és az encodert feature extractorként használod, eldobható.

### 4.3 A `strides` indexelés csapdája

A `Decoder.__init__` így indexel (`decoder.py:143`):

```python
stride = tuple(strides[i_level - 1]) if i_level > 0 else None
```

Vagyis az `i_level=0` szinten **nincs** upsample, és a listát **eggyel eltolva**
olvassa. Ezért `len(ch_mult) - 1` darab upsample fut, és a `strides` **utolsó
eleme soha nem kerül felhasználásra** (dummy).

A Stem /4 és /8-at csinál, ezt három upsample-lel kell visszaadni, tehát **négy
szint** kell:

```python
ch_mult=(1, 2, 4, 4)
strides=((2, 2), (2, 2), (1, 2), (1, 1))
#                                 ^^^^^^ dummy, sosem használt
```

A felhasználási sorrend fordított: `i_level=3` veszi a `strides[2]`-t, aztán
`i_level=2` a `strides[1]`-et, stb.

Három szinttel (`ch_mult=(1,2,4)`) csak két upsample futna, a kimenet 64×512
lenne, és az L1 loss shape hibával elszállna.

### 4.4 Mérések

| | érték |
|---|---|
| paraméterszám | 7,8M |
| gradiens lépés (batch=4) | ~278 ms |
| rekonstrukció (batch=1) | ~25 ms |
| VRAM (batch=4) | 4,0 GB |
| VRAM (batch=8) | 7,0 GB — **nem fér be** a CARLA mellé 8 GB-on |

Tipikus tanulási görbe (szintetikus adaton, 60 lépés): loss `0,106 → 0,046`,
a rekonstrukció tartománya `-1,00 .. 0,92`.

---

## 5. Miért külön szálon tanít

A rajzoló ciklus 30 FPS-t céloz, az **33 ms/képkocka**. Egy gradiens lépés
**278 ms**. Ha a tanítás a fő ciklusban futna, a kép 2-3 FPS-re esne, és a
billentyűk is akadoznának (a pygame események sem jutnának szóhoz).

Ezért a `Trainer` saját daemon szálon pörög. **A GIL nem gond**: a PyTorch a
CUDA hívások és a nagy tenzorműveletek idejére elengedi, tehát a két szál
ténylegesen párhuzamosan halad.

### Szálbiztonság

| érték | védelem | miért |
|---|---|---|
| `buffer` | `self.lock` | a `deque.append` önmagában szálbiztos, de a `random.sample` nem: közben változhat a méret |
| súlyok mentéskor | `self.save_lock` | különben a `state_dict()` másolás közben futna egy `optimizer.step()`, és fél-frissített állapot kerülne a checkpointba |
| `last_input`, `last_recon` | nincs | egy referencia-értékadás atomi, és nem baj, ha a fő szál egy képkockával régebbit lát |

Kilépéskor **először a szálat állítjuk le, csak utána mentünk** — fordítva
fél-frissített súlyok kerülnének a fájlba.

A lidar callback is **másik szálon** fut (a CARLA sajátján), ezért ott csak
lerakjuk az adatot egy `pending` dictbe, és a fő ciklus veszi át — így a vetítés
és a tanítás nem fogja a szenzor szálát.

---

## 5.5 Forgalom és hybrid physics

```python
NUM_TRAFFIC = None              # = ahány spawn pont van (Town04: ~370)
HYBRID_PHYSICS_RADIUS = 70.0    # méter
```

Alapértelmezésben a CARLA **minden** autóra teljes kerékfizikát számol
(felfüggesztés, gumi-tapadás, motor). Pár tucat autó fölött ez megfogja a
szimulációt: a szerver FPS leesik, az autopilot késve reagál — innen a rángatás
és a koccanások.

**Hybrid physics mode**: csak a `hero` körül `HYBRID_PHYSICS_RADIUS` méteren
belül van teljes fizika. A távolabbi autókat a Traffic Manager olcsó
„teleportálós" módban mozgatja — tolja őket a sáv mentén, kerékszámítás nélkül.
Ránézésre ugyanúgy közlekednek, csak töredék költséggel.

Ezért fér el a pálya **összes** spawn pontján autó.

A 70 m-es sugár bőven több a lidar 50 m-es hatótávjánál, tehát **minden autó,
amit a szenzor lát, valódi fizikával mozog** — a range image nem torzul attól,
hogy a háttérben egyszerűsített a mozgás.

### Két sorrendi részlet

- **Az ego megy le először, csak utána a forgalom.** A hybrid physics a `hero`
  köré rajzolja a kört (a `role_name="hero"` attribútum jelöli ki), tehát a
  hero-nak már léteznie kell. Ráadásul ha a forgalom foglalná el mind a ~370
  pontot, az egónak nem maradna hely.
- **A spawn egy batchben megy** (`apply_batch_sync`): 370 külön RPC hívás
  percekig tartana.

Tele pályán a `BACKSPACE` (új autó) nem talál azonnal szabad pontot — ezért a
kód vár és újrapróbál ötször, mielőtt hibát dobna.

---

## 6. Replay buffer

```python
BUFFER_SIZE = 512       # deque(maxlen=512), a legrégibb automatikusan kiesik
LEARNING_STARTS = 64    # ennyi kép alatt még csak gyűjtünk
```

Miért kell egyáltalán buffer: az élő tanítás miatt minden mintát kevesebbszer
lát a háló. A bufferrel egy kép több gradiens lépésben is szerepel, mielőtt
kiesik. Egyben csúszóablakot is ad — mindig a legutóbbi ~512 képből tanulunk.

**Figyelj a `Lepes/kep` értékre a HUD-on.** A tanítás jóval lassabb, mint az
adatgyűjtés: 278 ms/lépés vs. 25 ms/kép (40 Hz), tehát a mért arány **~0,07** —
sok kép úgy esik ki a bufferből, hogy egyszer sem tanult belőle.

Ez nem hiba: 40 Hz-en két egymást követő lidar kép szinte azonos, tehát a
kimaradó képek alig hordoznak új információt. Ha mégis zavar, a `BUFFER_SIZE`
növelése segít (egy kép tovább marad bent, több esélye van bekerülni egy
batchbe).

---

## 7. Mit nézz a HUD-on

```
Tanitas:              FUT
Lepes:                847
Loss (L1):        0.04590     ← a teljes rekonstrukciós hiba
Loss (kozeli):    0.03510     ← EZ a beszédesebb szám
Buffer:           512/512
Kepek:               6203
Lepes/kep:           0.14
```

A **`Loss (kozeli)`** a 15 méteren belüli cellák hibája (`NEAR_THRESHOLD = 0.333`
a normalizált skálán). Ez azért fontosabb, mert a távoli háttér (falak,
épületek) könnyen tanulható és dominálja az átlagot, miközben vezetés
szempontjából a közeli objektumok számítanak.

---

## 8. Fájlok

| fájl | szerep |
|---|---|
| `train.py` | a teljes tanítás: CARLA, autopilot, HUD, replay buffer, tanító szál |
| `AE/model.py` | `LidarAE` — encoder + decoder összefogva, Lightning modul |
| `AE/encoder.py` | gráf-alapú encoder |
| `AE/decoder.py` | ResNet decoder |
| `AE/layers.py` | `CircularConv2d`, `GraphLayer`, `PositionalEncoding2D`, `Stem` |
| `AE/geometry.py` | `pcd2range` / `range2pcd` / `process_scan` — pontfelhő ↔ range image |
| `AE/distributions.py` | `DiagonalGaussianDistribution` — csak VAE módban (`kl_weight > 0`) |
| `papers/` | a forrás-paper (TopoLiDM) |

**Az `AE/` mappa érintetlen** — pontosan az eredeti TopoLiDM kód (a topológiai
loss részek nélkül, lásd `AE/model.py` fejlécét).

### Ami a projekt eredeti állapotához képest változott

`carla_env/wrappers.py`, `Lidar` osztály:

| | eredeti | most | miért |
|---|---|---|---|
| `horizontal_fov` | 110° | **360°** | a `CircularConv2d` körkörös paddingja csak így helyes |
| `range` | 20 m | **50 m** | |
| `points_per_second` | 50 000 | **2 621 440** | 64×1024-es range image-hez 65 536 pont/fordulat kell, 40 Hz-en |
| `lower_fov` | −20° | −25° | |
| `rotation_frequency` | 30 | **40** | surubb mintavetel; a points_per_second-ot EGYUTT kell emelni |
| kimenet | csak BEV kép | **BEV + nyers XYZ** | a háló a range image-en tanul, a BEV a HUD-é |
| `np.fabs()` a BEV-ben | volt | **kivéve** | **valódi hiba volt** — lásd lent |

**Az `np.fabs()`-ról**: az abszolútérték a kép egy negyedébe gyűrte az egész
jelenetet — az autótól balra és jobbra, illetve előre és hátra eső pontok
egymásra tükröződtek. 110°-os, előre néző lidarnál ez nem tűnt fel, 360°-nál
viszont az autó mögötti forgalom ráhajtogatódna az előtte lévőre.

---

## 9. Hibakeresés

### `size mismatch` a checkpoint betöltésekor

```
RuntimeError: Error(s) in loading state_dict for LidarAE:
    size mismatch for decoder.up.3.block.0.conv1.weight: ...
```

A `lidar_ae.ckpt` **más háló-alakkal** készült (más `DDCONFIG`, vagy egy korábbi
BEV-es változat). Töröld:

```bash
rm lidar_ae.ckpt
```

### `CUDA out of memory`

`BATCH_SIZE = 4` → 4,0 GB. Ha a CARLA szerver ugyanazon a GPU-n fut és nem fér
be, vedd le 2-re.

### A rekonstrukció fekete/üres marad

Nézd a `Loss (kozeli)` értéket. Ha nem csökken, a háló nem tanul — ellenőrizd,
hogy a `T` gombbal nincs-e szüneteltetve a tanítás, és hogy a buffer megtelt-e
(`LEARNING_STARTS = 64` alatt még csak gyűjt).

### A range image nagy része üres (sok fekete)

Ellenőrizd, hogy a `points_per_second` tényleg 1 310 720 — kevesebbnél nem jut
elég sugár egy fordulatra, és a kép kilyukad.

---

## 10. Az RL-be kötés (következő lépés)

Az agent csak az **encodert** használja:

```python
from AE.model import LidarAE

model = LidarAE(ddconfig=DDCONFIG, kl_weight=KL_WEIGHT)
model.init_from_ckpt("lidar_ae.ckpt")
model.eval()

with torch.no_grad():
    z = model.encode(range_img, sample=False)   # (B, 16, 16, 128)
```

`sample=False`: determinisztikus, a mean kell, nem minta.

### ⚠️ A latens mérete — ezt meg kell oldani

A latens `(16, 16, 128)` = **32 768 érték**. Ez laposítva nagyságrendekkel
nagyobb, mint a kamera AE 64 elemű latense, és az SB3 `MultiInputPolicy` **nem
normalizál** — csak összefűzi a jeleket egy vektorra. Így a lidar latens
elnyomná a többi observationt (steer, throttle 0..1, angle ±3,14, waypointok
±50).

A `camera_ae.py` kommentjei pontosan ezt a hibát dokumentálják egy korábbi
körből: ott a latens −121..144 között mozgott, és az agent nem tanult (a jutalom
monoton csökkent).

Lehetséges megoldások:
- pooling + `Linear` az encoder után, egy pár száz elemű vektorra
- `z_channels` csökkentése
- a kamera AE mintájára `tanh * latent_scale` korlátozás
