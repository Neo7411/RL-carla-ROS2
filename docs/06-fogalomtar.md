# 06 — Fogalomtár

Minden szakkifejezés, ami a projektben előfordul, magyar magyarázattal.

---

## Megerősítéses tanulás (Reinforcement Learning)

**Ágens (agent)**
A tanuló döntéshozó. Nálunk a SAC neurális hálózat, ami megmondja, mennyit kormányozzon
és gyorsítson az autó.

**Környezet (environment)**
Amiben az ágens cselekszik. Nálunk a `CarlaRouteEnv` + a CARLA szimulátor.

**Állapot / megfigyelés (state / observation)**
Amit az ágens "lát" egy adott pillanatban. Nálunk egy szótár: `vae_latent` (64 szám),
`vehicle_measures` (4 szám), `maneuver` (1 egész), `waypoints` (15×2 szám).
*Szigorúan véve az "állapot" a világ teljes leírása, a "megfigyelés" pedig amit az
ágens ebből lát — a gyakorlatban a két szót felváltva használjuk.*

**Akció (action)**
Amit az ágens csinál. Nálunk 2 folytonos szám: `steer ∈ [-1, 1]` és `throttle ∈ [0, 1]`.

**Jutalom (reward)**
Egy szám lépésenként, ami minősíti a döntést. Nálunk 0–1 normál lépésnél, -10 hibás
epizódvégnél. Lásd `rewards.py`.

**Epizód (episode)**
Egy teljes menet a kezdőállapottól a végállapotig. Nálunk: spawnolástól addig, amíg
ütközik / lesodródik / megáll / túl gyors, vagy 3000 m-t megtesz.

**Lépés (step / timestep)**
Egy döntés-cselekvés-visszajelzés ciklus. Nálunk 0.05 szimulált másodperc (20 FPS).

**Politika (policy, π)**
A szabály, ami állapotból akciót ad: `π(állapot) → akció`. Ez maga a neurális háló.

**Értékfüggvény (value function, V vagy Q)**
Becslés arra, mennyi jutalom várható innentől. A `Q(s, a)` azt becsli, mennyit ér az
`a` akció az `s` állapotban. A SAC két Q-hálót tanít (critic).

**Diszkont faktor (γ, gamma)**
Mennyire számít a jövő a jelenhez képest. `γ = 0.98` nálunk. A "hatékony horizont"
nagyjából `1/(1-γ) = 50` lépés = 2.5 másodperc. Ha hosszabb távú tervezést akarsz
(pl. korai lassítás kanyar előtt), ezt kell növelni.

**Felfedezés vs. kiaknázás (exploration vs. exploitation)**
Próbáljon-e ki új dolgokat vagy használja a már ismert jót? A SAC entrópia-tagja
(`ent_coef='auto'`) automatikusan egyensúlyoz.

**Replay buffer (visszajátszási puffer)**
Múltbeli `(állapot, akció, jutalom, új állapot)` négyesek tárolója. Nálunk 300 000 elem.
Az off-policy algoritmusok (mint a SAC) ebből tanulnak újra és újra — ez teszi őket
mintahatékonnyá.

**On-policy vs. off-policy**
- *On-policy* (pl. PPO): csak a jelenlegi politika által gyűjtött adatból tanul, majd eldobja.
- *Off-policy* (pl. SAC, DQN): régi adatból is tanulhat → sokkal kevesebb környezeti lépés kell.
  Szimulátorban, ahol egy lépés lassú, ez döntő.

**SAC (Soft Actor-Critic)**
Off-policy algoritmus folytonos akciótérhez. A "soft" arra utal, hogy nemcsak a jutalmat
maximalizálja, hanem a politika **entrópiáját** is — vagyis a lehető legváltozatosabb
maradjon, amíg az nem rontja a teljesítményt. Ez robusztus tanulást és jó felfedezést ad.

**Actor / Critic**
- *Actor* — a politika hálózat, ami az akciót adja.
- *Critic* — az érték (Q) hálózat, ami megmondja, mennyire volt jó az akció.

**Target network (cél-hálózat)**
A critic egy lassan frissülő másolata, ami stabilizálja a tanulást. A `tau=0.02` mondja
meg, milyen gyorsan követi az eredetit (soft update): `target = 0.02*új + 0.98*régi`.

**Entrópia együttható (`ent_coef`)**
A felfedezés súlya. `'auto'` = a SAC magától hangolja.

**gSDE (generalized State-Dependent Exploration)**
`use_sde=True`. Ahelyett, hogy minden lépésnél új véletlen zajt húzna (ami "reszketős"
vezetést ad), a zaj az állapot függvénye, és epizódonként ritkábban változik. Folytonos
vezérlésnél sokkal simább felfedezést eredményez.

**Tanulási ráta (learning rate)**
Mekkora lépéseket tesz a gradiens szerint. Nálunk ütemezett: 1e-4-ről 5e-7-re csökken.

**Batch size / gradient step**
Egy tanulási lépéshez ennyi mintát húz ki a bufferből (256), és `train_freq=64` környezeti
lépésenként `gradient_steps=64` ilyen frissítést végez.

**Terminated vs. truncated**
- *terminated* — a feladat valóban véget ért (ütközés).
- *truncated* — külső limit miatt vágtuk el (időkorlát).
A megkülönböztetés a value-becslésnél számít. Ez a projekt mindig `False`-t ad
truncated-re — apró elméleti pontatlanság.

**Curriculum learning**
Fokozatosan nehezedő feladatok. Nálunk enyhén jelen van: a páros epizódokban garantáltan
kereszteződéses útvonalat kap, hogy biztosan tanuljon kanyarodni.

**Reward shaping / reward hacking**
A jutalom megtervezése. Veszélye: az ágens azt tanulja meg, amit *megjutalmazol*, nem amit
*akarsz*. Ezért van szorzat a `reward_fn5`-ben összeg helyett — hogy ne lehessen egy
tényezőt "kijátszani".

---

## Számítógépes látás / neurális hálók

**VAE (Variational Autoencoder)**
Olyan háló, ami képet tömörít kevés számmá (encode), majd vissza tudja alakítani (decode).
"Variációs", mert eloszlást tanul, nem pontot — ettől lesz sima a latent tér.

**Latent vektor / latent tér**
A tömörített reprezentáció. Nálunk 64 dimenziós. A `LSIZE = 64` config érték.

**Encoder / Decoder**
Kódoló (kép → 64 szám) és dekódoló (64 szám → kép).

**Reparameterization trick**
`z = μ + σ·ε`, ahol `ε ~ N(0,1)`. Így a mintavételezésen át is lehet gradienst terjeszteni.

**KL-divergencia**
Két eloszlás "távolsága". A VAE veszteségében ez húzza a latent eloszlást a standard
normálishoz.

**Konvolúció / ConvTranspose (transzponált konvolúció)**
A konvolúció kicsinyít és jellemzőket von ki; a transzponált konvolúció ("dekonvolúció")
nagyít, a decoderben.

**Stride (lépésköz)**
`stride=2` → a kimenet fele akkora, mint a bemenet.

**Latent dimenzió megválasztása**
Túl kicsi → elveszik a lényeg (nem látszanak a sávok). Túl nagy → nem tömörít eleget, és
zajt is átvisz. 64 tipikus érték autonóm vezetéshez.

**MultiInputPolicy**
Stable-Baselines3 politika, ami `Dict` megfigyelési teret kezel: minden kulcshoz külön
feldolgozó ág, majd összefűzés.

**Domain randomization**
Változatosabbá tenni a szimulációt (időjárás, napszak, textúrák), hogy a modell
általánosítson. Ebben a projektben a `set_weather` sor **ki van kommentezve** —
ez egy kihasználatlan lehetőség.

---

## CARLA

**CARLA**
Nyílt forráskódú autonóm vezetés szimulátor, Unreal Engine 4 alapon. Szerver-kliens
architektúra: a szerver a szimuláció, a kliens (a mi Python kódunk) TCP-n csatlakozik
(2000-es port).

**Szinkron mód (synchronous mode)**
A szerver **vár** a kliensre; a `world.tick()` lépteti. Ez teszi determinisztikussá és
reprodukálhatóvá a tanítást. `settings.synchronous_mode = True`.

**`fixed_delta_seconds`**
A szimulációs lépés hossza. Nálunk `1/20 = 0.05` s.

**`world.tick()`**
Egy szimulációs lépés végrehajtása. **A CARLA egyetlen legfontosabb hívása.**

**Actor (szereplő)**
Bármi, ami a világban létezik: jármű, gyalogos, szenzor, forgalmi lámpa.

**Blueprint**
Egy actor "sablonja". `world.get_blueprint_library().find("vehicle.tesla.model3")`.

**Spawn point (megjelenési pont)**
A térképen előre definiált helyek, ahol biztonságosan lehet járművet elhelyezni.
A Town02-nek kb. 100 ilyen van; a projekt ezek indexeivel dolgozik (`intersection_routes`).

**Waypoint (útpont)**
Egy pont a **sáv közepén**, orientációval együtt. Nem csak koordináta: tudja, melyik
úthoz (`road_id`), szakaszhoz (`section_id`), sávhoz (`lane_id`) tartozik, és mi jön
utána (`waypoint.next(d)`).

**`map.get_waypoint(location)`**
Egy tetszőleges pontot "ráilleszt" a legközelebbi sávra.

**Topológia (`map.get_topology()`)**
A térkép útszakaszainak listája (kezdő- és végwaypoint párok). Ebből épül a
navigációs gráf.

**Transform**
Pozíció + forgatás (`Location` + `Rotation`).

**Dashcam / spectator kamera**
Nálunk a dashcam (160×80) adja az ágens megfigyelését; a spectator (1920×1080) csak
a képernyőre.

**Szemantikus szegmentáció**
Kamera, ami minden pixelhez osztálycímkét ad (út, járda, autó, gyalogos...). A projekt
támogatja (`custom_palette`), de jelenleg nem használja.

**Ego vehicle**
"A mi autónk" — amit vezérlünk.

**`VehicleControl`**
A vezérlőparancs: `throttle`, `steer`, `brake`, `hand_brake`, `reverse`, `manual_gear_shift`.
A projekt csak a `throttle`-t és `steer`-t használja.

**`set_simulate_physics(False)`**
Fizika kikapcsolása — teleportálás előtt kell, hogy ne pörögjön el az autó.

**Town02**
Kis városi térkép a CARLA-ban: derékszögű utcák, T-elágazások, egysávos utak.
Ideális kezdéshez, mert nincs autópálya és sávváltás.

**Town04**
Nagy térkép autópályával és több sávval. A `maps/Town04.osm` fájl itt van, de nem használt.
⚠ Áttéréshez javítani kellene a `RoadOption` enumot (lásd
[05-navigation.md](05-navigation.md#_lane_change_link-169210-sor)).

---

## Útvonaltervezés

**A\* (A-csillag) algoritmus**
Legrövidebb út keresése gráfban. Kombinálja a tényleges eddigi költséget (g) egy becsült
hátralévő költséggel (h). Ha a heurisztika nem becsli túl a valóst ("elfogadható"), az
eredmény garantáltan optimális. Nálunk a heurisztika a légvonalbeli távolság.

**Gráf / csomópont / él**
A térkép gráf-reprezentációja: csomópont = útszakasz vége, él = útszakasz.
`networkx.DiGraph` implementáció.

**RoadOption**
A manőver típusa: `LANEFOLLOW` (sávkövetés), `LEFT`, `RIGHT`, `STRAIGHT`, `VOID`.
Ez kerül az ágens állapotába `maneuver` néven.

**Globális vs. lokális tervezés**
- *Globális* — honnan hova, melyik utcákon (A\*). Ezt használjuk.
- *Lokális* — hogyan kormányozzam most a kormányt (PID vagy RL). Nálunk **ezt tanulja
  meg az RL ágens** a `LocalPlanner` helyett.

**PID szabályozó**
Klasszikus vezérlési módszer: `output = K_P·hiba + K_I·∫hiba + K_D·d(hiba)/dt`.
A `controller.py`-ban van, de a projekt nem használja (az RL váltja ki).

**Resolution (felbontás)**
A waypointok közti távolság. Nálunk `1.0` m.

**Rálátási távolság (lookahead)**
Mennyit "lát előre" az ágens. Nálunk 15 waypoint × 1 m = **15 méter**.

---

## Egyéb

**Gymnasium / Gym**
Az RL környezetek szabványos interfésze. `reset()`, `step(action)`, `observation_space`,
`action_space`. A Gymnasium a Gym utódja (a `step` 5 értéket ad vissza 4 helyett).

**Stable-Baselines3 (SB3)**
Kész, megbízható RL algoritmus implementációk PyTorch-ban.

**Callback**
Függvény, amit a tanítási ciklus időnként meghív. Nálunk: `HParamCallback`,
`TensorboardCallback`, `CheckpointCallback`.

**TensorBoard**
Vizualizációs eszköz a tanítási görbékhez. Indítás:
```bash
tensorboard --logdir tensorboard/
```
Majd böngészőben: `http://localhost:6006`

**Checkpoint**
Köztes modellmentés. Nálunk 13 000 lépésenként (`TOTAL_STEPS // NUM_CHECKPOINTS`).

**Seed (véletlenszám-mag)**
Reprodukálhatósághoz. `SEED = 100`.

**Busy wait (aktív várakozás)**
`while x is None: pass` — folyamatosan pörgeti a CPU-t, amíg egy feltétel nem teljesül.
A `_get_observation()`-ban ilyen van. Működik, de pazarló.

**Weak reference (gyenge referencia)**
Olyan hivatkozás, ami nem akadályozza a szemétgyűjtést. A `wrappers.py` szenzor
callback-jeiben használt, hogy ne legyen körkörös hivatkozás.

**Dependency injection**
Amikor egy komponens kívülről kapja meg a függőségeit ahelyett, hogy magának hozná létre.
Nálunk a `CarlaRouteEnv` így kapja a `reward_fn`-t és `encode_state_fn`-t — ezért lehet
őket kicserélni a környezet módosítása nélkül.

**ROS2 (Robot Operating System 2)**
Robotikai middleware: node-ok publikálnak és feliratkoznak topicokra.
**A jelenlegi kódban nincs ROS2** — a repó neve a tervezett irányra utal.
Lásd [07-tovabbfejlesztes.md](07-tovabbfejlesztes.md).

---

➡️ Következő: [07-tovabbfejlesztes.md](07-tovabbfejlesztes.md)
