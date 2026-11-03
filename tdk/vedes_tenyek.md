# TDK védés — amit tudnod kell fejből

Minden szám és döntés, amit a bizottság számonkérhet. A hangsúly ott van, ahol a
**„miért pont annyi?"** kérdés jöhet — mert az a legkínosabb pont, ha a saját
paramétereidet nem tudod megindokolni.

---

## 1. Környezet

| | |
|---|---|
| Szimulátor | CARLA **0.9.16**, Unreal Engine 4 alapú, kliens–szerver |
| Kapcsolat | TCP, alapértelmezett port **2000** |
| Mód | **szinkron**, rögzített **20 Hz** (0,05 s lépésköz) |
| Térkép | **Town04** — „8-as" alakú, végtelenített autópálya + városi utcák |
| Útvonaltervezés | `GlobalRoutePlanner`, epizódonként **véletlen** kezdő- és célpont |

**Miért szinkron mód?** Mert így a szerver csak akkor lép, ha a kliens utasítja. Ezért a
szenzoradat és a döntés mindig ugyanahhoz a szimulációs lépéshez tartozik, és a tanítás
reprodukálható.

**Miért Town04?** Mert egyszerre van benne többsávos autópálya (fel- és lehajtóval,
lóhere csomóponttal) és városi utca. A körbeérő autópálya hosszú, megszakítás nélküli
szakaszokat ad — előzést tanítani ezen lehet.

**Miért véletlen útvonal epizódonként?** Hogy ne tanuljon rá egy konkrét útvonalra.

---

## 2. Az algoritmus: SAC

| | |
|---|---|
| Algoritmus | **Soft Actor-Critic**, off-policy, aktor–kritikus |
| Könyvtár | **Stable-Baselines3** (PyTorch), Gymnasium interfész |
| Diszkontfaktor | **γ = 0,98** |
| Tanítás hossza | **1 millió lépés** modellenként, **~12 óra** |
| Seed | **100**, minden modellnél ugyanaz |
| Kritikusok | **2 Q-háló**, közös jellemzőkinyerőn |
| Entrópia-hőmérséklet | **α automatikusan hangolt** |
| Entrópia-célérték | **−2** (= −1 × akciótér dimenziója) |
| Felfedezés | **gSDE**, a zaj **8 lépésenként** sorsolódik újra |
| Aktor/kritikus MLP | **500, 300** neuron |

**Miért SAC?**
- *modellmentes*, mert a CARLA fizikáját megtanulni nehéz lenne
- *off-policy*, mert újrahasznosítja a tapasztalatot → kevesebb interakcióból tanul
- *folytonos akciótér*, a DQN csak véges akciók közül választhatna
- az **entrópiatag** fenntartja a felfedezést: az ágens addig marad változatos, amíg ez
  nem rontja érdemben a jutalmat — előzésnél ez lényeges, mert ritka esemény

**Mi az a gSDE és miért?** Állapotfüggő felfedezés: a zaj az aktuális állapottól függ, és
nem minden lépésben sorsolódik újra, hanem **8 lépésenként**. Ettől a kormány- és
gázmozdulatok simábbak, nem remegnek lépésről lépésre.

**Miért két kritikus?** A célértéket a **két becslés kisebbikéből** számolja — ez
csökkenti a Q-értékek túlbecslését.

**Mi az akciótér?** Kormány és gáz → 2 dimenzió. Ezért −2 az entrópia-célérték.

**terminated vs. truncated:**
- *valódi végállapot (terminated)*: ütközés, pályaelhagyás, megállás, túl gyors
- *megszakítás (truncated)*: az útvonal vagy a kitűzött távolság elérése — ilyenkor az
  SB3 a következő állapot **becsült értékével** számol tovább

---

## 3. Szenzorok és adatgyűjtés

| | |
|---|---|
| Kamera renderelés | 640×320 → mentve **160×80**, 8 bites RGB PNG |
| LiDAR | **64 csatorna**, 64×1024 sugár/fordulat, ~30 ezer pont marad |
| LiDAR tárolás | nyers XYZ, méterben, 32 bites float, **ritkítás nélkül** |
| Rögzítés | szimuláció 20 Hz, de **minden 4. lépés** mentve = **5 Hz** |
| Adatmennyiség | **40 211** kamera–LiDAR adatpár |
| Felosztás | **70/15/15** → 28 149 / 6 031 / 6 031 |
| Forgalom gyűjtéskor | **200 jármű** |

**Miért csak minden 4. lépés?** Mert az egymást követő képek szinte azonosak,
fölöslegesen növelnék az adat méretét.

**Miért kellett keverni a felosztás előtt?** Mert az egymást követő képkockák szinte
azonosak — keverés nélkül a halmazok a pálya **más-más szakaszait** tartalmaznák, tehát
a validáció és a teszt más útszakaszt mérne, mint a tanítás.

**Hogy biztosítottad, hogy a kép és a pontfelhő összetartozzon?** A fájlnév a
**szimulációs képkocka sorszáma**.

**Miért tértítetted ki az autót gyűjtés közben?** Mert a CARLA autopilotja mindig
pontosan a sáv közepén halad. Ha csak azt rögzíteném, az adatban nem lennének
kisodródott, ferde helyzetek — pedig az ágensnek pont azokat is kezelnie kell.

**Miért nem nyers adatot adsz az ágensnek?** Három ok:
1. egy kamerakép több tízezer szám, a LiDAR több tízezer pont
2. a pontok **száma és sorrendje lépésről lépésre változik**
3. a nagy bemenet a **visszajátszási puffer** memóriaigényét is megnövelné

**Mihez képest újítás?** A kiinduló munka (Terapaptommakol et al.) a pontfelhőt kézzel
**egyetlen jellemzőre** szűkítette: a jármű előtti sávban a legközelebbi pont
távolságára — összesen 6 érték volt a bemenet. A tanult reprezentáció ezzel szemben a
teljes jelenetet megőrzi.

---

## 4. Autoenkóderek

### Kamera

| Modell | MSE | PSNR |
|---|---|---|
| **Konvolúciós** ✓ | **0,00088** | **30,58 dB** |
| Reziduális | 0,00495 | 23,06 dB |
| VGG-alapú | 0,00278 | 25,56 dB |

**Architektúra:** 4 konvolúciós blokk, 4×4 kernel, 2-es lépésköz, BatchNorm + LeakyReLU.
3×80×160 → 256×5×10 jellemzőtérkép → **128 elemű látens**. Dekóder: tükrözött
transzponált konvolúciók, szigmoid kimenet.

**Miért a konvolúciós?** Mert a legkisebb a hibája — az MSE-je a VGG-alapú harmada.

### LiDAR

| Modell | RankMe | k-NN | Simaság |
|---|---|---|---|
| BEV-alapú | **0,725** | **1,51×** | 1,44× |
| **Gráf-alapú** ✓ | 0,504 | **1,51×** | **1,86×** |

**Point-MAE kiesett:** a **98 304** értékű látense **nagyobb, mint a bemenet** — nem
tömörít.

**Miért nem a rekonstrukciós hibát hasonlítottad?** Mert a két modell **más-más bemenetet
állít vissza** (BEV rács vs. távolságkép), így a hibájuk nem vethető össze.

**Mit mérsz helyette?** Három mérés, **2000 véletlen képkockán**:
- **RankMe**: a látens dimenziók mekkora része hordoz információt (1 = teljes kihasználtság)
- **k-NN konzisztencia**: a látens térben legközelebbi képkockák valóban hasonló jelenetet
  mutatnak-e
- **Időbeli simaság**: két egymást követő képkocka látense hányszor van közelebb
  egymáshoz, mint két véletlené

**Miért a gráf-alapú, ha csak egy mutatóban nyer?** Mert az ágens szempontjából a simaság
a mérvadó: 20 Hz-es hurokban, ha a látens apró változásra nagyot ugrik, az **zaj a
bemeneten**. Mindkettő 0,5 fölött van RankMe-ben, tehát egyik látense sem omlott össze —
ez volt a kizáró kritérium.

**⚠ Ezt NEM mérted le ablációval** (BEV-látenssel tanított ágens vs. gráffal). Ez
indoklás, nem mérés. Mondd ki, ha rákérdeznek.

**Gráf-AE részletei:** távolságkép bemenet, **44×256** rács, FOV (−0,9°, −25°),
hatótáv 1–50 m. A TopoLiDM mintájára, EdgeConv gráfrétegekkel, azimut-körkörös
szomszédsággal. Látens: **16×11×32**. Veszteség: L1, az üres cellák **0,25-ös súllyal**.

**Miért 44 sor és nem 64?** Mert a +10°-os sor csak 19,6 %-ban telik meg — a kép felső
harmada az ég. A FOV-ot −0,9°-ra vágva a sorok 64 → 44, és a kitöltöttség 60,7 %-ról
**90,2 %-ra** nő.

---

## 5. Szenzorfúzió — a saját jellemzőkinyerő

**A probléma, amiért kellett:** a LiDAR látens **5 632 érték** (16×11×32), a kameráé
**128**, az állapot **38**. Amikor az SB3 beépített jellemzőkinyerőjére bíztad az
összefűzést, a nagy látensek **elnyomták** a többi bemenetet, és az ágens nem tanult meg
vezetni.

### A három ág

**LiDAR-ág:** 16×11×32 → **3× azimut-körkörös 3×3 konvolúció** (GroupNorm + ReLU) →
64×3×4 → lapítás (768) → lineáris → **256**.
Ezek a rétegek **az ágenssel együtt tanulnak**, tehát arra hangolódnak, ami a vezetéshez
kell.

**Kamera-ág:** 128 → LayerNorm → lineáris → **256**.
*Miért kell felskálázni?* Mert össze akarod adni a LiDAR 256-os vektorával.

**Állapot-ág:** **38 érték** → skálázás [−1, 1]-re → lineáris + ReLU + LayerNorm → **64**.

**A 38 érték pontosan:** sebesség, **az előző lépés kormány- és gázértéke**, a következő
waypoint irányszöge, a manőver one-hot kódolva, és a **következő 15 waypoint relatív
koordinátái**.

### A fúzió

LiDAR 256 **+** kamera 256 (összeadás) → **LayerNorm(256)** → konkatenálás az állapot
64-gyel → **320** → SAC MLP (500, 300).

**Miért összeadás és nem konkatenálás?** Mert a jellemzőkinyerő végén úgyis
konkatenálással áll össze a kimenet — fölösleges kétszer. Az összeadáshoz nem kell
vetítő réteg (mindkét ág 256), és a vektor mérete sem nő.

**Miért a LayerNorm az összeg után?** Mert a gráf-alapú kódoló kimenete **nem korlátos** —
ez rögzíti a vektor skáláját.

**Hogy súlyozod a két szenzort?** Sehogy, **nem állítod manuálisan** — az ágensre bízod,
honnan nyer több információt.

---

## 6. A jutalomfüggvények

### Minden modellnél közös

| | |
|---|---|
| Ütközés | **−30** |
| Letérés / megállás / sebességtúllépés | **−10** |
| Kormányrángatás | a kormányjel lépésenkénti változásának **2×-ese** |
| Előzési állapotgép | szaggatott vonal + szabad bal sáv kell a kihúzódáshoz |

**Miért háromszoros az ütközés büntetése?** Mert az a legsúlyosabb hiba — és így az
előzésért járó +5 miatt sem éri meg kockáztatni.

**Miért elég a −10, ha az epizód úgyis véget ér?** Mert már az is büntetés, hogy onnantól
nem kap több jutalmat; a −10 ehhez ad továbbit.

**Miért pont 2× a kormánybüntetés?** Hogy egy hirtelen 0,1-es kormánymozdulat **0,2**
levonással járjon — ez a szabad haladásért kapott lépésjutalom **ötöde**. A kis
korrekciókért viszont alig von le.

### R1 — a kiindulási modell

**Szabad úton:** `r = v^1,9 · c · a · s`

| Tényező | Mit mér | 1-nél | 0-nál |
|---|---|---|---|
| `v` | sebesség | 50 km/h | 0 |
| `c` | sávközép | sáv közepén | **2 m-re** |
| `a` | irányszög | útvonal irányában | merőlegesen |
| `s` | simaság | szórás 0 | szórás **0,35 m** |

**Miért szorzat?** Mert bármelyik kiesése **nullázza** a jutalmat — egyszerre kell
mindegyiket teljesítenie.

**Miért 1,9-es kitevő?** Hogy a lassú haladás **aránytalanul** keveset érjen: 25 km/h-nál
v = 0,5, de a tag értéke `0,5^1,9 ≈ 0,27`, nem a fele. Ez erősebben ösztönöz a sebességre
és vonzóbbá teszi az előzést.

**Miért 2 m a sávközép-határ?** Mert ez kicsit több a ~1,75 m-es fél sávszélességnél —
ott már a szomszéd sávban van.

**Hogy számolod a simaságot?** A sávközéptől mért távolság **szórása az utolsó 30
lépésben** (= 1,5 s). 0,35 m-es szórásnál már szemmel is látszik a cikázás.

**Követéskor:** `r = 0,8 · g(h) · m · c · a · s + p_TTC`

- `h` = követési időköz (távolság / **saját** sebesség)
- `g` = **lognormális**, maximuma **2 s**-nál
- `m = min((v_ego+1)/(v_lead+1), 1)`

**Miért 0,8-as szorzó?** Hogy a tökéletes követés kevesebbet érjen a szabad haladásnál
(0,8 vs. 1,0) — így akkor is megéri előzni, ha hibátlanul követ.

**Miért kell az `m`?** Mert a `g` **kijátszható**: a `h` a saját sebességgel oszt, tehát
lassítással is növelhető. Ha az ágens lefékezne és lemaradna, `h` 2 s fölé nőne, a
jutalma mégsem csökkenne. Az `m` a lassabb haladást arányosan bünteti.

**Miért +1 a számlálóban és a nevezőben?** Technikai: álló jármű esetén nullával kellene
osztani.

**Honnan a 2 s és a lognormális alak?** Szakirodalomból. A `h` csak pozitív lehet és
jobbra ferde eloszlású (a vezetők inkább közelebb mennek, mint lemaradnak) — ezt a
**lognormális** írja le legjobban (Mei–Bauer 1993). A `σ = 0,4365` az **NGSIM**
adathalmazon, **1341 valós követési eseményre** illesztett becslés (Zhu et al. 2020).

**Miért 40 m a követési küszöb?** Mert 50 km/h-nál ez ~3 s — több, mint a 2 s-os (~28 m)
követési távolság, így van ideje eldönteni, hogy követ vagy előz.

**A TTC-büntetés:** `0,3 · ln(t/7)`, 7 s alatt lép életbe, **1,5 s-nál** éri el a
**−0,46**-os maximumát.

**Miért van rá szükség, ha ott a `g`?** Mert a pillanatnyi követési időköz lehet ideális
úgy is, hogy az ágens közben **gyorsan közeledik** — ütközésveszély.

**Miért vágod le 1,5 s-nál?** Mert a logaritmus miatt nulla közelében végtelen lenne.

**Miért 0,3-as súly?** Hogy a legnagyobb büntetés (−0,46) kb. a **fele** legyen a jó
követésért járó 0,8-nak — a követési jutalom marad a fontosabb.

**Előzés közben:** `r = (0,3 + 0,6·v^1,9) · l · k`

- `l` = a célsáv közepétől mért távolság
- `k` = az elhúzódó manővert bünteti: **15 s** után, illetve elhaladás után sávonként
  **3 s** után, **4 s alatt** 1-ről **0,3**-ra esik

**Miért 15 s?** Mert 25 km/h sebességkülönbségnél egy előzés kihúzódással és
visszasorolással **~8 s** — egy rendes előzést ez nem büntet.

**Miért nem megy 0,3 alá a `k`?** Mert néha a forgalom miatt tényleg nem tud rögtön
visszasorolni — ilyenkor nem bünteted, csak kevesebbet kap.

**Mi a 0,3-as alaptag?** Hogy a **lassú előzés is jobban megérje**, mint a beragadás.

**A +5 bónusz:** szabályosan befejezett előzésért, **járművenként egyszer**.

**Miért csak egyszer járművenként?** Hogy ne tudja **ki-be sorolgatással gyűjtögetni**.

**Beragadás-büntetés:** `−0,1 · (1 + 0,4(1−v) + 0,4·min(t_b/3, 1))`
→ −0,10 (teljes sebességnél, elején) … **−0,18** (állva, 3 s után)

**Mikor jár?** Ha 25 m-en belül halad az előtte lévő mögött és **nem gyorsabb nála
legalább 5 km/h-val**.

**Ez az egyetlen tag, ami NEM szorzódik a sávtartási tényezőkkel** — mert különben a rossz
sávtartás **csökkentené a büntetést**.

**Rossz sávba menésért:** lépésenként **−0,5**. Ez sokkal több a beragadás −0,18-ánál,
tehát inkább maradjon a lassú mögött, mint hogy szabálytalanul előzzön. **A jelenetet nem
zárod le** — hogy megtanulja, hogyan jöjjön vissza.

### R2 — egyetlen változtatás

A TTC-büntetés, ami az R1-ben **csak akkor járt, ha előzni nem lehetett**, kiterjed az
**előzhető** helyzetre is. Ugyanaz a 7 s küszöb, 0,3 súly, max −0,46.

**Miért sült el rosszul?** Mert 25 km/h sebességkülönbségnél a 7 s **~48 m** — a büntetés
már akkor elindul, amikor 40 m-ről meglátja az előtte lévőt. Ezért inkább lassított és
mögötte maradt.

### R3 — négy változtatás

| | R2 | R3 |
|---|---|---|
| TTC-küszöb | 7 s | **3 s** |
| TTC-súly | 0,3 | **0,6** |
| Előzési jutalom (50 km/h) | 0,90 | **1,10** |
| Lemaradás-büntetés hatótáv | 25 m | **40 m** |
| +5 bónusz | egyben | **+2 / +3** (elhaladás / visszasorolás) |

**Miért sült el még rosszabbul?** A megduplázott súly + rövidebb küszöb miatt **még
ritkábban mert kiállni**. Ez lett az egyetlen modell, ami rendszeresen 60 km/h fölé
gyorsult (24×) — ezt a megemelt előzési lépésjutalom okozta.

### R4 — haladásalapú

`r = v · cos θ · l · k` — az **útvonal mentén megtett elmozdulás**.

**Mi maradt meg?** Az epizódzáró büntetések, az előzési állapotgép és a **+5 bónusz**.
**A TTC-büntetés kikerült** — az ütközést csak a −30-as terminális büntetés tanítja.

**Konkrétan mi változott:** 25 km/h-s jármű mögött lépésenként **+0,5** (az R3 −0,12-je
helyett). Előzés közben 50 km/h-nál 1,0 (az 1,10 helyett).

**Mi az ötlet?** A lassabb jármű mögött a haladás **magától** kevesebbet ér — nem kell
kézi ösztönzés az előzéshez.

### R5 — potenciálalapú formálás

`F(s, s′) = γ·Φ(s′) − Φ(s)` — Ng, Harada, Russell (1999).

`Φ = 2 + 3φ`, ahol `φ = clip((d₀ − d)/(d₀ + 5), 0, 1)`, előzésen kívül **0**.

- `d₀` = a megelőzendő autó távolsága **a kihúzódáskor**
- `d` = aktuális távolság
- Φ kihúzódáskor **2**, az elhaladás végén **5** — pont a +5 bónusz

**Miért kellett?** Mert a +5 csak a manőver végén jár, ~5 s-mal a kihúzódás után. A
**0,98-as diszkont** mellett ez a kihúzódás pillanatában már csak **~0,66**-ot ér — az
ágens alig érzi.

**Miért nem ront el semmit?** Mert a Φ úgy viselkedik, mint egy **szint**: ha nő, jutalom;
ha csökken, ugyanannyit veszít. Ha visszaér oda, ahonnan indult, **nem nyert semmit**. Ng
és társai bizonyítják, hogy ilyen alakú plusz jutalom mellett az ágens **ugyanazt a
viselkedést** tanulja meg a legjobbnak.

**Miért +5 m a nevezőben?** Mert a `φ` akkor lesz 1, amikor a megelőzött autó már **5
m-rel mögötte** van.

**Mi az indíték az egészre?** Az R1–R3 kézi tagjai **megkerülhetők** voltak: az R2-ben a
25–40 m-es lemaradás is jutalmat ért, így előzés helyett **megérte lemaradni**.

---

## 7. A kiértékelés

| | |
|---|---|
| Útvonalak | **10** előre rögzített (5 autópálya + 5 városból autópályára) |
| Hossz | **1,05 – 6,09 km** |
| Ismétlés | **5×** útvonalanként |
| Összesen | **250 bejárás** (modellenként **50**) |
| Megtett út | **169–171 km** modellenként |
| Akciók | **determinisztikus** |
| Súlyok | a tanítás **végén** mentett |
| Célba ért | **248 / 250** |

**Hogy biztosítod az azonos feltételt?** A forgalom véletlengenerátorának **kezdőértékeit
magad állítottad** — adott körben minden modell **pontosan ugyanazt a forgalmat** kapja.

**Mi történik hibánál?** Nem áll le. Letérés, 5 s megállás vagy 60 km/h fölé gyorsulás
esetén **felírod a hibát, visszahelyezed, a jelenet megy tovább**. **10 visszahelyezés
után** leáll a jelenet.

**Mikor nem hiba a megállás?** Ha a saját sávjában van előtte jármű — ott a várakozás a
helyes viselkedés.

**Az ütközés jár visszahelyezéssel?** **Nem.** Azt felírod, de a jelenet megy tovább.

**Miért nem hasonlítod a jutalmat?** Mert minden modell **más jutalomfüggvénnyel**
tanult — a jutalomalapú összehasonlítás nem lenne reprezentatív. Helyette
jutalomfüggetlen, **fizikai** mérőszámokat nézel.

**A kétlépéses átlagolás:**
1. útvonalanként: az 5 bejárás összes eseménye **osztva** az 5 bejárás összes
   kilométerével → km-re vetített érték
2. a tíz útvonalérték **számtani átlaga**

**Miért?** Mert az útvonalak hossza 1,05–6,09 km. Egyetlen összesítésben a 6 km-es
szakasz **közel hatszor** annyit nyomna. Így minden útvonal **egyforma súllyal** számít.

---

## 8. Az eredmények — fejből

### Sebesség

| | Átlagseb. | Menetidő | Célba ért |
|---|---|---|---|
| R1 | 39,6 km/h | 3121 s | 49/50 |
| R2 | 35,5 | 3543 | 50/50 |
| R3 | **31,7** | **3941** | 50/50 |
| R4 | 39,4 | 3174 | 49/50 |
| R5 | **40,1** | **3056** | **50/50** |

### Előzés

| | Elhaladás | Szabályos | Szabálytalan | Szab. arány | Sikeres kísérlet |
|---|---|---|---|---|---|
| R1 | 1009 | 948 | 61 | **94,0 %** | 74,6 % |
| R2 | 827 | 772 | 55 | 93,3 % | 60,4 % |
| R3 | **556** | 505 | 51 | 90,8 % | **45,8 %** |
| R4 | **1018** | 919 | 99 | 90,3 % | 72,3 % |
| R5 | 1017 | 934 | 83 | 91,8 % | **75,0 %** |

**Fontos minta:** autópályán **minden** ágens szabályosabb (95–98 %), mint a városból
indulókon (80–88 %).

### Kormányzás és sávtartás

| | Kormányváltozás | Sávközép |
|---|---|---|
| R1 | **0,028** | **86,4 %** |
| R2 | 0,037 | 73,7 % |
| R3 | 0,030 | **59,8 %** |
| R4 | 0,028 | 72,5 % |
| R5 | 0,029 | 70,3 % |

### Biztonság

| | Elütött | Elütési arány | Visszahelyezés | Letért | Megállt | Túl gyors |
|---|---|---|---|---|---|---|
| R1 | 119 | 11,8 % | 43 | 20 | 16 | 7 |
| R2 | **85** | 10,3 % | **12** | 9 | 3 | 0 |
| R3 | 84 | **15,1 %** | 53 | 15 | 14 | **24** |
| R4 | 113 | 11,1 % | **58** | **51** | 5 | 2 |
| R5 | 104 | **10,2 %** | 26 | 22 | 2 | 2 |

**Miért van kétféle elütésszám?** Mert az R3 abszolút számban kevesebbet ütött el (84), de
alig előzött (556) — ezért 15,1 % az aránya. Ezért közlöd **mindkettőt**.

**A visszahelyezések szinte mind a városból induló útvonalakon** voltak — autópályán az
R3-at leszámítva egyik ágenst sem kellett háromnál többször visszahelyezni.

### A két sikertelen bejárás

**R1**, 2. útvonal, 2. kör, **0,54 km** után: mind a 10 visszahelyezés **ugyanazon a
ponton**, egy kanyarban, a kormány 0,5–0,7-es kitérésben.

**R4**, 4. útvonal, 4. kör: a 10 letérésből **2 az 1,52 km-nél**, **8 a 3,16–3,35 km-es
szakaszon** — az **S alakú autópálya-felhajtón**. Egy előtte 2–3 m-re haladó jármű mögött
újra és újra kisodródott. Itt a többi modell legfeljebb **egyszer** tért le.

---

## 9. A következtetések, amiket vállalsz

1. **Nem érdemes minden helyzetre külön kézi tagot írni** — kiolthatják egymást, és az
   ágens a **szabály megkerülését** tanulja meg, nem a célt.
2. **Az előzés közbeni közeledésbüntetés (TTC) nem működött** — óvatosabbá tett, nem
   bátrabbá.
3. **A haladás jutalmazása jobban működik** — a lassabb autó mögött maradás magától
   kevesebbet ér.
4. **Az előzést az első lépéstől érdemes jutalmazni** — ezt mutatja az R5. Az eljárás nem
   változtat azon, mi a legjobb stratégia, csak **hamarabb jelez**.
5. **Nincs egyértelmű győztes** — az R1 szabályosabb és jobb sávtartású, az R5 gyorsabb és
   megbízhatóbb.

**Továbblépés (amit mondj, ha kérdezik):** az R5 irányában, visszatéve a sávtartást, hogy
ne rángassa fölöslegesen a kormányt; és mélyebben utánajárni, mi okozza az ütközéseket.

---

## 10. Amit NEM mértél — mondd ki, ha kérdezik

- **Több seed** — modellenként egy betanítás, seed = 100
- **Abláció a szenzorokra** — nincs LiDAR nélküli vagy kamera nélküli modell
- **Abláció a LiDAR-AE választására** — nem tanítottál külön ágenst BEV-látenssel
- **Más térkép** — csak Town04
- **Szabályalapú baseline** — nincs mihez viszonyítani az RL-t
- **Más RL-algoritmus** — csak SAC

Ezek nem szégyen. A szégyen az, ha mértnek állítod be őket.
