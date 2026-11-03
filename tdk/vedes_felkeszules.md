# TDK védés — felkészülés a legrosszindulatúbb bírálóra

Az alábbi kérdések a dolgozat tényleges gyenge pontjaira irányulnak. Mindegyiknél ott
a kérdés, hogy miért fáj, és egy védhető válasz. A válaszok sosem tagadják a problémát —
a bizottság azonnal kiszúrja, ha mellébeszélsz. Ismerd el, keretezd, és mondd meg, mit
jelent az eredményre nézve.

---

## 1. A legveszélyesebb: n = 1 modellenként

> **„Minden modellt pontosan egyszer tanított be, egyetlen seeddel. Honnan tudja, hogy
> az R2 és az R3 visszaesése nem egyszerűen a tanítás szórása? Az SAC köztudottan
> seed-érzékeny."**

**Miért fáj:** ez a dolgozat legnagyobb metodikai sebezhetősége. A 250 bejárás a
*kiértékelés* szórását kezeli, a *tanítás* szórását nem. Öt modell, öt betanítás, egy
seed (100) mindegyikhez. A mély RL irodalmában rutinszerű, hogy ugyanaz a konfiguráció
3–5 seeddel 10–20 %-os szórást mutat a végső teljesítményben — az R4 → R5 különbséged
(39,4 → 40,1 km/h) ennél jóval kisebb.

**Válasz:**

> „Igaza van, ez a dolgozat legnagyobb korlátja, és a szövegben nem is állítok többet,
> mint amit egy seed megenged. Egy betanítás 12 óra, öt modell öt seeddel 25 futás,
> vagyis 300 GPU-óra — ez nem fért bele. Amit viszont a seed-szórás nem magyaráz meg:
> az R1 → R3 esés nem néhány százalék, hanem **1009-ről 556-ra**, az elhaladások
> száma közel a felére. A sikeres kísérletek 74,6-ról 45,8 %-ra. Ez a nagyságrend
> kívül esik azon, amit a tanítás zaja reálisan okoz.
>
> Az R4 és az R5 közötti különbséget viszont **magam sem állítom szignifikánsnak** —
> a dolgozatban is azt írom, hogy a sebességkülönbség nem számottevő, és hogy a
> potenciálalapú formálás az elmélet szerint épp nem változtatja meg az optimális
> stratégiát. Amit az R5-nél valódi különbségnek tartok, az a visszahelyezések száma:
> 58-ról 26-ra."

**Amit NE mondj:** hogy a 250 bejárás kompenzálja. Nem kompenzálja, más dolgot mér, és
a bíráló pont ezt várja, hogy belesétálj.

---

## 2. Az R5 előnye eltűnik, ha megnézzük a számokat

> **„Azt állítja, az R5 a legjobb. De az elütési aránya 10,2 %, az R2-é 10,3 % — ez
> egy századszázalék. A sebességkülönbség az R4-hez képest 0,7 km/h. Mi itt a tényleges
> eredmény?"**

**Miért fáj:** mert az összegzésben magad is leírod, hogy „legjobbat nem lehet
egyértelműen kimondani". Ha a dián vagy az előadásban ennél erősebbet állítasz, a
bíráló szembeállítja a saját szövegeddel.

**Válasz:**

> „Pontosan ezért nem is állítom, hogy az R5 egyértelműen a legjobb — az összegzésben
> kifejezetten azt írom, hogy a két legjobb modell más-más téren erős. Az R1
> szabályosabban előz (94,0 % a 91,8 % ellen) és sokkal jobban tartja a sávot
> (86,4 % a 70,3 % ellen). Az R5 gyorsabb, mind az 50 bejárása célba ért, és feleannyi
> visszahelyezés kellett.
>
> Amit az R5 kapcsán valóban állítok, az nem az, hogy a legjobb modell, hanem hogy
> a **potenciálalapú formálás működik úgy, ahogy az elmélet mondja**: az R4-hez képest
> a sebesség, az előzésszám és az ütközések érdemben nem változtak — ahogy Ng és
> társai tétele alapján várható —, de a jutalom korábbi érkezése határozottabb
> manővereket eredményezett. Ez a megerősítés, nem a rangsor."

---

## 3. Az előzés „szabályosságát" te definiáltad

> **„A szabályos elhaladás arányát a saját előzési állapotgépe dönti el. Ön írta a
> jutalomfüggvényt is és a mérőszámot is. Nem körkörös ez?"**

**Miért fáj:** komoly kérdés. A „szabályos elhaladás" és a „sikeres kísérlet" is a te
állapotgéped kimenete, ugyanaz a logika, ami az R1–R3 jutalmában is benne van. Elvileg
lehetne olyan modell, ami pont az állapotgéped vakfoltjait használja ki.

**Válasz:**

> „Az állapotgép **minden modellnél azonos**, és a kiértékeléskor ugyanaz fut, mint
> tanításkor — tehát ha torzít, minden modellt ugyanúgy torzít, a sorrendet nem
> változtatja meg. A körkörösség valódi kockázata ott lenne, ha csak ezt a mérőszámot
> nézném. Ezért mérek mellette olyan dolgokat is, amelyeket **nem én definiálok,
> hanem a szimulátor ad**: az átlagsebességet, a menetidőt, a CARLA ütközésérzékelőjétől
> jövő elütéseket, és a sávközéptől mért távolságot. Ezek egyike sem a jutalomfüggvény
> vagy az állapotgép terméke.
>
> És pont ezek támasztják alá ugyanazt a képet: az R3 nemcsak a saját állapotgépem
> szerint volt a legrosszabb, hanem 31,7 km/h-val a leglassabb is, és egyedüliként
> gyorsult rendszeresen 60 km/h fölé."

---

## 4. A LiDAR-választás indoklása ellentmondásos

> **„A gráf-alapú autoenkódert választotta, pedig a RankMe-je 0,504, a BEV-é 0,725. A
> k-NN konzisztenciában döntetlen. Egyetlen mutatóban jobb — a simaságban. Miért pont
> az dönt?"**

**Miért fáj:** három mérésből kettőben nem nyert a választott modell. Ez a dolgozatban
így is szerepel, őszintén — de egy indoklást követel.

**Válasz:**

> „Azért a simaság dönt, mert az ágens szempontjából ez a mérvadó. A RankMe azt méri,
> hány dimenzió hordoz információt — ez egy reprezentáció *kapacitását* jellemzi. A
> simaság azt, hogy két egymást követő képkocka látense mennyire marad egymáshoz közel.
> Egy 20 Hz-es vezérlési hurokban az ágens lépésenként kap új megfigyelést; ha a látens
> apró geometriai változásra nagyot ugrik, az számára **zaj a bemeneten**, amiből
> nehezebb stabil stratégiát tanulni.
>
> Hozzáteszem: mindkét modell 0,5 fölött van RankMe-ben, tehát **egyik látense sem
> omlott össze** néhány dimenzióra — ez volt a kizáró kritérium, és azt mindkettő
> teljesíti. Az ezen felüli kapacitástöbbletet kevésbé tartottam értékesnek, mint a
> időbeli stabilitást."

**Ha rákérdez, hogy mérted-e:** itt légy óvatos. Nem futtattál ablációt két külön
betanított ágenssel (BEV-látenssel vs. gráf-látenssel), tehát ezt **nem mérted, hanem
indokoltad**. Mondd ki: *„Ezt nem mértem le végponttól végpontig, ez egy indoklás a
reprezentáció tulajdonságai alapján — az ablációt a további munka részének tartom."*

---

## 5. A Point-MAE kiejtése

> **„A Point-MAE-t azzal ejtette ki, hogy a látense nagyobb, mint a bemenet. Ez a
> modell hibája vagy a konfigurációé?"**

**Miért fáj:** a 98 304 értékű látens a te `num_group` / `group_size` beállításodból
jön, nem a módszer elvi korlátja. Egy transzformer-alapú pontfelhő-enkóder elvileg
tömöríthetne is.

**Válasz:**

> „A konkrét konfigurációban nem tömörített, és ezt a dolgozatban így is írom le —
> ezen a beállításon nem volt értelme folytatni. Elvi állítást nem teszek arról, hogy
> a Point-MAE alkalmatlan a feladatra; azt állítom, hogy az általam betanított
> változat nem adott a bemenetnél kisebb reprezentációt, márpedig a cél épp a
> tömörítés volt."

---

## 6. Overfitting a kiértékelő útvonalakra

> **„Ugyanazon a Town04 térképen tanított és értékelt. A tíz útvonal is ugyanabból a
> térképből van. Mennyire általánosít a modellje?"**

**Miért fáj:** nincs külön tesztpálya. Az ágens ugyanazt a geometriát látta tanítás
közben, amin mérted.

**Válasz:**

> „A térkép ugyanaz, de a helyzetek nem: a forgalom kezdőhelyzete bejárásonként
> véletlen, tehát a modell tanítás közben sosem látta pont azokat a forgalmi
> szituációkat, amelyeken mérem. Amit **nem állítok**, az az, hogy a modell más
> térképre vagy valós útra átvihető — ehhez külön térképen kellene mérni, és ez a
> dolgozat hatókörén kívül esik.
>
> Egy adalék az általánosításhoz: a városból autópályára vezető útvonalakon minden
> modell rosszabbul teljesített, mint a tiszta autópályán — a szabályos arány
> 95–98 %-ról 80–88 %-ra esik. Ez épp azt mutatja, hogy a modellek **nem tanulták meg
> kívülről a pályát**, hanem a nehezebb geometrián valóban nehezebben boldogulnak."

---

## 7. A visszahelyezés mint mérőszám

> **„Visszahelyezi az autót és folytatja a jelenetet. Egy valós rendszerben a letérés
> után nincs folytatás. Nem szépíti ez az eredményeket?"**

**Válasz:**

> „Nem szépíti, mert a visszahelyezést **külön mérőszámként felírom** — pontosan azt
> mutatja, hányszor kellett volna egy embernek beavatkoznia. Ez önálló biztonsági
> metrika, nem elfedett hiba. Ha az első hibánál leállítanám a jelenetet, két dolgot
> veszítenék: az útvonal hátralévő részén mért előzéseket és ütközéseket, és azt az
> információt, hogy a hiba egyszeri volt-e vagy ismétlődik. Épp ez utóbbiból derült
> ki a dolgozat két legérdekesebb hibaesete: az R1 a 2. útvonalon mind a tízszer
> ugyanabban a kanyarban akadt el, az R4 pedig az S alakú felhajtón sodródott ki
> újra és újra."

---

## 8. Az ütközési arány nevezője

> **„Az elütési arányt az elhaladások számához viszonyítja. Az R3 kevesebbet is ütött
> el (84) és kevesebbet is előzött (556) — így lett 15,1 %. Nem torzít ez a nevező?"**

**Miért fáj:** a 11,8 % (R1) és a 15,1 % (R3) nem ugyanazon az alapon áll, mert az
elhaladások száma kétszeresen tér el.

**Válasz:**

> „Ezért közlöm **mindkettőt**: az abszolút számot és az arányt is. Önmagában egyik
> sem elég. Abszolút számban az R3 ütött el a legkevesebbet (84), de úgy, hogy alig
> előzött — ezért van szükség a kitettséghez viszonyított arányra is. A modellek
> nagyjából azonos utat tettek meg, 169–171 km-t, így a kilométerre vetített értékek
> is összevethetők, és azokat is kiszámoltam."

---

## 9. Miért pont SAC? Miért nincs baseline?

> **„Nincs összehasonlítás sem egy másik algoritmussal, sem egy klasszikus
> szabályalapú vezérlővel. Honnan tudjuk, hogy az RL itt egyáltalán jobb?"**

**Válasz:**

> „A dolgozat kérdése nem az volt, hogy az RL jobb-e egy szabályalapú vezérlőnél,
> hanem az, hogy **a jutalomfüggvény megfogalmazása hogyan hat** a tanult viselkedésre.
> Ehhez az öt jutalomváltozat egymás baseline-ja, azonos algoritmus és azonos
> hiperparaméterek mellett. A SAC választását a folytonos akcióteret kezelő, mintahatékony
> off-policy algoritmusok közül az indokolja, hogy az entrópiaszabályozása a tanulás
> során fenntartja a felfedezést — előzésnél ez lényeges, mert a manőver elkezdése
> ritka esemény.
>
> Egy szabályalapú baseline valóban hasznos referencia lenne, és ezt a további munka
> részének tartom."

---

## 10. A kétlépéses átlagolás

> **„Miért nem egyszerűen az összes eseményt osztja az összes kilométerrel?"**

Erre jó a válaszod a dolgozatban, ismételd el magabiztosan:

> „Mert az útvonalak hossza 1,05 és 6,09 km között változik. Egyetlen összesítésben a
> hatkilométeres szakasz közel hatszor annyit nyomna, mint az egykilométeres, vagyis
> a leghosszabb útvonal geometriája dominálná az eredményt. A kétlépéses átlagolással
> minden útvonal egyforma súllyal számít."

---

## 11. Mit tanult az ágens valójában?

> **„Van bármilyen bizonyítéka arra, hogy az ágens a LiDAR-t használja? Lehet, hogy
> csak a waypointokból és a sebességből vezet, a szenzorfúzió meg dekoráció."**

**Miért fáj:** ez a legkellemetlenebb kérdés, mert **nincs rá ablációd**. Nem futtattál
olyan modellt, ami csak kamerával vagy csak állapotvektorral tanult.

**Válasz — vállald fel:**

> „Ezt a dolgozatban nem mértem le, és nem is állítom. Ablációval lehetne eldönteni:
> ugyanaz a jutalom, ugyanaz a tanítás, LiDAR-ág nélkül. Közvetett jelzésem van rá —
> az ütközések és az előzési viselkedés érzékenyen reagáltak a jutalom változtatására,
> ami nehezen magyarázható pusztán waypoint-követéssel —, de ez nem bizonyíték.
> Az ablációs mérést a további munka legfontosabb elemének tartom."

**Ez a legjobb válasz, amit adhatsz:** őszinte, pontos, és mutatja, hogy érted a
különbséget bizonyíték és sejtés között. A bíráló ezt jobban díjazza, mint egy
kikerülő mondatot.

---

## Általános taktika

1. **Soha ne tagadj valós korlátot.** Ismerd el, majd mondd meg, mit jelent az
   eredményre — és mit nem.
2. **Különböztesd meg a mért és az indokolt állításokat.** „Ezt mértem" / „Ezt nem
   mértem, ez indoklás." A bizottság a kettő összekeverését bünteti, nem a hiányt.
3. **A nagy különbségekre támaszkodj, ne a kicsikre.** 1009 → 556 védhető. 39,4 →
   40,1 nem.
4. **Használd a saját negatív eredményeidet.** Az R2 és az R3 kudarca a dolgozat
   legerősebb része: megmutatja, hogy nem a kedvező eredményt kerested.
5. **„A további munka része."** Nem kifogás, hanem a kutatás normális nyelve — de csak
   akkor hiteles, ha konkrétan megmondod, mit csinálnál (abláció, több seed,
   szabályalapú baseline, másik térkép).
