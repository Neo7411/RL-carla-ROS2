# 05 — `carla_env/navigation/` — útvonaltervezés

Ez a modul a CARLA hivatalos PythonAPI-jából származik (Intel Labs, MIT licenc), kisebb
módosításokkal. **A projekt csak egy kis részét használja aktívan.**

## Melyik fájl használatban van?

| Fájl | Használt? | Mire |
|------|-----------|------|
| `planner.py` | ✅ **igen** | `compute_route_waypoints()` — a `new_route()` hívja |
| `global_route_planner.py` | ✅ **igen** | A\* gráfkeresés |
| `global_route_planner_dao.py` | ✅ **igen** | térkép-adathozzáférés |
| `local_planner.py` | ⚠ részben | **csak a `RoadOption` enumért** — a `LocalPlanner` osztály nem fut |
| `controller.py` | ❌ nem | PID szabályozók (a `LocalPlanner` használná) |
| `agent.py`, `basic_agent.py`, `roaming_agent.py` | ❌ nem | klasszikus, nem-RL ágensek |

> ⚠ **A három `*agent.py` fájl jelenleg importálhatatlan!** Ezekben a fájlokban
> `from agents.navigation.local_planner import LocalPlanner` szerepel — az eredeti CARLA
> `agents` csomagra hivatkoznak, nem a helyi `carla_env.navigation`-re. Mivel semmi nem
> importálja őket, ez nem okoz futáshibát, de tudni kell róla. Ha használni akarod őket,
> át kell írni az importokat.

---

## A nagy kép: hogyan lesz útvonal?

```
new_route()                         [carla_route_env.py:167]
   │
   ▼
compute_route_waypoints(map, start_wp, end_wp, resolution=1.0)   [planner.py:12]
   │
   ├─► GlobalRoutePlannerDAO(map, 1.0)          ← adathozzáférés
   ├─► GlobalRoutePlanner(dao)
   │      └─ setup()                            ← GRÁF ÉPÍTÉSE (lassú!)
   │           ├─ _build_graph()                   csomópontok + élek
   │           ├─ _find_loose_ends()               zsákutcák kezelése
   │           └─ _lane_change_link()              sávváltás élek (0 költséggel)
   │
   ├─► grp.trace_route(start_loc, end_loc)      ← A* KERESÉS
   │      ├─ _path_search()                        nx.astar_path
   │      ├─ _turn_decision()                      LEFT/RIGHT/STRAIGHT eldöntése
   │      └─ waypointok kigyűjtése 1 m-enként
   │
   └─► manőver-előrehozás 5 ponttal              [planner.py:115-124]

eredmény: [(waypoint, RoadOption), (waypoint, RoadOption), ...]
```

---

# 1. `planner.py` — a belépési pont (126 sor)

## `compute_route_waypoints(world_map, start_waypoint, end_waypoint, resolution=1.0, plan=None)`

Két üzemmódja van:

### A) `plan=None` (ezt használjuk) — 33–42. sor

```python
if plan is None:
    dao = GlobalRoutePlannerDAO(world_map, resolution)
    grp = GlobalRoutePlanner(dao)
    grp.setup()

    route = grp.trace_route(
        start_waypoint.transform.location,
        end_waypoint.transform.location)
```

Automatikus útkeresés A\*-gal.

> ⚠ **Teljesítmény-probléma:** a `grp.setup()` **minden egyes útvonalnál újraépíti az egész
> térkép gráfját**. Ez a `new_route()` minden hívásánál megtörténik — tehát minden
> epizódban, sőt epizódon belül is minden útvonal-váltásnál. A Town02 kicsi, így ez
> "csak" néhány száz ms, de ez tiszta pazarlás: a gráf a térképtől függ, ami sosem változik.
> **Ez a legkönnyebben elérhető gyorsítás a projektben** — lásd
> [07-tovabbfejlesztes.md](07-tovabbfejlesztes.md).

### B) `plan=[RoadOption.LEFT, RoadOption.STRAIGHT, ...]` — 43–113. sor

Manuális mód: nem A\*-t használ, hanem megadod, hogy minden kereszteződésnél melyik irányba
menjen. **Ezt a projekt nem használja**, de érdemes érteni.

```python
route = []
current_waypoint = start_waypoint
for i, action in enumerate(plan):
    # 1. Menj a következő elágazásig
    wp_choice = [current_waypoint]
    while len(wp_choice) == 1:
        current_waypoint = wp_choice[0]
        route.append((current_waypoint, RoadOption.LANEFOLLOW))
        wp_choice = current_waypoint.next(resolution)
        ...
```

A `waypoint.next(distance)` **listát** ad vissza: ha egy elem van benne, egyenes út;
ha több, elágazás.

```python
    # 2. Elágazásnál: válaszd ki a helyes ágat keresztszorzattal
    direction = 0
    if   action == RoadOption.LEFT:     direction = 1
    elif action == RoadOption.RIGHT:    direction = -1
    elif action == RoadOption.STRAIGHT: direction = 0

    for wp_select in wp_choice:
        v_select = vector(current_location, wp_select.transform.location)
        if direction == 0:
            cross = abs(np.cross(v_current, v_select)[-1])   # legkisebb eltérés = egyenes
        else:
            cross = direction * np.cross(v_current, v_select)[-1]
        if cross < select_criteria:
            select_criteria = cross
            current_waypoint = wp_select
```

A keresztszorzat z-komponense mondja meg, hogy egy ág balra vagy jobbra tér el a jelenlegi
iránytól.

## A manőver-előrehozás (115–124. sor) ★ — ez a projekt saját módosítása

```python
num_wp_to_extend_actions_with = 5
action = route[0][1]
for i in range(1, len(route)):
    next_action = route[i][1]
    if next_action != action:
        if next_action != RoadOption.LANEFOLLOW:
            for j in range(num_wp_to_extend_actions_with):
                route[i-j-1] = (route[i-j-1][0], route[i][1])
    action = next_action
```

**Mit csinál?** Ha az útvonalon a 100. pontnál kezdődik egy `LEFT` manőver, akkor a
95–99. pontokat is `LEFT`-re állítja.

**Miért kell?** Mert az ágens állapotában szerepel a `maneuver` mező. Ha a kanyar jelzése
csak a kereszteződés *belsejében* jelenne meg, az ágens túl későn tudná meg, hogy kanyarodni
kell — a bekanyarodáshoz már korábban lassítani és pozicionálni kell.

**5 waypoint × 1 m = 5 méteres előjelzés.** Ezt az értéket lehet hangolni: nagyobb érték
korábbi figyelmeztetést ad.

```
útvonal:  ...  LANEFOLLOW LANEFOLLOW LANEFOLLOW LEFT LEFT LEFT ...
                                     ↑ eredetileg itt kezdődött a LEFT

utána:    ...  LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT ...
               └──── 5 ponttal előrehozva ────┘
```

---

# 2. `global_route_planner_dao.py` — adathozzáférés (72 sor)

**DAO = Data Access Object.** Elválasztja a "honnan jönnek az adatok" kérdést a
"mit csinálunk velük" logikától.

## `get_topology()` (26–62. sor)

```python
for segment in self._wmap.get_topology():
    wp1, wp2 = segment[0], segment[1]
    l1, l2 = wp1.transform.location, wp2.transform.location
    x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)
```

A CARLA `get_topology()` **pár waypointot** ad vissza: minden útszakasz eleje és vége.
A kerekítés (`np.round(..., 0)`, tehát egész méterre) azért kell, hogy az azonos
csomópontok **azonos kulcsot** kapjanak a szótárban (lebegőpontos pontatlanság ellen).

```python
    seg_dict['path'] = []
    endloc = wp2.transform.location
    if wp1.transform.location.distance(endloc) > self._sampling_resolution:
        w = wp1.next(self._sampling_resolution)[0]
        while w.transform.location.distance(endloc) > self._sampling_resolution:
            seg_dict['path'].append(w)
            w = w.next(self._sampling_resolution)[0]
    else:
        seg_dict['path'].append(wp1.next(self._sampling_resolution/2.0)[0])
```

**A `path` a szakasz "kitöltése"**: 1 méterenkénti waypointok a szakasz eleje és vége között.
Ezért lesz a végső útvonalon 1 m-enként pont.

A visszaadott szótár szerkezete:
```python
{
  'entry':    waypoint,       # a szakasz eleje
  'exit':     waypoint,       # a szakasz vége
  'entryxyz': (x, y, z),      # kerekített koordináták (gráf-kulcs)
  'exitxyz':  (x, y, z),
  'path':     [waypoint, ...] # köztes pontok 1 m-enként
}
```

---

# 3. `global_route_planner.py` — A\* keresés (404 sor)

## `setup()` (37–45. sor)

```python
self._topology = self._dao.get_topology()
self._graph, self._id_map, self._road_id_to_edge = self._build_graph()
self._find_loose_ends()
self._lane_change_link()
```

## `_build_graph()` (47–103. sor)

Egy **irányított gráfot** (`nx.DiGraph`) épít:

```
    csomópont = útszakasz vége/kezdete (x, y, z)
    él        = maga az útszakasz
```

```python
graph.add_edge(
    n1, n2,
    length=len(path) + 1,          # ← ez az A* KÖLTSÉGE (≈ méterben, mert 1 m felbontás)
    path=path,                     # a köztes waypointok
    entry_waypoint=entry_wp, exit_waypoint=exit_wp,
    entry_vector=...,              # belépő irányvektor
    exit_vector=...,               # kilépő irányvektor
    net_vector=...,                # a húr iránya (belépéstől kilépésig)
    intersection=intersection,     # kereszteződés-e
    type=RoadOption.LANEFOLLOW)
```

A három vektor a **kanyar-döntéshez** kell (lásd `_turn_decision`).

Két segédszótár:
- `id_map`: `{(x, y, z): csomópont_id}` — koordinátából csomópont
- `road_id_to_edge`: `{road_id: {section_id: {lane_id: (n1, n2)}}}` — CARLA út-azonosítóból él

## `_find_loose_ends()` (105–147. sor)

Néhány útszakasznak "lelóg a vége" (nem csatlakozik semmihez a topológiában — pl. a térkép
széle). Ezeket **negatív azonosítójú** csomópontokként (`n2 = -1*count_loose_ends`) adja
hozzá, hogy legalább be lehessen hajtani rájuk.

## `_lane_change_link()` (169–210. sor)

```python
if bool(waypoint.lane_change & carla.LaneChange.Right) and not right_found:
    next_waypoint = waypoint.get_right_lane()
    if next_waypoint is not None and next_waypoint.lane_type == carla.LaneType.Driving and \
       waypoint.road_id == next_waypoint.road_id:
        next_road_option = RoadOption.CHANGELANERIGHT
        ...
        self._graph.add_edge(..., path=[], length=0, type=next_road_option, ...)
```

**`length=0`** — a sávváltás **ingyenes** az A\* szempontjából. Így az útkereső bármikor
válthat sávot, ha az rövidebb utat ad.

> ⚠ **Ez a kód `RoadOption.CHANGELANERIGHT`-ot és `CHANGELANELEFT`-et használ, de a
> `local_planner.py`-ban lévő `RoadOption` enum ezeket NEM tartalmazza!** Ott csak
> `LANEFOLLOW=0, LEFT=1, RIGHT=2, STRAIGHT=3, VOID=-1` van. Az eredeti CARLA verzióban
> van `CHANGELANELEFT=5, CHANGELANERIGHT=6` is, de ez a repó egy régebbi/csonkolt
> enumot használ.
>
> **Következmény:** ha az A\* sávváltó élt választana, `AttributeError: CHANGELANERIGHT`
> hibát kapnál. A Town02 egysávos utcái miatt ez a gyakorlatban **nem fordul elő**,
> de ha Town04-re (autópálya, több sáv) váltanál, azonnal elszállna. **Ezt tudni kell,
> mielőtt térképet váltasz!**

## `_path_search(origin, destination)` (221–237. sor) ★

```python
start, end = self._localize(origin), self._localize(destination)

route = nx.astar_path(
    self._graph, source=start[0], target=end[0],
    heuristic=self._distance_heuristic, weight='length')
route.append(end[1])
```

**Az A\* algoritmus:**
- `weight='length'` — a tényleges eddigi költség (g)
- `heuristic=_distance_heuristic` — a becsült hátralévő költség (h) = **légvonalbeli
  távolság** a célig

```python
def _distance_heuristic(self, n1, n2):
    l1 = np.array(self._graph.nodes[n1]['vertex'])
    l2 = np.array(self._graph.nodes[n2]['vertex'])
    return np.linalg.norm(l1-l2)
```

A légvonalbeli távolság **elfogadható (admissible) heurisztika**: sosem becsli túl a valós
utat, mert az úton menni mindig legalább annyi, mint légvonalban. Ez garantálja, hogy az
A\* az optimális utat találja meg.

## `_turn_decision(index, route, threshold=5°)` (263–322. sor) ★★

Ez a legbonyolultabb függvény: eldönti, hogy egy kereszteződésben **balra, jobbra vagy
egyenesen** kell menni.

```python
calculate_turn = current_edge['type'].value == RoadOption.LANEFOLLOW.value and \
                 not current_edge['intersection'] and \
                 next_edge['type'].value == RoadOption.LANEFOLLOW.value and \
                 next_edge['intersection']
```

Csak akkor számol kanyart, ha **most nem** kereszteződésben vagyunk, de **a következő él
kereszteződés**. Tehát a kereszteződés *előtt* álló élen dönt.

```python
cv, nv = current_edge['exit_vector'], next_edge['net_vector']
cross_list = []
for neighbor in self._graph.successors(current_node):
    select_edge = self._graph.edges[current_node, neighbor]
    if select_edge['type'].value == RoadOption.LANEFOLLOW.value:
        if neighbor != route[index+1]:
            sv = select_edge['net_vector']
            cross_list.append(np.cross(cv, sv)[2])       # a TÖBBI lehetőség
next_cross = np.cross(cv, nv)[2]                          # a VÁLASZTOTT irány

deviation = math.acos(np.clip(np.dot(cv, nv)/(np.linalg.norm(cv)*np.linalg.norm(nv)), -1.0, 1.0))
```

**A trükk:** nem abszolút szöget néz, hanem **összehasonlítja a többi lehetőséggel**:

```python
if deviation < threshold:                              # < 5° eltérés
    decision = RoadOption.STRAIGHT
elif cross_list and next_cross < min(cross_list):      # a legbalosabb opció
    decision = RoadOption.LEFT
elif cross_list and next_cross > max(cross_list):      # a legjobbosabb opció
    decision = RoadOption.RIGHT
elif next_cross < 0:
    decision = RoadOption.LEFT
elif next_cross > 0:
    decision = RoadOption.RIGHT
```

**Miért relatív?** Mert egy enyhén ívelt "egyenes" út (pl. 15°-os ív) abszolút szög alapján
"kanyarnak" tűnne. De ha a kereszteződés másik ága 90°-ra van, akkor a 15°-os ág
**relatíve** egyenes. Ez a relatív logika sokkal robusztusabb.

### Az "intersection memory" (275–282. sor)

```python
if self._previous_decision != RoadOption.VOID and \
   self._intersection_end_node > 0 and \
   self._intersection_end_node != previous_node and \
   next_edge['type'] == RoadOption.LANEFOLLOW and \
   next_edge['intersection']:
    decision = self._previous_decision       # ← megismétli az előző döntést
```

Egy nagy kereszteződés több apró élből állhat. Ha már eldöntöttük, hogy `LEFT`, akkor a
kereszteződés összes további élén is `LEFT` maradjon — ne kezdjen újra dönteni menet közben.

## `trace_route(origin, destination)` (356–404. sor)

Ez fűzi össze az egészet: az A\* által talált csomópont-sorozatból előállítja a
**(waypoint, RoadOption) párok listáját**.

```python
for i in range(len(route) - 1):
    road_option = self._turn_decision(i, route)
    edge = self._graph.edges[route[i], route[i+1]]

    if edge['type'] != LANEFOLLOW and edge['type'] != VOID:
        # sávváltó él: csak 2 pontot ad hozzá (nincs végigkövetés)
        ...
    else:
        path = [edge['entry_waypoint']] + edge['path'] + [edge['exit_waypoint']]
        closest_index = self._find_closest_in_list(current_waypoint, path)
        for waypoint in path[closest_index:]:
            current_waypoint = waypoint
            route_trace.append((current_waypoint, road_option))
            # ...megállási feltételek a cél közelében...
```

A `_find_closest_in_list` azért kell, hogy ne "ugorjon vissza" — az aktuális pozíciótól
folytassa, ne a szakasz elejétől.

---

# 4. `local_planner.py` — mi ebből használt? (293 sor)

## `RoadOption` (20–31. sor) ✅ EZ HASZNÁLT

```python
class RoadOption(Enum):
    LANEFOLLOW = 0
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    VOID = -1

    def __eq__(self, other):
        return self.value == other.value
```

Ez az enum kerül az ágens állapotába `maneuver` néven (`Discrete(4)`).

> **A felüldefiniált `__eq__`:** azért van, hogy különböző modulokból importált
> `RoadOption` példányok is egyenlőnek számítsanak (a Python enum egyébként identitás
> alapján hasonlít). Mellékhatás: az `__eq__` felüldefiniálása **elrontja a hash-elhetőséget**
> (Pythonban ha `__eq__`-t definiálsz, a `__hash__` `None` lesz, hacsak nem adod meg).
> Ezért nem lehetne `RoadOption`-t szótárkulcsként vagy halmazelemként használni.
> A jelenlegi kód nem teszi, tehát nem baj.

## `LocalPlanner` osztály (34–244. sor) ❌ NEM HASZNÁLT

Ez egy **klasszikus, nem-RL vezérlő**: PID szabályozókkal követi a waypointokat.

```python
def run_step(self, debug=True):
    ...
    self.target_waypoint, self._target_road_option = self._waypoint_buffer[0]
    control = self._vehicle_controller.run_step(self._target_speed, self.target_waypoint)
```

**Miért érdekes, ha nem használt?** Mert ez a **baseline**: ha az RL ágensedet össze
akarod hasonlítani egy hagyományos megoldással, ez a viszonyítási alap. Ugyanazt a feladatot
oldja meg PID-del, tanulás nélkül.

**A `controller.py` PID-jei:**
```python
args_lateral_dict      = {'K_P': 1.95, 'K_D': 0.01, 'K_I': 1.4,  'dt': 0.05}   # kormányzás
args_longitudinal_dict = {'K_P': 1.0,  'K_D': 0,    'K_I': 1,    'dt': 0.05}   # sebesség
```

A PID hibajele oldalirányban a **cél-waypoint és az autó orientációja közti szög**,
hosszirányban a **sebességkülönbség**.

---

# 5. `carla_env/tools/misc.py` — segédfüggvények (108 sor)

| Függvény | Mit csinál | Használja |
|----------|-----------|-----------|
| `draw_waypoints(world, waypoints, z)` | nyilakat rajzol a szimulátorba | `LocalPlanner` (nem fut) |
| `get_speed(vehicle)` | m/s → km/h | `controller.py` |
| `is_within_distance_ahead(...)` | célpont előttünk van-e adott távolságon belül | `agent.py` (nem fut) |
| `compute_magnitude_angle(...)` | távolság + szög két pont között | `agent.py` (nem fut) |
| `distance_vehicle(waypoint, transform)` | 2D távolság | `LocalPlanner` |
| **`vector(location_1, location_2)`** | **egységvektor két pont között** | `planner.py`, `global_route_planner.py` |

> ⚠ **Emlékeztető a névütközésre:** ez a `vector(l1, l2)` **két** argumentumot vár és
> egységvektort ad. A `wrappers.py`-ban lévő `vector(v)` **egyet** vár és nyers koordinátákat
> ad. Két teljesen különböző függvény azonos néven!

---

# 6. `carla_env/tools/hud.py` — a képernyő-kijelzés (218 sor)

Nem befolyásolja a tanulást, de hasznos hibakereséshez.

- `HUD.on_world_tick(timestamp)` — a szerver oldali FPS és szimulációs idő mérése
- `HUD.tick(world, clock)` — összeállítja a bal oldali infopanelt (sebesség, gyorsulás,
  pozíció, gáz/kormány/fék állás)
- `HUD.render(display, extra_info)` — kirajzolja; az `extra_info` a `CarlaRouteEnv.render()`
  által feltöltött lista (epizódszám, jutalom, manőver...)
- `FadingText` — ideiglenes értesítések (ütközés, sávelhagyás), amik elhalványulnak
- `HelpText` — a fájl elején lévő docstring megjelenítése H billentyűre

---

➡️ Következő: [06-fogalomtar.md](06-fogalomtar.md)
