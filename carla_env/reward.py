import carla
import numpy as np

# --- Beallitasok (itt szerkesztheto minden) --------------------------------
EARLY_STOP = True             # korai epizod-vege engedelyezve?
MAX_SPEED = 60.0              # ezen felul terminal [km/h]
TARGET_SPEED = 50.0           # ideal sebesseg [km/h]
SPEED_WEIGHT = 1.9            # a sebesseg sulya (1.0 = nincs sulyozas)
MAX_DISTANCE = 2.0            # max eltres a sav kozeptol arra, amerre nincs azonos iranyu sav [m]

# Savelhagyas (csak reward_fn). Terminal csak a lathato szabalysertes: ezeken a
# felfestes-tipusokon atlepni (a vegyes Solid/Broken-t nem, mert nem tudjuk,
# melyik oldala szaggatott). Szaggatotton at engedely (elozes) nelkul: nem
# terminal, csak OFF_LANE_PENALTY / lepes, amig vissza nem er.
SOLID_MARKS = ("Solid", "SolidSolid", "Curb", "Grass")
OFF_LANE_PENALTY = -0.1       # / lepes, mint a kullogas (LAG_PENALTY)
MAX_STD_CENTER_LANE = 0.35    # max szoras a kozeptol valo tavolsagban [m]
MAX_ANGLE_CENTER_LANE = 90    # max szogeltres [fok]
PENALTY_REWARD = -10          # terminal buntetes
STOP_TIMEOUT = 5.0            # ennyi mp allas utan terminal [s]


FOLLOW_GAP_S = 2.0            # kovetesi tavolsag, ha nem elozhet [s]
OVERTAKE_START_M = 1.0        # ennyivel a route-tol a cel sav fele: kiallt elozni [m]
OVERTAKE_MIN = 0.3            # elozes kozben a minimalis jutalom / lepes...
OVERTAKE_SPEED = 0.6          # ...plusz ennyi * speed_factor ** SPEED_WEIGHT
                              # (25 km/h: 0.51 > kovetes 0.35; 50 km/h: 0.9 < sajat sav 1.0)
OVERTAKE_BONUS = 5.0          # egyszeri jutalom kesz elozesert
PASS_BEHIND_M = 5.0           # a megelozott auto ennyivel mogem kerult [m]
BACK_IN_LANE_M = 0.5          # a route-tol ez alatt: visszaert a savjaba [m]
COLLISION_PENALTY = -30       # jarmuvel utkozes (terminal)

# Elozes kozben ne erje meg a savhataron menni, es kint maradni se:
LANE_LINE_FACTOR = 0.7        # a ket sav hataran ennyiszeres a jutalom (savkozepen 1.0)
RETURN_GRACE_S = 3.0          # a megelozott mogem kerulese utan savonkent ennyi ideig nincs
                              # csokkentes [s] (a visszasorolas maga ~3 s/sav - azt ne buntesse)
OVERTAKE_MAX_S = 15.0         # ha eddig nem kerult mogem, akkor is csokken [s]
LINGER_DECAY_S = 4.0          # ennyi ido alatt csokken a jutalom...
LINGER_MIN = 0.3              # ...ennyiszeresere (0.9 -> 0.27, kevesebb, mint a kovetes)

# Kovetes, ha nem elozhet (Blocked): ne legyen jobb a beragadas, mint az elozes.
# Az elozessel nem versenyez: ha elozhetne, az mar kullogas (LAG_PENALTY).
BLOCKED_MAX = 0.8             # (elozes 50 km/h-val: 0.9, szabad sav: 1.0)
# A kovetesi ido (headway) jutalma lognormal (Zhu et al. 2020, TR-C 117:
# NGSIM-re illesztve sigma=0.4365, csucs 1.26 s-nal), de a csucs FOLLOW_GAP_S-en
# (fek nincs), 1-re normalva: 1 s 0.28, 1.5 s 0.8, 2 s 1.0, 3 s 0.65, 4 s 0.28 -
# a tul nagy lemaradas is kevesebbet er.
HEADWAY_SIGMA = 0.4365
# Ha kozeledik a leadhez: TTC_WEIGHT * log(TTC / TTC_MAX_S) (Zhu et al. 2020),
# 3 s: -0.26, 1 s: -0.58. Csak Blocked-ban - az elozes elotti felzarkozast nem.
TTC_MAX_S = 7.0
TTC_WEIGHT = 0.3

# Kullogas: elozhetne (van hova kiallni), megis kozel a lead mogott megy, es
# nem gyorsabb nala. Ilyenkor kovetesi jutalom helyett buntetes jar.
LAG_DIST_M = 25.0             # ennel kozelebb a lead...
LAG_MARGIN_KMH = 5.0          # ...es nem gyorsabb nala ennyivel: kullog
LAG_PENALTY = -0.1            # / lepes. gamma=0.98 mellett a vegtelen kullogas -0.1/0.02 = -5,
                              # ez NE legyen rosszabb, mint a PENALTY_REWARD - kulonben inkabb
                              # lesodrodik, minthogy kullogjon

# Mennyi ideje all EGYHUZAMBAN az auto [s]. Elindulaskor es minden epizod
# elejen nullazodik.
_low_speed_timer = 0.0

# reward_fn allapota, epizodonkent nullazodik:
_overtaking = False       # kiallt elozni, es meg nem ert vissza a route savjaba
_to_lane = 0              # hova allt ki (savban, a route-hoz): +1/+2 balra, -1/-2 jobbra
_target_id = None         # akit eloz
_passed = False           # a _target_id mar PASS_BEHIND_M-rel mogotte van
_return_ok = False        # a route fele szaggatott a vonal (szabad visszasorolni)
_overtaken_ids = set()    # akiert mar jart bonusz
_smooth_hold = 0          # visszasorolas utan ennyi lepesig nincs cikkcakk-buntetes
_overtake_t = 0.0         # miota eloz [s]
_passed_t = None          # az _overtake_t, amikor a _target_id mogem kerult
_lane_idx = 0             # melyik savban van az ego kozepe (a route savhoz, + balra)


def reward_fn_og(env):
    global _low_speed_timer

    terminal_r1eason = "Running..."
    speed_kmh = env.vehicle.get_speed()
    if env.step_count == 0:
        _low_speed_timer = 0.0

    # --- 1) Epizod-vege feltetelek -----------------------------------------
    if EARLY_STOP and not env.terminal_state:
        _low_speed_timer = _low_speed_timer + 1.0 / env.fps if speed_kmh < 1.0 else 0.0

        if _low_speed_timer > STOP_TIMEOUT:
            env.terminal_state = True
            terminal_reason = "Vehicle stopped"

        elif env.distance_from_center > MAX_DISTANCE:
            env.terminal_state = True
            terminal_reason = "Off-track"
        elif MAX_SPEED > 0 and speed_kmh > MAX_SPEED:
            env.terminal_state = True
            terminal_reason = "Too fast"

    if env.terminal_state:
        _low_speed_timer = 0.0
        if terminal_reason != "Running...":
            print(f"{env.episode_idx}| Terminal: {terminal_reason}")
        if env.success_state:
            print(f"{env.episode_idx}| Success")
        env.extra_info.extend([terminal_reason, ""])
        return PENALTY_REWARD

    surr = env.get_vehicle_surroundings()

    if speed_kmh <= TARGET_SPEED:
        speed_factor = speed_kmh / TARGET_SPEED
    else:
        speed_factor = 1.0 - (speed_kmh - TARGET_SPEED) / (MAX_SPEED - TARGET_SPEED)
    speed_factor = float(np.clip(speed_factor, 0.0, 1.0))

    centering_factor = max(1.0 - env.distance_from_center / MAX_DISTANCE, 0.0)


    next_angle = env.vehicle.get_angle(env.current_waypoint)
    next_angle_factor = max(1.0 - abs(next_angle) / np.deg2rad(MAX_ANGLE_CENTER_LANE), 0.0)

    std = np.std(env.distance_from_center_history)
    smoothness_factor = max(1.0 - abs(std) / MAX_STD_CENTER_LANE, 0.0)

    env.extra_info.extend([terminal_reason, ""])


    return (speed_factor ** SPEED_WEIGHT) * centering_factor * next_angle_factor * smoothness_factor


def reward_fn(env):
    """Sav-kozep tartas + elozes. A forgalmat az env.surroundings adja
    (get_vehicle_surroundings, lepesenkent egyszer).

      - Savelhagyas: folytonos vonal atlepese terminal (Solid line). Arra,
        amerre nincs azonos iranyu sav, 2 m-en tul Off-track. Szaggatotton at
        engedely nelkul (nem elozes) nem terminal, csak OFF_LANE_PENALTY /
        lepes, amig vissza nem er. Csak az atlepes pillanata szamit: aki
        szaggatotton kiallt, azt a kesobb folytonosra valto vonal nem oli meg.
      - Nincs elottem senki: mint a reward_fn_og.
      - Elottem van valaki, es van hova kiallni (lasd fent a sorrendet):
        kiallhat, a cel savban elozesi jutalmat kap. Elozes kozben
        OVERTAKE_MIN + OVERTAKE_SPEED * sebesseg - kevesebb, mint a sajat
        savban, de tobb, mint lassan kovetni. A savhataron LANE_LINE_FACTOR-
        szoros, a megelozes utan (vagy OVERTAKE_MAX_S utan) pedig csokken,
        amig vissza nem sorol.
      - Megelozte (PASS_BEHIND_M-rel mogotte van), es szaggatott vonalon
        sorolt vissza a route savjaba: +OVERTAKE_BONUS.
      - Elottem van valaki, de nem elozhet: a FOLLOW_GAP_S kovetesi tavolsagot
        kell tartani (fek nincs, gazelvetellel), a sebesseget a lead autohoz
        merjuk, legfeljebb BLOCKED_MAX.
    """
    global _low_speed_timer, _overtaking, _to_lane, _target_id, _passed, _return_ok
    global _overtaken_ids, _smooth_hold, _overtake_t, _passed_t, _lane_idx

    terminal_reason = "Running..."
    speed_kmh = env.vehicle.get_speed()
    surr = env.surroundings
    if env.step_count == 0:
        _low_speed_timer = 0.0
        _overtaking, _to_lane, _target_id, _passed, _return_ok = False, 0, None, False, False
        _overtaken_ids = set()
        _smooth_hold = 0
        _overtake_t, _passed_t = 0.0, None
        _lane_idx = 0

    # Elojeles tavolsag a route-tol [m]: + = balra, - = jobbra. A nagysaga a
    # distance_from_center, az elojele a route waypoint jobb vektorabol.
    wp = env.current_waypoint.transform
    loc = env.vehicle.get_transform().location
    r = wp.get_right_vector()
    side = (loc.x - wp.location.x) * r.x + (loc.y - wp.location.y) * r.y
    offset = env.distance_from_center if side < 0.0 else -env.distance_from_center

    lane_w = surr["lane_width"]

    # A route sav melletti azonos iranyu savok, oldalankent max 2: sav (+1/+2
    # balra, -1/-2 jobbra) -> a felfestes tipusa kozte es a route fele eso
    # szomszedja kozott. Ami nincs (vagy szembesav), az kimarad.
    route_wp = env.current_waypoint
    marks = {}
    for s in (1, -1):
        w = route_wp
        for n in (1, 2):
            mark = w.left_lane_marking if s > 0 else w.right_lane_marking
            w = w.get_left_lane() if s > 0 else w.get_right_lane()
            if w is None or w.lane_type != carla.LaneType.Driving or w.lane_id * route_wp.lane_id <= 0:
                break
            marks[s * n] = str(mark.type)

    # Melyik savban van most az ego kozepe. Ha savot valtott, a kulso sav
    # felfestesen lepett at (0 -> +1 es +1 -> 0 is a +1-es) - ha az folytonos,
    # terminal. Nem letezo savnal nincs felfestes: ott a 2 m-es hatar dont.
    lane_idx = int(round(offset / lane_w))
    solid_crossed = None
    if lane_idx != _lane_idx:
        outer = lane_idx if abs(lane_idx) > abs(_lane_idx) else _lane_idx
        if marks.get(outer) in SOLID_MARKS:
            solid_crossed = marks[outer]
        _lane_idx = lane_idx

    has_lead = surr["front_id"] is not None
    # Hova allhat ki most (a sajat savomhoz kepest): az elso a sorrendben,
    # ahol odaig szaggatott, a cel sav szabad, es 2 savnal a kozbulson at
    # lehet vagni. Jobbra csak akkor, ha balra nem lehet. None: sehova.
    to_lane = None
    if has_lead:
        lanes = surr["lanes"]
        for n in (1, 2, -1, -2):
            lane = lanes.get(n)
            if lane and lane["broken"] and lane["free"] and (abs(n) == 1 or lanes[n // 2]["cross"]):
                to_lane = n
                break
    can_overtake = to_lane is not None
    blocked = has_lead and not can_overtake

    # --- 1) Elozes allapot -------------------------------------------------
    # Kiallt: a cel sav fele hagyja el a savot, mikozben elozhet. Innentol
    # addig tart, amig vissza nem er a route savjaba (a sav elhagyasa kozben
    # az env mar a masik savot latja sajatjanak, ezert kell megjegyezni).
    # Csak a route savbol (lane_idx == 0): a szomszed savbol a to_lane mar
    # ahhoz a savhoz kepest lenne, rossz savszammal.
    if (not _overtaking and can_overtake and lane_idx == 0
            and offset * to_lane > 0.0 and abs(offset) > OVERTAKE_START_M):
        _overtaking, _to_lane, _target_id, _passed, _return_ok = \
            True, to_lane, surr["front_id"], False, False
        _overtake_t, _passed_t = 0.0, None
    if _overtaking:
        _overtake_t += 1.0 / env.fps
        lon = surr["lon"].get(_target_id)
        if not _passed and lon is not None and lon < -PASS_BEHIND_M:
            _passed, _passed_t = True, _overtake_t
        # A route savon kivul (kozeppel a savhataron tul) a route feloli
        # felfestes az, amin vissza kell sorolni. 2 savnal a route melletti
        # savbol nezett marad meg, mert az utolso.
        if abs(offset) > lane_w / 2.0:
            _return_ok = surr["right_marking" if _to_lane > 0 else "left_marking"] == "Broken"

    # --- 2) Epizod-vege feltetelek -----------------------------------------
    if EARLY_STOP and not env.terminal_state:
        _low_speed_timer = _low_speed_timer + 1.0 / env.fps if speed_kmh < 1.0 else 0.0

        if _low_speed_timer > STOP_TIMEOUT:
            env.terminal_state = True
            terminal_reason = "Vehicle stopped"

        elif solid_crossed is not None:
            env.terminal_state = True
            terminal_reason = f"Solid line ({solid_crossed})"
        # A hatar arra, amerre azonos iranyu sav van, annyi savval tagabb
        # (hogy melyikbe szabad atmenni, azt a folytonos vonal donti el, fent).
        elif not (-MAX_DISTANCE - lane_w * sum(1 for k in marks if k < 0)
                  <= offset <= MAX_DISTANCE + lane_w * sum(1 for k in marks if k > 0)):
            env.terminal_state = True
            terminal_reason = "Off-track"
        elif MAX_SPEED > 0 and speed_kmh > MAX_SPEED:
            env.terminal_state = True
            terminal_reason = "Too fast"

    if env.terminal_state:
        _low_speed_timer = 0.0
        # Az utkozest a collision szenzor jelzi (env._on_collision), nem ez a fuggveny.
        if env.collision_with is not None and terminal_reason == "Running...":
            terminal_reason = f"Collision ({env.collision_with})"
        env.terminal_reason = terminal_reason  # a TensorboardCallback logolja
        if terminal_reason != "Running...":
            print(f"{env.episode_idx}| Terminal: {terminal_reason}")
        if env.success_state:
            print(f"{env.episode_idx}| Success")
        env.extra_info.extend([terminal_reason, ""])
        if env.collision_with is not None and env.collision_with.startswith("vehicle."):
            return COLLISION_PENALTY
        return PENALTY_REWARD

    # --- 3) Jutalom --------------------------------------------------------
    if speed_kmh <= TARGET_SPEED:
        speed_factor = speed_kmh / TARGET_SPEED
    else:
        speed_factor = 1.0 - (speed_kmh - TARGET_SPEED) / (MAX_SPEED - TARGET_SPEED)
    speed_factor = float(np.clip(speed_factor, 0.0, 1.0))

    # Az elozes kozbeni jutalom. Ezt kapja kint, es a kiallas atmenete is
    # ebbe megy at - ezert egyszer szamoljuk, igy 1 m-nel nincs ugras.
    # A legkozelebbi savkozeptol (a route sav es a cel sav kozott barmelyik)
    # a savhatarig LANE_LINE_FACTOR-ig csokken: a savvaltas at tud menni
    # rajta, de a hataron menni nem eri meg. Korabban vegig 0.9 volt.
    goal = _to_lane if _overtaking else (to_lane or 0)
    step = 1 if goal >= 0 else -1
    lane_d = min(abs(offset - n * lane_w) for n in range(0, goal + step, step))
    lane_factor = 1.0 - (1.0 - LANE_LINE_FACTOR) * min(lane_d / (lane_w / 2.0), 1.0)
    # Kint maradas: a megelozott mogem kerulese utan savonkent RETURN_GRACE_S
    # mulva, vagy OVERTAKE_MAX_S utan mindenkepp csokken, LINGER_MIN-ig.
    # Korabban a kint maradas 0.9-et, a visszasorolas 1.0-t ert.
    linger = 0.0
    if _overtaking:
        linger = _overtake_t - OVERTAKE_MAX_S
        if _passed:
            linger = max(linger, _overtake_t - _passed_t - RETURN_GRACE_S * abs(_to_lane))
    linger_factor = max(1.0 - max(linger, 0.0) / LINGER_DECAY_S, LINGER_MIN)
    overtake_term = (OVERTAKE_MIN + OVERTAKE_SPEED * speed_factor ** SPEED_WEIGHT) \
        * lane_factor * linger_factor

    # Szabad savban van-e: a route savban, vagy elozes kozben a route es a cel
    # sav kozott. Mashol (szaggatotton at, engedely nelkul) nem terminal - azt
    # a policy nem mindig latja -, csak buntetes, amig vissza nem jon.
    # Korabban ez 2 m-en Off-track volt.
    allowed_lane = lane_idx == 0 or (_overtaking and lane_idx * _to_lane > 0
                                     and abs(lane_idx) <= abs(_to_lane))

    if not allowed_lane:
        state = "Off lane %+d" % lane_idx
        reward = OFF_LANE_PENALTY
    elif _overtaking:
        state = "Overtaking %s%d" % ("L" if _to_lane > 0 else "R", abs(_to_lane))
        reward = overtake_term
        # Visszaert a route savjaba: vege. Bonusz, ha megelozte, es szaggatott
        # vonalon jott vissza - autonkent egyszer.
        if abs(offset) < BACK_IN_LANE_M:
            _overtaking = False
            _smooth_hold = env.distance_from_center_history.maxlen
            if _passed and _return_ok and _target_id not in _overtaken_ids:
                _overtaken_ids.add(_target_id)
                env.overtakes += 1
                reward += OVERTAKE_BONUS
        env.overtake_reward += reward
    else:
        centering_factor = max(1.0 - env.distance_from_center / MAX_DISTANCE, 0.0)

        next_angle = env.vehicle.get_angle(env.current_waypoint)
        next_angle_factor = max(1.0 - abs(next_angle) / np.deg2rad(MAX_ANGLE_CENTER_LANE), 0.0)

        # A visszasorolas utan a kozeptavolsag szorasa meg a savvaltast latja,
        # ezert egy ablaknyi ideig nem buntetjuk.
        if _smooth_hold > 0:
            _smooth_hold -= 1
            smoothness_factor = 1.0
        else:
            std = np.std(env.distance_from_center_history)
            smoothness_factor = max(1.0 - abs(std) / MAX_STD_CENTER_LANE, 0.0)

        lagging = False
        ttc_term = 0.0
        if blocked:
            # Nem elozhet: a FOLLOW_GAP_S kovetesi idot kell tartani - kozelebb
            # es messzebb is kevesebb (lognormal, lasd fent). A sebesseget a
            # lead autohoz merjuk, nem a TARGET_SPEED-hez, igy csigazva nem eri
            # meg kovetni. A teteje BLOCKED_MAX: korabban ~1.0 volt, tobb, mint
            # az elozes - a beragadas volt a legjobb allapot.
            state = "Blocked"
            # A lognormal csucsa exp(mu - sigma^2): ide tolva FOLLOW_GAP_S-re,
            # es a csucsbeli ertekkel osztva, hogy ott 1 legyen. Allo egonal a
            # kovetesi ido ertelmetlen - 1 m/s-mal szamolunk (a match ugyis ~0).
            headway = surr["front_dist"] / max(speed_kmh / 3.6, 1.0)
            mu = np.log(FOLLOW_GAP_S) + HEADWAY_SIGMA ** 2
            gap_term = FOLLOW_GAP_S / headway * np.exp(
                HEADWAY_SIGMA ** 2 / 2 - (np.log(headway) - mu) ** 2 / (2 * HEADWAY_SIGMA ** 2))
            match = min((speed_kmh + 1.0) / (surr["front_speed"] + 1.0), 1.0)
            speed_term = BLOCKED_MAX * gap_term * match
            # Kozeledik: TTC buntetes, mar TTC_MAX_S-mal a rafutas elott jelez
            # (fek nincs, gazelvetellel idoben kell lassitani). 0.1 s alatt
            # nem szamolunk tovabb, kulonben log(0).
            closing = (speed_kmh - surr["front_speed"]) / 3.6
            ttc = surr["front_dist"] / closing if closing > 0.0 else np.inf
            if ttc < TTC_MAX_S:
                ttc_term = TTC_WEIGHT * np.log(max(ttc, 0.1) / TTC_MAX_S)
            env.blocked_time += 1.0 / env.fps
        elif can_overtake:
            # Elozhetne. Ha megis kozel a lead mogott kullog, buntetes: korabban
            # itt is jart a sebesseg szerinti jutalom (25 km/h-val +0.35), es a
            # mogotte maradas biztos, pozitiv lokalis optimum volt.
            lagging = (surr["front_dist"] < LAG_DIST_M
                       and speed_kmh < surr["front_speed"] + LAG_MARGIN_KMH)
            state = "%s %s%d" % ("Lagging" if lagging else "Can overtake",
                                 "L" if to_lane > 0 else "R", abs(to_lane))
            speed_term = LAG_PENALTY if lagging else speed_factor ** SPEED_WEIGHT
        else:
            state = "Free"
            speed_term = speed_factor ** SPEED_WEIGHT
        reward = speed_term * centering_factor * next_angle_factor * smoothness_factor
        if lagging:
            # A buntetest nem szorozzuk a sav-faktorokkal - kulonben a sav
            # szelen, cikkcakkozva kisebb lenne.
            reward = speed_term
        reward += ttc_term   # ugyanigy: nem szorozzuk (csak Blocked-ban nem 0)

        # Elkezdett kiallni (elozhet, es a cel sav fele tart): 0 m-en a
        # kovetes, 1 m-en mar a teljes elozesi jutalom jar, kozte linearisan -
        # nincs "volgy". Korabban itt a centering es a smoothness (a kiallast
        # cikkcakknak latta) 1 m-ig 0.35-rol 0.004-re vitte le a jutalmat.
        if can_overtake and offset * to_lane > 0.0:
            k = min(abs(offset) / OVERTAKE_START_M, 1.0)
            reward = (1 - k) * speed_term * next_angle_factor + k * overtake_term

    env.extra_info.extend([
        terminal_reason,
        "State:  % 19s" % state,
        "Overtakes:           % 7d" % env.overtakes,
        ""])
    return float(reward)
