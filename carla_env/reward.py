import numpy as np


# Mennyi ideje all EGYHUZAMBAN az auto [s]. Elindulaskor es minden epizod
# elejen nullazodik.
_low_speed_timer = 0.0

# reward_fn allapota, epizodonkent nullazodik:
_overtaking = False       # kiallt elozni, es meg nem ert vissza a route savjaba
_to_lane = 0              # hova allt ki (savban, a route-hoz): +1/+2 balra, -1/-2 jobbra
_target_id = None         # akit eloz
_passed = False           # a _target_id mar 5 m-rel mogotte van
_return_ok = False        # a route fele szaggatott a vonal (szabad visszasorolni)
_overtaken_ids = set()    # akiert mar jart bonusz
_smooth_hold = 0          # visszasorolas utan ennyi lepesig nincs cikkcakk-buntetes
_overtake_t = 0.0         # miota eloz [s]
_passed_t = None          # az _overtake_t, amikor a _target_id mogem kerult
_prev_steer = 0.0         # az elozo lepes kiadott kormanya
_lag_t = 0.0              # miota kullog egyhuzamban [s]
_prev_phi = 0.0           # reward_fn_5: az elozo lepes potencialja, Phi(s)
_lon0 = 0.0               # reward_fn_5: a target tavolsaga a kiallaskor [m]


def reward_fn_og(env):
    # =========================================================================
    # REWARD OG - az eredeti savtarto jutalom, forgalom nelkul.
    #
    # A kiindulo repo (alberto-mate/CARLA-SB3-RL-Training-Environment)
    # reward_fn5-je alapjan: csak savtartas es sebesseg, az elozest nem
    # ismeri. A lepesjutalom negy 0..1 kozotti tenyezo szorzata:
    #   sebesseg^1.9 * sav kozepen tartas * a route iranyaba nezes * simasag.
    # Ha barmelyik 0, a jutalom is 0, igy egyszerre kell mindet teljesiteni.
    # A ^1.9 miatt a lassu haladas keveset er (25 km/h: 0.27, 50 km/h: 1.0).
    #
    # Terminal (megallt, lement, tul gyors, utkozes): mind -10. Az Off-track
    # a route savjanak kozepetol mert 2 m, tehat savot valtani (elozni) sem
    # lehet.
    # Az env.terminal_reason-t nem allitja, ezert vele a TensorBoard
    # terminal/* metrikai nem mukodnek (minden epizod route_done-nak latszik).
    # =========================================================================
    global _low_speed_timer

    terminal_reason = "Running..."
    speed_kmh = env.vehicle.get_speed()
    if env.step_count == 0:
        _low_speed_timer = 0.0

    # --- 1) Epizod-vege feltetelek -----------------------------------------
    # Megallt (5 s-ig 1 km/h alatt), 2 m-nel messzebb van a route savjanak
    # kozepetol, vagy 60 km/h felett megy. Az utkozest a collision szenzor
    # jelzi (o allitja a terminal_state-et), az is -10.
    if not env.terminal_state:
        _low_speed_timer = _low_speed_timer + 1.0 / env.fps if speed_kmh < 1.0 else 0.0

        if _low_speed_timer > 5.0:
            env.terminal_state = True
            terminal_reason = "Vehicle stopped"

        elif env.distance_from_center > 2.0:
            env.terminal_state = True
            terminal_reason = "Off-track"
        elif speed_kmh > 60.0:
            env.terminal_state = True
            terminal_reason = "Too fast"

    if env.terminal_state:
        _low_speed_timer = 0.0
        if terminal_reason != "Running...":
            print(f"{env.episode_idx}| Terminal: {terminal_reason}")
        if env.success_state:
            print(f"{env.episode_idx}| Success")
        env.extra_info.extend([terminal_reason, ""])
        return -10

    # Lekerdezi a forgalmat, de nem hasznalja semmire.
    surr = env.get_vehicle_surroundings()

    # --- 2) Jutalom --------------------------------------------------------
    # speed_factor: 0..1, 50 km/h-nal 1. Alatta linearis, 50-60 km/h kozott
    # 0-ra esik (60 felett "Too fast" terminal).
    if speed_kmh <= 50.0:
        speed_factor = speed_kmh / 50.0
    else:
        speed_factor = 1.0 - (speed_kmh - 50.0) / 10.0
    speed_factor = float(np.clip(speed_factor, 0.0, 1.0))

    # centering_factor: a route savjanak kozepen 1, 2 m-re tole 0.
    centering_factor = max(1.0 - env.distance_from_center / 2.0, 0.0)


    # next_angle_factor: a route iranyaba nez 1, 90 fokra tole 0.
    next_angle = env.vehicle.get_angle(env.current_waypoint)
    next_angle_factor = max(1.0 - abs(next_angle) / np.deg2rad(90), 0.0)

    # smoothness_factor: a kozeptavolsag szorasa az utobbi 30 lepesben
    # (cikkcakk), 0 m-nel 1, 0.35 m-nel 0.
    std = np.std(env.distance_from_center_history)
    smoothness_factor = max(1.0 - abs(std) / 0.35, 0.0)

    env.extra_info.extend([terminal_reason, ""])


    return (speed_factor ** 1.9) * centering_factor * next_angle_factor * smoothness_factor


def reward_fn_1(env):
    # =========================================================================
    # REWARD 1 - savtartas + szabalyos elozes, az og kiegeszitese forgalomra.
    #
    # Az og-hez kepest uj:
    #   - elozesi allapotgep: csak balra (egy vagy ket savot), szaggatott
    #     vonalon at, szabad savba lehet kiallni, ha a sajat savomban auto
    #     van elottem. Szabalyos elozes (elhaladt a target mellett, es
    #     szaggatott vonalon jott vissza): +5 bonusz, autonkent egyszer,
    #   - Off-track a legkozelebbi BARMELYIK sav kozepetol 2 m-re (az og-ben a
    #     route savjatol, ott a kiallas is Off-track lett volna),
    #   - jarmuvel utkozes -30, a tobbi terminal marad -10,
    #   - "Blocked" (van elottem auto, de nem elozhetek): sebesseg-buntetes
    #     helyett a kovetesi tavolsagot jutalmazza (gap_term, 2 s headway-nel
    #     a legtobb) + TTC-buntetes,
    #   - "Lagging" (elozhetne, de 25 m-en belul kullog a lead mogott):
    #     negativ jutalom, ami az idovel no,
    #   - kihuzodasi rampa: a kiallas elso 1 m-en a jutalom linearisan
    #     atusztatodik a kovetesbol az elozesi jutalomba,
    #   - rossz savban -0.5, kormany-rangatas buntetes.
    #
    # Lepesjutalmak (sav kozepen, egyenesen, kormanybuntetes nelkul):
    #   szabad sav, 50 km/h                                   1.0
    #   elozes 50 km/h-val a bal sav kozepen                  0.9
    #   elozes 25 km/h-val a bal sav kozepen                  0.46
    #   kovetes (Blocked) 2 s headway-jel, a lead sebessegevel 0.8
    #   kulloges (Lagging) 25 m-en belul                      -0.10..-0.18
    #   rossz sav                                             -0.5
    #
    # Eredmeny (SAC_reward_1, az utolso 20% epizod atlaga): 1.6 elozes/epizod,
    # 715 m, 36.6 km/h, az epizodok 77%-a jarmu-utkozessel ert veget.
    # =========================================================================
    global _low_speed_timer, _overtaking, _to_lane, _target_id, _passed, _return_ok
    global _overtaken_ids, _smooth_hold, _overtake_t, _passed_t, _prev_steer, _lag_t

    terminal_reason = "Running..."
    speed_kmh = env.vehicle.get_speed()
    # A forgalom az egohoz kepest. A step() lepesenkent egyszer szamolja
    # (get_vehicle_surroundings): lead auto, szomszed savok, vonalak, lon/auto.
    surr = env.surroundings
    # Epizod eleje: az elozesi allapotgep minden valtozoja nullazodik.
    if env.step_count == 0:
        _low_speed_timer = 0.0
        _overtaking, _to_lane, _target_id, _passed, _return_ok = False, 0, None, False, False
        _overtaken_ids = set()
        _smooth_hold = 0
        _overtake_t, _passed_t = 0.0, None
        _prev_steer = env.vehicle.control.steer
        _lag_t = 0.0

    # --- 1) Hol vagyok a route savjahoz kepest ------------------------------
    # offset: elojeles tavolsag a route savjanak kozepetol [m], + balra,
    # - jobbra. A distance_from_center elojel nelkuli, az oldalt a waypoint
    # jobb vektorara vetitett helyzet adja meg.
    wp = env.current_waypoint.transform
    loc = env.vehicle.get_transform().location
    r = wp.get_right_vector()
    side = (loc.x - wp.location.x) * r.x + (loc.y - wp.location.y) * r.y
    offset = env.distance_from_center if side < 0.0 else -env.distance_from_center

    lane_w = surr["lane_width"]

    # Hanyadik savban vagyok a route savjahoz kepest: 0 = a sajat, +1 = eggyel
    # balra, -1 = eggyel jobbra.
    lane_idx = int(round(offset / lane_w))

    # --- 2) Elozhetek-e most ------------------------------------------------
    # has_lead: van auto a sajat savomban elottem, 40 m-en belul.
    # to_lane: hova allhatok ki elozni. Csak balra: +1, vagy ha az foglalt, +2.
    # Feltetel: minden atlepett vonal szaggatott ("broken"), a celsav szabad
    # (-8..+20 m-en nincs benne auto), +2-nel a +1-es savon at lehet vagni
    # ("cross": mellettem +-8 m-en nincs benne senki).
    has_lead = surr["front_id"] is not None
    to_lane = None
    if has_lead:
        lanes = surr["lanes"]
        for n in (1, 2):
            lane = lanes.get(n)
            if lane and lane["broken"] and lane["free"] and (abs(n) == 1 or lanes[n // 2]["cross"]):
                to_lane = n
                break
    can_overtake = to_lane is not None
    # blocked: van elottem auto, de nem elozhetem meg, kovetni kell.
    blocked = has_lead and not can_overtake

    # --- 3) Elozesi allapotgep ----------------------------------------------
    # Kiallas: elozhetek, meg a sajat savomban vagyok, es mar 1 m-nel tobbet
    # mozdultam a celsav fele. Ekkor rogzul a celsav (_to_lane) es a
    # megelozendo auto (_target_id), az elozes a visszaerkezesig tart.
    if (not _overtaking and can_overtake and lane_idx == 0
            and offset * to_lane > 0.0 and abs(offset) > 1.0):
        _overtaking, _to_lane, _target_id, _passed, _return_ok = \
            True, to_lane, surr["front_id"], False, False
        _overtake_t, _passed_t = 0.0, None
    if _overtaking:
        _overtake_t += 1.0 / env.fps
        # A target hosszanti helyzete az egohoz kepest [m], + elottem.
        lon = surr["lon"].get(_target_id)
        # Elhaladtam mellette: a target mar 5 m-rel mogottem van.
        if not _passed and lon is not None and lon < -5.0:
            _passed, _passed_t = True, _overtake_t
        # Amig nem a route savjaban vagyok, figyelem a visszafele atlependo
        # vonalat: csak szaggatotton at szabad visszasorolni (a bonuszhoz kell).
        if abs(offset) > lane_w / 2.0:
            _return_ok = surr["right_marking" if _to_lane > 0 else "left_marking"] == "Broken"

    # --- 4) Epizod-vege feltetelek ------------------------------------------
    # Megallt (5 s-ig 1 km/h alatt), lement az utrol (2 m-nel messzebb
    # barmelyik Driving sav kozepetol), vagy 60 km/h felett megy.
    if not env.terminal_state:
        _low_speed_timer = _low_speed_timer + 1.0 / env.fps if speed_kmh < 1.0 else 0.0
        # A legkozelebbi Driving sav kozepe (barmelyik sav, barmelyik irany).
        near = env.map.get_waypoint(loc).transform.location

        if _low_speed_timer > 5.0:
            env.terminal_state = True
            terminal_reason = "Vehicle stopped"
        elif np.hypot(loc.x - near.x, loc.y - near.y) > 2.0:
            env.terminal_state = True
            terminal_reason = "Off-track"
        elif speed_kmh > 60.0:
            env.terminal_state = True
            terminal_reason = "Too fast"

    # Terminal lepes: jarmuvel utkozes -30, minden mas -10.
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
            return -30
        return -10

    # --- 5) Jutalom ---------------------------------------------------------
    # speed_factor: 0..1, 50 km/h-nal 1. Alatta linearis, 50-60 km/h kozott
    # 0-ra esik (60 felett "Too fast" terminal).
    if speed_kmh <= 50.0:
        speed_factor = speed_kmh / 50.0
    else:
        speed_factor = 1.0 - (speed_kmh - 50.0) / 10.0
    speed_factor = float(np.clip(speed_factor, 0.0, 1.0))
    # lane_factor (elozeshez): milyen messze vagyok a legkozelebbi megengedett
    # sav kozepetol (a route savja es a celsav kozti barmelyik). Sav kozepen
    # 1.0, ket sav hataran 0.7 - ne a vonalon menjen.
    goal = _to_lane if _overtaking else (to_lane or 0)
    step = 1 if goal >= 0 else -1
    lane_d = min(abs(offset - n * lane_w) for n in range(0, goal + step, step))
    lane_factor = 1.0 - 0.3 * min(lane_d / (lane_w / 2.0), 1.0)
    # linger_factor (elozeshez): ne ragadjon a belso savban. 15 s elozes utan,
    # vagy a target melletti elhaladas utan savonkent 3 s utan 4 s alatt
    # 1.0-rol 0.3-ra esik.
    linger = 0.0
    if _overtaking:
        linger = _overtake_t - 15.0
        if _passed:
            linger = max(linger, _overtake_t - _passed_t - 3.0 * abs(_to_lane))
    linger_factor = max(1.0 - max(linger, 0.0) / 4.0, 0.3)
    # Elozes kozbeni lepesjutalom: a celsav kozepen 50 km/h-val 0.9, 25 km/h-val
    # 0.46. A 0.3-as alap azert van, hogy lassan elozni is jobb legyen, mint
    # a lead mogott kullogni.
    overtake_term = (0.3 + 0.6 * speed_factor ** 1.9) * lane_factor * linger_factor
    # Megengedett sav: a route savja, vagy elozes kozben a route savja es a
    # celsav kozotti savok.
    allowed_lane = lane_idx == 0 or (_overtaking and lane_idx * _to_lane > 0
                                     and abs(lane_idx) <= abs(_to_lane))

    lagging = False
    if not allowed_lane:
        # Rossz savban van (jobbra kiallt, vagy elozes nelkul balra ment).
        state = "Off lane %+d" % lane_idx
        reward = -0.5
    elif _overtaking:
        # Elozes: overtake_term. Ha visszaert a route savjanak kozepere (0.5
        # m-en belul), vege az elozesnek. Ha kozben elhaladt a target mellett,
        # es szaggatott vonalon jott vissza, +5 bonusz (autonkent egyszer).
        state = "Overtaking %s%d" % ("L" if _to_lane > 0 else "R", abs(_to_lane))
        reward = overtake_term
        if abs(offset) < 0.5:
            _overtaking = False
            # A visszasorolast a smoothness meg history-nyi (30) lepesig ne
            # buntesse cikkcakknak.
            _smooth_hold = env.distance_from_center_history.maxlen
            if _passed and _return_ok and _target_id not in _overtaken_ids:
                _overtaken_ids.add(_target_id)
                env.overtakes += 1
                reward += 5.0
        env.overtake_reward += reward
    else:
        # Sajat savban, nem eloz. Alap: sebesseg * centering * irany * simasag.
        # centering_factor: a sav kozepen 1, 2 m-re tole 0.
        centering_factor = max(1.0 - env.distance_from_center / 2.0, 0.0)

        # next_angle_factor: a route iranyaba nez 1, 90 fokra tole 0.
        next_angle = env.vehicle.get_angle(env.current_waypoint)
        next_angle_factor = max(1.0 - abs(next_angle) / np.deg2rad(90), 0.0)
        # smoothness_factor: a kozeptavolsag szorasa az utobbi 30 lepesben
        # (cikkcakk), 0 m-nel 1, 0.35 m-nel 0. Visszasorolas utan egy ideig 1.
        if _smooth_hold > 0:
            _smooth_hold -= 1
            smoothness_factor = 1.0
        else:
            std = np.std(env.distance_from_center_history)
            smoothness_factor = max(1.0 - abs(std) / 0.35, 0.0)

        ttc_term = 0.0
        if blocked:
            # Nem elozhet, kovetnie kell, sebesseg-buntetes nincs.
            # gap_term: lognormalis a headway-re (tavolsag / sajat sebesseg
            # [s]), a maximuma 2 s-nal van (ott 1.0). match: ne legyen lassabb
            # a leadnel. TTC (az utkozesig hatralevo ido) 7 s alatt log
            # buntetes, 1.5 s-nal -0.46.
            state = "Blocked"
            headway = surr["front_dist"] / max(speed_kmh / 3.6, 1.0)
            mu = np.log(2.0) + 0.4365 ** 2
            gap_term = 2.0 / headway * np.exp(
                0.4365 ** 2 / 2 - (np.log(headway) - mu) ** 2 / (2 * 0.4365 ** 2))
            match = min((speed_kmh + 1.0) / (surr["front_speed"] + 1.0), 1.0)
            speed_term = 0.8 * gap_term * match
            closing = (speed_kmh - surr["front_speed"]) / 3.6
            ttc = surr["front_dist"] / closing if closing > 0.0 else np.inf
            if ttc < 7.0:
                ttc_term = 0.3 * np.log(max(ttc, 1.5) / 7.0)
            env.blocked_time += 1.0 / env.fps
        elif can_overtake:
            # Elozhet (szabad a celsav). Lagging: 25 m-en belul van, es nem megy
            # legalabb 5 km/h-val gyorsabban a leadnel. Kulonben a szabad sav
            # jutalma jar. Itt nincs TTC-buntetes (a reward 2 ezt potolja).
            lagging = (surr["front_dist"] < 25.0
                       and speed_kmh < surr["front_speed"] + 5.0)
            state = "%s %s%d" % ("Lagging" if lagging else "Can overtake",
                                 "L" if to_lane > 0 else "R", abs(to_lane))
            if lagging:
                # Minel lassabb, es minel regebb ragad be, annal tobb (max -0.18).
                speed_term = -0.1 * (1.0 + 0.4 * (1.0 - speed_factor)
                                     + 0.4 * min(_lag_t / 3.0, 1.0))
            else:
                speed_term = speed_factor ** 1.9
        else:
            # Senki nincs elottem: szabad haladas.
            state = "Free"
            speed_term = speed_factor ** 1.9
        reward = speed_term * centering_factor * next_angle_factor * smoothness_factor
        # Lagging-nel a negativ speed_term nem szorzodik a faktorokkal, kulonben
        # a rossz savtartas csokkentene a buntetest.
        if lagging:
            reward = speed_term
        reward += ttc_term
        # Kihuzodasi rampa: elozhet, es mar a celsav fele mozdult (0 -> 1 m, a
        # kiallas elott). A jutalom linearisan atusztatodik a kovetesbol az
        # overtake_term-be, centering/smoothness nelkul.
        if can_overtake and offset * to_lane > 0.0:
            k = min(abs(offset) / 1.0, 1.0)
            reward = (1 - k) * speed_term * next_angle_factor + k * overtake_term
    # Miota lagging egyhuzamban [s], ezzel no a lagging buntetes.
    _lag_t = _lag_t + 1.0 / env.fps if lagging else 0.0

    # Kormany-rangatas buntetes: 2 * |a kormany valtozasa ebben a lepesben|.
    steer = env.vehicle.control.steer
    reward -= 2.0 * abs(steer - _prev_steer)
    _prev_steer = steer

    env.extra_info.extend([
        terminal_reason,
        "State:  % 19s" % state,
        "Overtakes:           % 7d" % env.overtakes,
        ""])
    return float(reward)


def reward_fn_2(env):
    # =========================================================================
    # REWARD 2 - a reward 1 + TTC-buntetes a "Can overtake" allapotban is.
    #
    # A reward 1-ben elozheto helyzetben (szabad a celsav) nem volt TTC, igy
    # megerte teljes sebesseggel rafutni a lead autora, es csak az utolso
    # pillanatban kihuzodni. Itt a "Blocked"-hoz hasonloan ott is jar a TTC
    # (7 s alatt, 0.3 * log), es a kihuzodas elso 1 m-en fogy el (rampa).
    # Minden mas azonos a reward 1-gyel, a ket valtozast "A:" jeloli.
    #
    # Eredmeny (SAC_reward_2_part2, az utolso 20% epizod atlaga): 2.1
    # elozes/epizod, 1086 m, 33.6 km/h, 66% jarmu-utkozes. Kevesebbet utkozik,
    # de nem mer hevesen elozni: a 7 s-os kuszob 50 vs 25 km/h-nal ~48 m,
    # tehat mar a lead meglatasakor (40 m) buntet, ezert inkabb lelassit,
    # vagy 25-40 m-rel lemarad (ott buntetlen, 25 km/h-nal +0.27/lepes). Ezt
    # javitja a reward 3.
    # =========================================================================
    global _low_speed_timer, _overtaking, _to_lane, _target_id, _passed, _return_ok
    global _overtaken_ids, _smooth_hold, _overtake_t, _passed_t, _prev_steer, _lag_t

    terminal_reason = "Running..."
    speed_kmh = env.vehicle.get_speed()
    # A forgalom az egohoz kepest. A step() lepesenkent egyszer szamolja
    # (get_vehicle_surroundings): lead auto, szomszed savok, vonalak, lon/auto.
    surr = env.surroundings
    # Epizod eleje: az elozesi allapotgep minden valtozoja nullazodik.
    if env.step_count == 0:
        _low_speed_timer = 0.0
        _overtaking, _to_lane, _target_id, _passed, _return_ok = False, 0, None, False, False
        _overtaken_ids = set()
        _smooth_hold = 0
        _overtake_t, _passed_t = 0.0, None
        _prev_steer = env.vehicle.control.steer
        _lag_t = 0.0

    # --- 1) Hol vagyok a route savjahoz kepest ------------------------------
    # offset: elojeles tavolsag a route savjanak kozepetol [m], + balra,
    # - jobbra. A distance_from_center elojel nelkuli, az oldalt a waypoint
    # jobb vektorara vetitett helyzet adja meg.
    wp = env.current_waypoint.transform
    loc = env.vehicle.get_transform().location
    r = wp.get_right_vector()
    side = (loc.x - wp.location.x) * r.x + (loc.y - wp.location.y) * r.y
    offset = env.distance_from_center if side < 0.0 else -env.distance_from_center

    lane_w = surr["lane_width"]

    # Hanyadik savban vagyok a route savjahoz kepest: 0 = a sajat, +1 = eggyel
    # balra, -1 = eggyel jobbra.
    lane_idx = int(round(offset / lane_w))

    # --- 2) Elozhetek-e most ------------------------------------------------
    # has_lead: van auto a sajat savomban elottem, 40 m-en belul.
    # to_lane: hova allhatok ki elozni. Csak balra: +1, vagy ha az foglalt, +2.
    # Feltetel: minden atlepett vonal szaggatott ("broken"), a celsav szabad
    # (-8..+20 m-en nincs benne auto), +2-nel a +1-es savon at lehet vagni
    # ("cross": mellettem +-8 m-en nincs benne senki).
    has_lead = surr["front_id"] is not None
    to_lane = None
    if has_lead:
        lanes = surr["lanes"]
        for n in (1, 2):
            lane = lanes.get(n)
            if lane and lane["broken"] and lane["free"] and (abs(n) == 1 or lanes[n // 2]["cross"]):
                to_lane = n
                break
    can_overtake = to_lane is not None
    # blocked: van elottem auto, de nem elozhetem meg, kovetni kell.
    blocked = has_lead and not can_overtake

    # --- 3) Elozesi allapotgep ----------------------------------------------
    # Kiallas: elozhetek, meg a sajat savomban vagyok, es mar 1 m-nel tobbet
    # mozdultam a celsav fele. Ekkor rogzul a celsav (_to_lane) es a
    # megelozendo auto (_target_id), az elozes a visszaerkezesig tart.
    if (not _overtaking and can_overtake and lane_idx == 0
            and offset * to_lane > 0.0 and abs(offset) > 1.0):
        _overtaking, _to_lane, _target_id, _passed, _return_ok = \
            True, to_lane, surr["front_id"], False, False
        _overtake_t, _passed_t = 0.0, None
    if _overtaking:
        _overtake_t += 1.0 / env.fps
        # A target hosszanti helyzete az egohoz kepest [m], + elottem.
        lon = surr["lon"].get(_target_id)
        # Elhaladtam mellette: a target mar 5 m-rel mogottem van.
        if not _passed and lon is not None and lon < -5.0:
            _passed, _passed_t = True, _overtake_t
        # Amig nem a route savjaban vagyok, figyelem a visszafele atlependo
        # vonalat: csak szaggatotton at szabad visszasorolni (a bonuszhoz kell).
        if abs(offset) > lane_w / 2.0:
            _return_ok = surr["right_marking" if _to_lane > 0 else "left_marking"] == "Broken"

    # --- 4) Epizod-vege feltetelek ------------------------------------------
    # Megallt (5 s-ig 1 km/h alatt), lement az utrol (2 m-nel messzebb
    # barmelyik Driving sav kozepetol), vagy 60 km/h felett megy.
    if not env.terminal_state:
        _low_speed_timer = _low_speed_timer + 1.0 / env.fps if speed_kmh < 1.0 else 0.0
        # A legkozelebbi Driving sav kozepe (barmelyik sav, barmelyik irany).
        near = env.map.get_waypoint(loc).transform.location

        if _low_speed_timer > 5.0:
            env.terminal_state = True
            terminal_reason = "Vehicle stopped"
        elif np.hypot(loc.x - near.x, loc.y - near.y) > 2.0:
            env.terminal_state = True
            terminal_reason = "Off-track"
        elif speed_kmh > 60.0:
            env.terminal_state = True
            terminal_reason = "Too fast"

    # Terminal lepes: jarmuvel utkozes -30, minden mas -10.
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
            return -30
        return -10

    # --- 5) Jutalom ---------------------------------------------------------
    # speed_factor: 0..1, 50 km/h-nal 1. Alatta linearis, 50-60 km/h kozott
    # 0-ra esik (60 felett "Too fast" terminal).
    if speed_kmh <= 50.0:
        speed_factor = speed_kmh / 50.0
    else:
        speed_factor = 1.0 - (speed_kmh - 50.0) / 10.0
    speed_factor = float(np.clip(speed_factor, 0.0, 1.0))
    # lane_factor (elozeshez): milyen messze vagyok a legkozelebbi megengedett
    # sav kozepetol (a route savja es a celsav kozti barmelyik). Sav kozepen
    # 1.0, ket sav hataran 0.7 - ne a vonalon menjen.
    goal = _to_lane if _overtaking else (to_lane or 0)
    step = 1 if goal >= 0 else -1
    lane_d = min(abs(offset - n * lane_w) for n in range(0, goal + step, step))
    lane_factor = 1.0 - 0.3 * min(lane_d / (lane_w / 2.0), 1.0)
    # linger_factor (elozeshez): ne ragadjon a belso savban. 15 s elozes utan,
    # vagy a target melletti elhaladas utan savonkent 3 s utan 4 s alatt
    # 1.0-rol 0.3-ra esik.
    linger = 0.0
    if _overtaking:
        linger = _overtake_t - 15.0
        if _passed:
            linger = max(linger, _overtake_t - _passed_t - 3.0 * abs(_to_lane))
    linger_factor = max(1.0 - max(linger, 0.0) / 4.0, 0.3)
    # Elozes kozbeni lepesjutalom: a celsav kozepen 50 km/h-val 0.9, 25 km/h-val
    # 0.46. A 0.3-as alap azert van, hogy lassan elozni is jobb legyen, mint
    # a lead mogott kullogni.
    overtake_term = (0.3 + 0.6 * speed_factor ** 1.9) * lane_factor * linger_factor
    # Megengedett sav: a route savja, vagy elozes kozben a route savja es a
    # celsav kozotti savok.
    allowed_lane = lane_idx == 0 or (_overtaking and lane_idx * _to_lane > 0
                                     and abs(lane_idx) <= abs(_to_lane))

    lagging = False
    if not allowed_lane:
        # Rossz savban van (jobbra kiallt, vagy elozes nelkul balra ment).
        state = "Off lane %+d" % lane_idx
        reward = -0.5
    elif _overtaking:
        # Elozes: overtake_term. Ha visszaert a route savjanak kozepere (0.5
        # m-en belul), vege az elozesnek. Ha kozben elhaladt a target mellett,
        # es szaggatott vonalon jott vissza, +5 bonusz (autonkent egyszer).
        state = "Overtaking %s%d" % ("L" if _to_lane > 0 else "R", abs(_to_lane))
        reward = overtake_term
        if abs(offset) < 0.5:
            _overtaking = False
            # A visszasorolast a smoothness meg history-nyi (30) lepesig ne
            # buntesse cikkcakknak.
            _smooth_hold = env.distance_from_center_history.maxlen
            if _passed and _return_ok and _target_id not in _overtaken_ids:
                _overtaken_ids.add(_target_id)
                env.overtakes += 1
                reward += 5.0
        env.overtake_reward += reward
    else:
        # Sajat savban, nem eloz. Alap: sebesseg * centering * irany * simasag.
        # centering_factor: a sav kozepen 1, 2 m-re tole 0.
        centering_factor = max(1.0 - env.distance_from_center / 2.0, 0.0)

        # next_angle_factor: a route iranyaba nez 1, 90 fokra tole 0.
        next_angle = env.vehicle.get_angle(env.current_waypoint)
        next_angle_factor = max(1.0 - abs(next_angle) / np.deg2rad(90), 0.0)
        # smoothness_factor: a kozeptavolsag szorasa az utobbi 30 lepesben
        # (cikkcakk), 0 m-nel 1, 0.35 m-nel 0. Visszasorolas utan egy ideig 1.
        if _smooth_hold > 0:
            _smooth_hold -= 1
            smoothness_factor = 1.0
        else:
            std = np.std(env.distance_from_center_history)
            smoothness_factor = max(1.0 - abs(std) / 0.35, 0.0)

        ttc_term = 0.0
        if blocked:
            # Nem elozhet, kovetnie kell, sebesseg-buntetes nincs.
            # gap_term: lognormalis a headway-re (tavolsag / sajat sebesseg
            # [s]), a maximuma 2 s-nal van (ott 1.0). match: ne legyen lassabb
            # a leadnel. TTC (az utkozesig hatralevo ido) 7 s alatt log
            # buntetes, 1.5 s-nal -0.46.
            state = "Blocked"
            headway = surr["front_dist"] / max(speed_kmh / 3.6, 1.0)
            mu = np.log(2.0) + 0.4365 ** 2
            gap_term = 2.0 / headway * np.exp(
                0.4365 ** 2 / 2 - (np.log(headway) - mu) ** 2 / (2 * 0.4365 ** 2))
            match = min((speed_kmh + 1.0) / (surr["front_speed"] + 1.0), 1.0)
            speed_term = 0.8 * gap_term * match
            closing = (speed_kmh - surr["front_speed"]) / 3.6
            ttc = surr["front_dist"] / closing if closing > 0.0 else np.inf
            if ttc < 7.0:
                ttc_term = 0.3 * np.log(max(ttc, 1.5) / 7.0)
            env.blocked_time += 1.0 / env.fps
        elif can_overtake:
            # A: itt is jar a TTC-buntetes, hogy ne erje meg teljes sebesseggel
            # rafutni a lead autora, mielott kihuzodik.
            closing = (speed_kmh - surr["front_speed"]) / 3.6
            ttc = surr["front_dist"] / closing if closing > 0.0 else np.inf
            if ttc < 7.0:
                ttc_term = 0.3 * np.log(max(ttc, 1.5) / 7.0)
            # Lagging: 25 m-en belul van, es nem megy legalabb 5 km/h-val
            # gyorsabban a leadnel. Kulonben a szabad sav jutalma jar.
            lagging = (surr["front_dist"] < 25.0
                       and speed_kmh < surr["front_speed"] + 5.0)
            state = "%s %s%d" % ("Lagging" if lagging else "Can overtake",
                                 "L" if to_lane > 0 else "R", abs(to_lane))
            if lagging:
                # Minel lassabb, es minel regebb ragad be, annal tobb (max -0.18).
                speed_term = -0.1 * (1.0 + 0.4 * (1.0 - speed_factor)
                                     + 0.4 * min(_lag_t / 3.0, 1.0))
            else:
                speed_term = speed_factor ** 1.9
        else:
            # Senki nincs elottem: szabad haladas.
            state = "Free"
            speed_term = speed_factor ** 1.9
        reward = speed_term * centering_factor * next_angle_factor * smoothness_factor
        # Lagging-nel a negativ speed_term nem szorzodik a faktorokkal, kulonben
        # a rossz savtartas csokkentene a buntetest.
        if lagging:
            reward = speed_term
        reward += ttc_term
        # Kihuzodasi rampa: elozhet, es mar a celsav fele mozdult (0 -> 1 m, a
        # kiallas elott). A jutalom linearisan atusztatodik a kovetesbol az
        # overtake_term-be, centering/smoothness nelkul.
        if can_overtake and offset * to_lane > 0.0:
            k = min(abs(offset) / 1.0, 1.0)
            # A: a TTC-buntetes a kihuzodas elso 1 m-en fogy el.
            reward = (1 - k) * (speed_term * next_angle_factor + ttc_term) + k * overtake_term
    # Miota lagging egyhuzamban [s], ezzel no a lagging buntetes.
    _lag_t = _lag_t + 1.0 / env.fps if lagging else 0.0

    # Kormany-rangatas buntetes: 2 * |a kormany valtozasa ebben a lepesben|.
    steer = env.vehicle.control.steer
    reward -= 2.0 * abs(steer - _prev_steer)
    _prev_steer = steer

    env.extra_info.extend([
        terminal_reason,
        "State:  % 19s" % state,
        "Overtakes:           % 7d" % env.overtakes,
        ""])
    return float(reward)


def reward_fn_3(env):
    # =========================================================================
    # REWARD 3 - a reward 2 hangolasa batrabb elozesre.
    #
    # A reward 2 felepitese marad (allapotgep, kovetes, TTC, lagging, rampa),
    # csak 4 helyen valtozik, a kodban "R3" jeloli oket. A reward 2-vel a
    # modell nem mert hevesen elozni, a lead auto mogott inkabb lelassitott
    # vagy lemaradt. Az okok, es amit itt valtoztatunk rajtuk:
    #
    #   R3-A) A "Can overtake" TTC-buntetes 7 s-os kuszobe 50 vs 25 km/h-nal
    #         ~48 m, vagyis mar akkor buntetett, amikor a lead autot meglatta
    #         (40 m). Uj: kuszob 3 s, suly 0.3 -> 0.6. Csak a keso kihuzodast
    #         bunteti (50 vs 25 km/h-nal ~21 m-en belul), a gyors rafutast nem.
    #         A "Blocked" TTC (7 s, 0.3) nem valtozik.
    #   R3-B) Elozes kozben a lepesjutalom max 0.9 volt, kevesebb, mint a sajat
    #         savban 50 km/h-val (1.0): kiallni sosem erte meg jobban.
    #         Uj: 0.3 + 0.6*v^1.9 -> 0.4 + 0.7*v^1.9, 50 km/h-val 1.1.
    #   R3-D) 25-40 m-rel lemaradva a lead sebessegevel menni buntetlen volt,
    #         sot pozitiv (25 km/h-nal +0.27/lepes). Uj: a lagging-nek nincs
    #         tavolsagfeltetele, a teljes 40 m-es lead-tavon jar.
    #   R3-E) A +5 bonusz a kiallas utan ~5 s-mal jott. Gamma 0.98 mellett
    #         (20 fps, ~2.5 s horizont) ez a kiallaskor csak ~0.66-ot ert.
    #         Uj: +2 az elhaladaskor (target 5 m-rel mogottem, ~2.5 s mulva,
    #         ~0.73-at er) es +3 a szabalyos visszasorolaskor (~0.40),
    #         osszesen ~1.13. A gyorsabb elhaladast a diszkont kevesbe nyeli el.
    #
    # Lepesjutalmak 25 km/h-s lead mogott, szabad bal savval (sav kozepen,
    # egyenesen, kormanybuntetes nelkul):          reward 2       reward 3
    #   kulloges 25 km/h-val 25 m-en belul (lagging)   -0.12..-0.16   ugyanaz
    #   25-40 m-rel lemaradva 25 km/h-val               +0.27          -0.12..-0.16
    #   rafutas 50 km/h-val, 30 m-re                     0.86           1.0
    #   rafutas 50 km/h-val, 15 m-re                     0.65           0.80
    #   elozes 50 km/h-val a bal sav kozepen             0.90           1.10
    #   elozes 25 km/h-val a bal sav kozepen             0.46           0.59
    # =========================================================================
    global _low_speed_timer, _overtaking, _to_lane, _target_id, _passed, _return_ok
    global _overtaken_ids, _smooth_hold, _overtake_t, _passed_t, _prev_steer, _lag_t

    terminal_reason = "Running..."
    speed_kmh = env.vehicle.get_speed()
    # A forgalom az egohoz kepest. A step() lepesenkent egyszer szamolja
    # (get_vehicle_surroundings): lead auto, szomszed savok, vonalak, lon/auto.
    surr = env.surroundings
    # Epizod eleje: az elozesi allapotgep minden valtozoja nullazodik.
    if env.step_count == 0:
        _low_speed_timer = 0.0
        _overtaking, _to_lane, _target_id, _passed, _return_ok = False, 0, None, False, False
        _overtaken_ids = set()
        _smooth_hold = 0
        _overtake_t, _passed_t = 0.0, None
        _prev_steer = env.vehicle.control.steer
        _lag_t = 0.0

    # --- 1) Hol vagyok a route savjahoz kepest ------------------------------
    # offset: elojeles tavolsag a route savjanak kozepetol [m], + balra,
    # - jobbra. A distance_from_center elojel nelkuli, az oldalt a waypoint
    # jobb vektorara vetitett helyzet adja meg.
    wp = env.current_waypoint.transform
    loc = env.vehicle.get_transform().location
    r = wp.get_right_vector()
    side = (loc.x - wp.location.x) * r.x + (loc.y - wp.location.y) * r.y
    offset = env.distance_from_center if side < 0.0 else -env.distance_from_center

    lane_w = surr["lane_width"]

    # Hanyadik savban vagyok a route savjahoz kepest: 0 = a sajat, +1 = eggyel
    # balra, -1 = eggyel jobbra.
    lane_idx = int(round(offset / lane_w))

    # --- 2) Elozhetek-e most ------------------------------------------------
    # has_lead: van auto a sajat savomban elottem, 40 m-en belul.
    # to_lane: hova allhatok ki elozni. Csak balra: +1, vagy ha az foglalt, +2.
    # Feltetel: minden atlepett vonal szaggatott ("broken"), a celsav szabad
    # (-8..+20 m-en nincs benne auto), +2-nel a +1-es savon at lehet vagni
    # ("cross": mellettem +-8 m-en nincs benne senki).
    has_lead = surr["front_id"] is not None
    to_lane = None
    if has_lead:
        lanes = surr["lanes"]
        for n in (1, 2):
            lane = lanes.get(n)
            if lane and lane["broken"] and lane["free"] and (abs(n) == 1 or lanes[n // 2]["cross"]):
                to_lane = n
                break
    can_overtake = to_lane is not None
    # blocked: van elottem auto, de nem elozhetem meg, kovetni kell.
    blocked = has_lead and not can_overtake

    # --- 3) Elozesi allapotgep ----------------------------------------------
    # Kiallas: elozhetek, meg a sajat savomban vagyok, es mar 1 m-nel tobbet
    # mozdultam a celsav fele. Ekkor rogzul a celsav (_to_lane) es a
    # megelozendo auto (_target_id), az elozes a visszaerkezesig tart.
    if (not _overtaking and can_overtake and lane_idx == 0
            and offset * to_lane > 0.0 and abs(offset) > 1.0):
        _overtaking, _to_lane, _target_id, _passed, _return_ok = \
            True, to_lane, surr["front_id"], False, False
        _overtake_t, _passed_t = 0.0, None
    # R3-E: elhaladasi bonusz, ha ebben a lepesben haladt el (lent adjuk hozza).
    pass_bonus = 0.0
    if _overtaking:
        _overtake_t += 1.0 / env.fps
        # A target hosszanti helyzete az egohoz kepest [m], + elottem.
        lon = surr["lon"].get(_target_id)
        # Elhaladtam mellette: a target mar 5 m-rel mogottem van.
        if not _passed and lon is not None and lon < -5.0:
            _passed, _passed_t = True, _overtake_t
            # R3-E: +2 az elhaladaskor, autonkent egyszer.
            if _target_id not in _overtaken_ids:
                pass_bonus = 2.0
        # Amig nem a route savjaban vagyok, figyelem a visszafele atlependo
        # vonalat: csak szaggatotton at szabad visszasorolni (a bonuszhoz kell).
        if abs(offset) > lane_w / 2.0:
            _return_ok = surr["right_marking" if _to_lane > 0 else "left_marking"] == "Broken"

    # --- 4) Epizod-vege feltetelek ------------------------------------------
    # Megallt (5 s-ig 1 km/h alatt), lement az utrol (2 m-nel messzebb
    # barmelyik Driving sav kozepetol), vagy 60 km/h felett megy.
    if not env.terminal_state:
        _low_speed_timer = _low_speed_timer + 1.0 / env.fps if speed_kmh < 1.0 else 0.0
        # A legkozelebbi Driving sav kozepe (barmelyik sav, barmelyik irany).
        near = env.map.get_waypoint(loc).transform.location

        if _low_speed_timer > 5.0:
            env.terminal_state = True
            terminal_reason = "Vehicle stopped"
        elif np.hypot(loc.x - near.x, loc.y - near.y) > 2.0:
            env.terminal_state = True
            terminal_reason = "Off-track"
        elif speed_kmh > 60.0:
            env.terminal_state = True
            terminal_reason = "Too fast"

    # Terminal lepes: jarmuvel utkozes -30, minden mas -10.
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
            return -30
        return -10

    # --- 5) Jutalom ---------------------------------------------------------
    # speed_factor: 0..1, 50 km/h-nal 1. Alatta linearis, 50-60 km/h kozott
    # 0-ra esik (60 felett "Too fast" terminal).
    if speed_kmh <= 50.0:
        speed_factor = speed_kmh / 50.0
    else:
        speed_factor = 1.0 - (speed_kmh - 50.0) / 10.0
    speed_factor = float(np.clip(speed_factor, 0.0, 1.0))
    # lane_factor (elozeshez): milyen messze vagyok a legkozelebbi megengedett
    # sav kozepetol (a route savja es a celsav kozti barmelyik). Sav kozepen
    # 1.0, ket sav hataran 0.7 - ne a vonalon menjen.
    goal = _to_lane if _overtaking else (to_lane or 0)
    step = 1 if goal >= 0 else -1
    lane_d = min(abs(offset - n * lane_w) for n in range(0, goal + step, step))
    lane_factor = 1.0 - 0.3 * min(lane_d / (lane_w / 2.0), 1.0)
    # linger_factor (elozeshez): ne ragadjon a belso savban. 15 s elozes utan,
    # vagy a target melletti elhaladas utan savonkent 3 s utan 4 s alatt
    # 1.0-rol 0.3-ra esik.
    linger = 0.0
    if _overtaking:
        linger = _overtake_t - 15.0
        if _passed:
            linger = max(linger, _overtake_t - _passed_t - 3.0 * abs(_to_lane))
    linger_factor = max(1.0 - max(linger, 0.0) / 4.0, 0.3)
    # R3-B: elozes kozbeni lepesjutalom (reward 2: 0.3 + 0.6 * ...). A celsav
    # kozepen 50 km/h-val 1.1, 25 km/h-val 0.59. Gyorsan elozni igy tobbet
    # er, mint a sajat savban maradni (1.0), az elhaladas utan pedig a
    # linger_factor kuldi vissza.
    overtake_term = (0.4 + 0.7 * speed_factor ** 1.9) * lane_factor * linger_factor
    # Megengedett sav: a route savja, vagy elozes kozben a route savja es a
    # celsav kozotti savok.
    allowed_lane = lane_idx == 0 or (_overtaking and lane_idx * _to_lane > 0
                                     and abs(lane_idx) <= abs(_to_lane))

    lagging = False
    if not allowed_lane:
        # Rossz savban van (jobbra kiallt, vagy elozes nelkul balra ment).
        state = "Off lane %+d" % lane_idx
        reward = -0.5
    elif _overtaking:
        # Elozes: overtake_term (+ R3-E elhaladasi bonusz). Ha visszaert a route
        # savjanak kozepere (0.5 m-en belul), vege az elozesnek. Ha kozben
        # elhaladt a target mellett, es szaggatott vonalon jott vissza, +3.
        state = "Overtaking %s%d" % ("L" if _to_lane > 0 else "R", abs(_to_lane))
        reward = overtake_term + pass_bonus
        if abs(offset) < 0.5:
            _overtaking = False
            # A visszasorolast a smoothness meg history-nyi (30) lepesig ne
            # buntesse cikkcakknak.
            _smooth_hold = env.distance_from_center_history.maxlen
            if _passed and _return_ok and _target_id not in _overtaken_ids:
                _overtaken_ids.add(_target_id)
                env.overtakes += 1
                # R3-E: +3 a szabalyos visszasorolaskor (reward 2: +5).
                reward += 3.0
        env.overtake_reward += reward
    else:
        # Sajat savban, nem eloz. Alap: sebesseg * centering * irany * simasag.
        # centering_factor: a sav kozepen 1, 2 m-re tole 0.
        centering_factor = max(1.0 - env.distance_from_center / 2.0, 0.0)

        # next_angle_factor: a route iranyaba nez 1, 90 fokra tole 0.
        next_angle = env.vehicle.get_angle(env.current_waypoint)
        next_angle_factor = max(1.0 - abs(next_angle) / np.deg2rad(90), 0.0)
        # smoothness_factor: a kozeptavolsag szorasa az utobbi 30 lepesben
        # (cikkcakk), 0 m-nel 1, 0.35 m-nel 0. Visszasorolas utan egy ideig 1.
        if _smooth_hold > 0:
            _smooth_hold -= 1
            smoothness_factor = 1.0
        else:
            std = np.std(env.distance_from_center_history)
            smoothness_factor = max(1.0 - abs(std) / 0.35, 0.0)

        ttc_term = 0.0
        if blocked:
            # Nem elozhet, kovetnie kell, sebesseg-buntetes nincs.
            # gap_term: lognormalis a headway-re (tavolsag / sajat sebesseg
            # [s]), a maximuma ~2 s korul van. match: ne legyen lassabb a
            # leadnel. TTC (az utkozesig hatralevo ido) 7 s alatt log
            # buntetes, 1.5 s-nal -0.46.
            state = "Blocked"
            headway = surr["front_dist"] / max(speed_kmh / 3.6, 1.0)
            mu = np.log(2.0) + 0.4365 ** 2
            gap_term = 2.0 / headway * np.exp(
                0.4365 ** 2 / 2 - (np.log(headway) - mu) ** 2 / (2 * 0.4365 ** 2))
            match = min((speed_kmh + 1.0) / (surr["front_speed"] + 1.0), 1.0)
            speed_term = 0.8 * gap_term * match
            closing = (speed_kmh - surr["front_speed"]) / 3.6
            ttc = surr["front_dist"] / closing if closing > 0.0 else np.inf
            if ttc < 7.0:
                ttc_term = 0.3 * np.log(max(ttc, 1.5) / 7.0)
            env.blocked_time += 1.0 / env.fps
        elif can_overtake:
            # R3-A: TTC-buntetes csak 3 s alatt, 0.6-os sullyal (reward 2: 7 s,
            # 0.3). 2 s-nal -0.24, 1.5 s-nal -0.42. 50 vs 25 km/h-nal ~21 m-en
            # belul kezdodik: addig teljes sebesseggel rafuthat, de addigra ki
            # kell huzodnia. A kihuzodas elso 1 m-en elfogy (lent, rampa).
            closing = (speed_kmh - surr["front_speed"]) / 3.6
            ttc = surr["front_dist"] / closing if closing > 0.0 else np.inf
            if ttc < 3.0:
                ttc_term = 0.6 * np.log(max(ttc, 1.5) / 3.0)
            # R3-D: lagging = elozhetne, de nem megy legalabb 5 km/h-val
            # gyorsabban a leadnel, a teljes 40 m-es lead-tavon (reward 2: csak
            # 25 m-en belul, igy 25-40 m-rel lemaradva buntetlenul kovethetett).
            lagging = speed_kmh < surr["front_speed"] + 5.0
            state = "%s %s%d" % ("Lagging" if lagging else "Can overtake",
                                 "L" if to_lane > 0 else "R", abs(to_lane))
            if lagging:
                # Minel lassabb, es minel regebb ragad be, annal tobb (max -0.18).
                speed_term = -0.1 * (1.0 + 0.4 * (1.0 - speed_factor)
                                     + 0.4 * min(_lag_t / 3.0, 1.0))
            else:
                speed_term = speed_factor ** 1.9
        else:
            # Senki nincs elottem: szabad haladas.
            state = "Free"
            speed_term = speed_factor ** 1.9
        reward = speed_term * centering_factor * next_angle_factor * smoothness_factor
        # Lagging-nel a negativ speed_term nem szorzodik a faktorokkal, kulonben
        # a rossz savtartas csokkentene a buntetest.
        if lagging:
            reward = speed_term
        reward += ttc_term
        # Kihuzodasi rampa: elozhet, es mar a celsav fele mozdult (0 -> 1 m, a
        # kiallas elott). A jutalom linearisan atusztatodik a kovetesbol az
        # overtake_term-be, centering/smoothness nelkul, a TTC is elfogy.
        if can_overtake and offset * to_lane > 0.0:
            k = min(abs(offset) / 1.0, 1.0)
            reward = (1 - k) * (speed_term * next_angle_factor + ttc_term) + k * overtake_term
    # Miota lagging egyhuzamban [s], ezzel no a lagging buntetes.
    _lag_t = _lag_t + 1.0 / env.fps if lagging else 0.0

    # Kormany-rangatas buntetes: 2 * |a kormany valtozasa ebben a lepesben|.
    steer = env.vehicle.control.steer
    reward -= 2.0 * abs(steer - _prev_steer)
    _prev_steer = steer

    env.extra_info.extend([
        terminal_reason,
        "State:  % 19s" % state,
        "Overtakes:           % 7d" % env.overtakes,
        ""])
    return float(reward)


def reward_fn_4(env):
    # =========================================================================
    # REWARD 4 - haladas-alapu jutalom. Mas megkozelites, mint az 1-3.
    #
    # Az 1-3 jutalmakban minden forgalmi helyzetre kezzel formalt tag van
    # (kovetesi gap_term, TTC-buntetes, lagging-buntetes, kihuzodasi rampa,
    # centering/smoothness szorzok, sebesseg^1.9). Itt ezek helyett egyetlen
    # fizikai mennyiseg a jutalom: mennyit halad a route menten egy lepesben.
    # Hipotezis: a lassu auto mogott a haladas magatol kisebb (25 km/h-nal
    # fele annyi, mint 50-nel), igy az elozes kezi formalas nelkul is megeri.
    # A kerdes: a kezi tagok nelkul jobban vagy rosszabbul tanul-e elozni.
    #
    # Ami MARAD az 1-3-bol (ezek a feladat szabalyai, nem formalas):
    #   - a terminal feltetelek es ertekek (-30 jarmu-utkozes, -10 a tobbi),
    #   - az elozesi allapotgep: csak balra, szaggatott vonalon, szabad savba,
    #     rossz savban -0.5, a target melletti elhaladas utan vissza kell
    #     sorolni (linger_factor), szabalyos elozesert +5 bonusz,
    #   - a kormany-rangatas buntetes (kenyelem),
    #   - a metrikak (overtakes, overtake_reward, blocked_time) ugyanugy, hogy
    #     a futasok osszehasonlithatok legyenek.
    # Ami KIMARAD: gap_term, TTC, lagging, rampa, centering, smoothness, ^1.9.
    # Az utkozest csak a -30 terminal tanitja.
    #
    # Lepesjutalom minden megengedett savban: haladas * lane_factor * linger_factor
    #   50 km/h a sav kozepen                    1.0
    #   25 km/h-s lead mogott, a sebessegevel     0.5
    #   elozes 50 km/h-val a bal sav kozepen      1.0
    #   50 km/h ket sav hataran (kihuzodas)       0.7
    # =========================================================================
    global _low_speed_timer, _overtaking, _to_lane, _target_id, _passed, _return_ok
    global _overtaken_ids, _overtake_t, _passed_t, _prev_steer

    terminal_reason = "Running..."
    speed_kmh = env.vehicle.get_speed()
    # A forgalom az egohoz kepest. A step() lepesenkent egyszer szamolja
    # (get_vehicle_surroundings): lead auto, szomszed savok, vonalak, lon/auto.
    surr = env.surroundings
    # Epizod eleje: az elozesi allapotgep minden valtozoja nullazodik.
    if env.step_count == 0:
        _low_speed_timer = 0.0
        _overtaking, _to_lane, _target_id, _passed, _return_ok = False, 0, None, False, False
        _overtaken_ids = set()
        _overtake_t, _passed_t = 0.0, None
        _prev_steer = env.vehicle.control.steer

    # --- 1) Hol vagyok a route savjahoz kepest ------------------------------
    # offset: elojeles tavolsag a route savjanak kozepetol [m], + balra,
    # - jobbra. A distance_from_center elojel nelkuli, az oldalt a waypoint
    # jobb vektorara vetitett helyzet adja meg.
    wp = env.current_waypoint.transform
    loc = env.vehicle.get_transform().location
    r = wp.get_right_vector()
    side = (loc.x - wp.location.x) * r.x + (loc.y - wp.location.y) * r.y
    offset = env.distance_from_center if side < 0.0 else -env.distance_from_center

    lane_w = surr["lane_width"]

    # Hanyadik savban vagyok a route savjahoz kepest: 0 = a sajat, +1 = eggyel
    # balra, -1 = eggyel jobbra.
    lane_idx = int(round(offset / lane_w))

    # --- 2) Elozhetek-e most ------------------------------------------------
    # has_lead: van auto a sajat savomban elottem, 40 m-en belul.
    # to_lane: hova allhatok ki elozni. Csak balra: +1, vagy ha az foglalt, +2.
    # Feltetel: minden atlepett vonal szaggatott ("broken"), a celsav szabad
    # (-8..+20 m-en nincs benne auto), +2-nel a +1-es savon at lehet vagni
    # ("cross": mellettem +-8 m-en nincs benne senki).
    has_lead = surr["front_id"] is not None
    to_lane = None
    if has_lead:
        lanes = surr["lanes"]
        for n in (1, 2):
            lane = lanes.get(n)
            if lane and lane["broken"] and lane["free"] and (abs(n) == 1 or lanes[n // 2]["cross"]):
                to_lane = n
                break
    can_overtake = to_lane is not None
    # blocked: van elottem auto, de nem elozhetem meg, kovetni kell.
    blocked = has_lead and not can_overtake

    # --- 3) Elozesi allapotgep ----------------------------------------------
    # Kiallas: elozhetek, meg a sajat savomban vagyok, es mar 1 m-nel tobbet
    # mozdultam a celsav fele. Ekkor rogzul a celsav (_to_lane) es a
    # megelozendo auto (_target_id), az elozes a visszaerkezesig tart.
    if (not _overtaking and can_overtake and lane_idx == 0
            and offset * to_lane > 0.0 and abs(offset) > 1.0):
        _overtaking, _to_lane, _target_id, _passed, _return_ok = \
            True, to_lane, surr["front_id"], False, False
        _overtake_t, _passed_t = 0.0, None
    if _overtaking:
        _overtake_t += 1.0 / env.fps
        # A target hosszanti helyzete az egohoz kepest [m], + elottem.
        lon = surr["lon"].get(_target_id)
        # Elhaladtam mellette: a target mar 5 m-rel mogottem van.
        if not _passed and lon is not None and lon < -5.0:
            _passed, _passed_t = True, _overtake_t
        # Amig nem a route savjaban vagyok, figyelem a visszafele atlependo
        # vonalat: csak szaggatotton at szabad visszasorolni (a bonuszhoz kell).
        if abs(offset) > lane_w / 2.0:
            _return_ok = surr["right_marking" if _to_lane > 0 else "left_marking"] == "Broken"

    # --- 4) Epizod-vege feltetelek ------------------------------------------
    # Megallt (5 s-ig 1 km/h alatt), lement az utrol (2 m-nel messzebb
    # barmelyik Driving sav kozepetol), vagy 60 km/h felett megy.
    if not env.terminal_state:
        _low_speed_timer = _low_speed_timer + 1.0 / env.fps if speed_kmh < 1.0 else 0.0
        # A legkozelebbi Driving sav kozepe (barmelyik sav, barmelyik irany).
        near = env.map.get_waypoint(loc).transform.location

        if _low_speed_timer > 5.0:
            env.terminal_state = True
            terminal_reason = "Vehicle stopped"
        elif np.hypot(loc.x - near.x, loc.y - near.y) > 2.0:
            env.terminal_state = True
            terminal_reason = "Off-track"
        elif speed_kmh > 60.0:
            env.terminal_state = True
            terminal_reason = "Too fast"

    # Terminal lepes: jarmuvel utkozes -30, minden mas -10.
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
            return -30
        return -10

    # --- 5) Jutalom ---------------------------------------------------------
    # speed_factor: 0..1, 50 km/h-nal 1. Alatta linearis, 50-60 km/h kozott
    # 0-ra esik (60 felett "Too fast" terminal).
    if speed_kmh <= 50.0:
        speed_factor = speed_kmh / 50.0
    else:
        speed_factor = 1.0 - (speed_kmh - 50.0) / 10.0
    speed_factor = float(np.clip(speed_factor, 0.0, 1.0))
    # Haladas: a route iranyu sebesseg 50 km/h-ra normalva. A cos(szog) a
    # route-hoz kepest ferden menest veszi le (savvaltasnal ~5-10 fok, ez csak
    # ~1%), visszafele menve negativ. Linearis a sebessegben: 25 km/h fele
    # annyi haladas, mint 50 km/h.
    progress = speed_factor * np.cos(env.vehicle.get_angle(env.current_waypoint))
    # lane_factor: az egyetlen savtartasi tag. Milyen messze vagyok a
    # legkozelebbi megengedett sav kozepetol: ha elozhetek vagy elozok, a route
    # savja es a celsav kozti barmelyik sav jo, kulonben csak a route savja.
    # Sav kozepen 1.0, ket sav hataran (1.75 m-re a kozeptol) 0.7.
    goal = _to_lane if _overtaking else (to_lane or 0)
    step = 1 if goal >= 0 else -1
    lane_d = min(abs(offset - n * lane_w) for n in range(0, goal + step, step))
    lane_factor = 1.0 - 0.3 * min(lane_d / (lane_w / 2.0), 1.0)
    # linger_factor (elozeshez): ne ragadjon a belso savban (jobbra tartas).
    # 15 s elozes utan, vagy a target melletti elhaladas utan savonkent 3 s
    # utan 4 s alatt 1.0-rol 0.3-ra esik. Elozesen kivul 1.0.
    linger = 0.0
    if _overtaking:
        linger = _overtake_t - 15.0
        if _passed:
            linger = max(linger, _overtake_t - _passed_t - 3.0 * abs(_to_lane))
    linger_factor = max(1.0 - max(linger, 0.0) / 4.0, 0.3)
    # Megengedett sav: a route savja, vagy elozes kozben a route savja es a
    # celsav kozotti savok.
    allowed_lane = lane_idx == 0 or (_overtaking and lane_idx * _to_lane > 0
                                     and abs(lane_idx) <= abs(_to_lane))

    if not allowed_lane:
        # Rossz savban van (jobbra kiallt, vagy elozes nelkul balra ment).
        state = "Off lane %+d" % lane_idx
        reward = -0.5
    else:
        # Minden megengedett allapotban ugyanaz a keplet, az allapotok csak a
        # HUD-nak, a metrikaknak es az elozes lezarasanak kellenek.
        reward = progress * lane_factor * linger_factor
        if _overtaking:
            # Ha visszaert a route savjanak kozepere (0.5 m-en belul), vege az
            # elozesnek. Ha kozben elhaladt a target mellett, es szaggatott
            # vonalon jott vissza, +5 bonusz (autonkent egyszer).
            state = "Overtaking %s%d" % ("L" if _to_lane > 0 else "R", abs(_to_lane))
            if abs(offset) < 0.5:
                _overtaking = False
                if _passed and _return_ok and _target_id not in _overtaken_ids:
                    _overtaken_ids.add(_target_id)
                    env.overtakes += 1
                    reward += 5.0
            env.overtake_reward += reward
        elif blocked:
            state = "Blocked"
            env.blocked_time += 1.0 / env.fps
        elif can_overtake:
            state = "Can overtake %s%d" % ("L" if to_lane > 0 else "R", abs(to_lane))
        else:
            state = "Free"

    # Kormany-rangatas buntetes: 2 * |a kormany valtozasa ebben a lepesben|.
    steer = env.vehicle.control.steer
    reward -= 2.0 * abs(steer - _prev_steer)
    _prev_steer = steer

    env.extra_info.extend([
        terminal_reason,
        "State:  % 19s" % state,
        "Overtakes:           % 7d" % env.overtakes,
        ""])
    return float(reward)


def reward_fn_5(env):
    # =========================================================================
    # REWARD 5 - a reward 4 (haladas-alapu) + potencial-alapu formalas az
    # elozesre. Az uj reszeket "R5" jeloli, minden mas a reward 4.
    #
    # Ng, Harada, Russell (1999): Policy invariance under reward
    # transformations. A formalo tag minden lepesben
    #     F = gamma * Phi(s') - Phi(s),
    # ahol Phi csak az allapottol fugg. A tetel szerint az ilyen tag nem
    # valtoztatja meg, melyik strategia az optimalis (osszegben kiejti
    # magat), csak elobbre hozza a jutalmat, igy gyorsabban tanul. A reward 3
    # R3-E bonusz-kettebontasa ugyanezt probalja kezzel, itt elmeletileg
    # megalapozottan tortenik. A reward 4 vs reward 5 osszevetes ezt meri.
    #
    # Phi = elozes kozben 2 + 3 * (az elhaladas mekkora resze van meg), azon
    # kivul 0. Az elhaladas resze: (lon a kiallaskor - lon most) /
    # (lon a kiallaskor + 5 m), 0..1. Kiallaskor 2, a target mellett 5 m-rel
    # elhaladva 5, utana 5 marad a visszasorolasig. Igy:
    #   - kiallaskor F = +1.96: a +5 bonusz ertekenek egy resze mar a
    #     kiallaskor megjon, nem csak ~5 s mulva,
    #   - elhaladas kozben minel gyorsabban jon elore a targethez kepest,
    #     annal tobbet kap azonnal (ha lassabb a targetnel, negativ),
    #   - amig Phi > 0, lepesenkent -(1 - gamma) * Phi = -0.04..-0.1 is jar,
    #     igy az elhuzodo elozes azonnal rosszabbnak latszik,
    #   - visszasorolaskor F = -5, ezt a +5 bonusz kiegyenliti,
    #   - feladott elozes (visszasorol elhaladas nelkul) vagy utkozes elozes
    #     kozben visszaveszi, amit addig kapott (-Phi).
    # Mindez nem valtoztat azon, mi az optimalis: a kezdoallapotbol a
    # diszkontalt osszeg pontosan ugyanannyi, mint a reward 4-nel, csak a
    # jutalom jon elobb. A gamma itt 0.98, egyezzen a SAC gamma-javal (config).
    #
    # FIGYELEM: a diszkontalatlan osszeg (TensorBoard total_reward,
    # mean_reward) elozesenkent ~10-zel kisebb, mint a reward 4-nel (a
    # -(1 - gamma) * Phi tagok miatt). A futasokat ne ezekkel hasonlitsd ossze,
    # hanem a feladat-metrikakkal (elozes/km, utkozes, sebesseg).
    # =========================================================================
    global _low_speed_timer, _overtaking, _to_lane, _target_id, _passed, _return_ok
    global _overtaken_ids, _overtake_t, _passed_t, _prev_steer, _prev_phi, _lon0

    terminal_reason = "Running..."
    speed_kmh = env.vehicle.get_speed()
    # A forgalom az egohoz kepest. A step() lepesenkent egyszer szamolja
    # (get_vehicle_surroundings): lead auto, szomszed savok, vonalak, lon/auto.
    surr = env.surroundings
    # Epizod eleje: az elozesi allapotgep minden valtozoja nullazodik.
    if env.step_count == 0:
        _low_speed_timer = 0.0
        _overtaking, _to_lane, _target_id, _passed, _return_ok = False, 0, None, False, False
        _overtaken_ids = set()
        _overtake_t, _passed_t = 0.0, None
        _prev_steer = env.vehicle.control.steer
        _prev_phi = 0.0

    # --- 1) Hol vagyok a route savjahoz kepest ------------------------------
    # offset: elojeles tavolsag a route savjanak kozepetol [m], + balra,
    # - jobbra. A distance_from_center elojel nelkuli, az oldalt a waypoint
    # jobb vektorara vetitett helyzet adja meg.
    wp = env.current_waypoint.transform
    loc = env.vehicle.get_transform().location
    r = wp.get_right_vector()
    side = (loc.x - wp.location.x) * r.x + (loc.y - wp.location.y) * r.y
    offset = env.distance_from_center if side < 0.0 else -env.distance_from_center

    lane_w = surr["lane_width"]

    # Hanyadik savban vagyok a route savjahoz kepest: 0 = a sajat, +1 = eggyel
    # balra, -1 = eggyel jobbra.
    lane_idx = int(round(offset / lane_w))

    # --- 2) Elozhetek-e most ------------------------------------------------
    # has_lead: van auto a sajat savomban elottem, 40 m-en belul.
    # to_lane: hova allhatok ki elozni. Csak balra: +1, vagy ha az foglalt, +2.
    # Feltetel: minden atlepett vonal szaggatott ("broken"), a celsav szabad
    # (-8..+20 m-en nincs benne auto), +2-nel a +1-es savon at lehet vagni
    # ("cross": mellettem +-8 m-en nincs benne senki).
    has_lead = surr["front_id"] is not None
    to_lane = None
    if has_lead:
        lanes = surr["lanes"]
        for n in (1, 2):
            lane = lanes.get(n)
            if lane and lane["broken"] and lane["free"] and (abs(n) == 1 or lanes[n // 2]["cross"]):
                to_lane = n
                break
    can_overtake = to_lane is not None
    # blocked: van elottem auto, de nem elozhetem meg, kovetni kell.
    blocked = has_lead and not can_overtake

    # --- 3) Elozesi allapotgep ----------------------------------------------
    # Kiallas: elozhetek, meg a sajat savomban vagyok, es mar 1 m-nel tobbet
    # mozdultam a celsav fele. Ekkor rogzul a celsav (_to_lane) es a
    # megelozendo auto (_target_id), az elozes a visszaerkezesig tart.
    if (not _overtaking and can_overtake and lane_idx == 0
            and offset * to_lane > 0.0 and abs(offset) > 1.0):
        _overtaking, _to_lane, _target_id, _passed, _return_ok = \
            True, to_lane, surr["front_id"], False, False
        _overtake_t, _passed_t = 0.0, None
        # R5: a target tavolsaga a kiallaskor, ehhez merjuk az elhaladast.
        _lon0 = surr["front_dist"]
    if _overtaking:
        _overtake_t += 1.0 / env.fps
        # A target hosszanti helyzete az egohoz kepest [m], + elottem.
        lon = surr["lon"].get(_target_id)
        # Elhaladtam mellette: a target mar 5 m-rel mogottem van.
        if not _passed and lon is not None and lon < -5.0:
            _passed, _passed_t = True, _overtake_t
        # Amig nem a route savjaban vagyok, figyelem a visszafele atlependo
        # vonalat: csak szaggatotton at szabad visszasorolni (a bonuszhoz kell).
        if abs(offset) > lane_w / 2.0:
            _return_ok = surr["right_marking" if _to_lane > 0 else "left_marking"] == "Broken"

    # --- 4) Epizod-vege feltetelek ------------------------------------------
    # Megallt (5 s-ig 1 km/h alatt), lement az utrol (2 m-nel messzebb
    # barmelyik Driving sav kozepetol), vagy 60 km/h felett megy.
    if not env.terminal_state:
        _low_speed_timer = _low_speed_timer + 1.0 / env.fps if speed_kmh < 1.0 else 0.0
        # A legkozelebbi Driving sav kozepe (barmelyik sav, barmelyik irany).
        near = env.map.get_waypoint(loc).transform.location

        if _low_speed_timer > 5.0:
            env.terminal_state = True
            terminal_reason = "Vehicle stopped"
        elif np.hypot(loc.x - near.x, loc.y - near.y) > 2.0:
            env.terminal_state = True
            terminal_reason = "Off-track"
        elif speed_kmh > 60.0:
            env.terminal_state = True
            terminal_reason = "Too fast"

    # Terminal lepes: jarmuvel utkozes -30, minden mas -10.
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
        # R5: a terminal allapot potencialja 0, igy itt F = -Phi(s). Ha elozes
        # kozben er veget (pl. utkozes), visszaveszi az addig kapott formalast.
        if env.collision_with is not None and env.collision_with.startswith("vehicle."):
            return -30 - _prev_phi
        return -10 - _prev_phi

    # --- 5) Jutalom ---------------------------------------------------------
    # speed_factor: 0..1, 50 km/h-nal 1. Alatta linearis, 50-60 km/h kozott
    # 0-ra esik (60 felett "Too fast" terminal).
    if speed_kmh <= 50.0:
        speed_factor = speed_kmh / 50.0
    else:
        speed_factor = 1.0 - (speed_kmh - 50.0) / 10.0
    speed_factor = float(np.clip(speed_factor, 0.0, 1.0))
    # Haladas: a route iranyu sebesseg 50 km/h-ra normalva. A cos(szog) a
    # route-hoz kepest ferden menest veszi le (savvaltasnal ~5-10 fok, ez csak
    # ~1%), visszafele menve negativ. Linearis a sebessegben: 25 km/h fele
    # annyi haladas, mint 50 km/h.
    progress = speed_factor * np.cos(env.vehicle.get_angle(env.current_waypoint))
    # lane_factor: az egyetlen savtartasi tag. Milyen messze vagyok a
    # legkozelebbi megengedett sav kozepetol: ha elozhetek vagy elozok, a route
    # savja es a celsav kozti barmelyik sav jo, kulonben csak a route savja.
    # Sav kozepen 1.0, ket sav hataran (1.75 m-re a kozeptol) 0.7.
    goal = _to_lane if _overtaking else (to_lane or 0)
    step = 1 if goal >= 0 else -1
    lane_d = min(abs(offset - n * lane_w) for n in range(0, goal + step, step))
    lane_factor = 1.0 - 0.3 * min(lane_d / (lane_w / 2.0), 1.0)
    # linger_factor (elozeshez): ne ragadjon a belso savban (jobbra tartas).
    # 15 s elozes utan, vagy a target melletti elhaladas utan savonkent 3 s
    # utan 4 s alatt 1.0-rol 0.3-ra esik. Elozesen kivul 1.0.
    linger = 0.0
    if _overtaking:
        linger = _overtake_t - 15.0
        if _passed:
            linger = max(linger, _overtake_t - _passed_t - 3.0 * abs(_to_lane))
    linger_factor = max(1.0 - max(linger, 0.0) / 4.0, 0.3)
    # Megengedett sav: a route savja, vagy elozes kozben a route savja es a
    # celsav kozotti savok.
    allowed_lane = lane_idx == 0 or (_overtaking and lane_idx * _to_lane > 0
                                     and abs(lane_idx) <= abs(_to_lane))

    if not allowed_lane:
        # Rossz savban van (jobbra kiallt, vagy elozes nelkul balra ment).
        state = "Off lane %+d" % lane_idx
        reward = -0.5
    else:
        # Minden megengedett allapotban ugyanaz a keplet, az allapotok csak a
        # HUD-nak, a metrikaknak es az elozes lezarasanak kellenek.
        reward = progress * lane_factor * linger_factor
        if _overtaking:
            # Ha visszaert a route savjanak kozepere (0.5 m-en belul), vege az
            # elozesnek. Ha kozben elhaladt a target mellett, es szaggatott
            # vonalon jott vissza, +5 bonusz (autonkent egyszer).
            state = "Overtaking %s%d" % ("L" if _to_lane > 0 else "R", abs(_to_lane))
            if abs(offset) < 0.5:
                _overtaking = False
                if _passed and _return_ok and _target_id not in _overtaken_ids:
                    _overtaken_ids.add(_target_id)
                    env.overtakes += 1
                    reward += 5.0
            env.overtake_reward += reward
        elif blocked:
            state = "Blocked"
            env.blocked_time += 1.0 / env.fps
        elif can_overtake:
            state = "Can overtake %s%d" % ("L" if to_lane > 0 else "R", abs(to_lane))
        else:
            state = "Free"

    # R5: Phi(s') a lepes utani allapotban (az elozes itt mar lezarulhatott),
    # es a formalo tag F = 0.98 * Phi(s') - Phi(s).
    phi = 0.0
    if _overtaking:
        frac = 1.0
        if not _passed:
            lon = surr["lon"].get(_target_id)
            frac = 0.0 if lon is None else float(np.clip((_lon0 - lon) / (_lon0 + 5.0), 0.0, 1.0))
        phi = 2.0 + 3.0 * frac
    reward += 0.98 * phi - _prev_phi
    _prev_phi = phi

    # Kormany-rangatas buntetes: 2 * |a kormany valtozasa ebben a lepesben|.
    steer = env.vehicle.control.steer
    reward -= 2.0 * abs(steer - _prev_steer)
    _prev_steer = steer

    env.extra_info.extend([
        terminal_reason,
        "State:  % 19s" % state,
        "Overtakes:           % 7d" % env.overtakes,
        ""])
    return float(reward)
