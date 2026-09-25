import numpy as np

# --- Beallitasok (itt szerkesztheto minden) --------------------------------
EARLY_STOP = True             # korai epizod-vege engedelyezve?
MAX_SPEED = 60.0              # ezen felul terminal [km/h]
TARGET_SPEED = 50.0           # ideal sebesseg [km/h]
SPEED_WEIGHT = 1.5            # a sebesseg sulya (1.0 = nincs sulyozas)
MAX_DISTANCE = 2.0            # max eltres a sav kozeptol [m]
MAX_STD_CENTER_LANE = 0.35    # max szoras a kozeptol valo tavolsagban [m]
MAX_ANGLE_CENTER_LANE = 90    # max szogeltres [fok]
PENALTY_REWARD = -10          # terminal buntetes
STOP_TIMEOUT = 5.0            # ennyi mp allas utan terminal [s]

# --- Forgalom, balrol elozes (csak reward_fn) ------------------------------
FOLLOW_GAP_S = 2.0            # kovetesi tavolsag, ha nem elozhet [s]
OVERTAKE_START_M = 1.0        # ennyivel balra a route-tol: kiallt elozni [m]
OVERTAKE_MIN = 0.3            # elozes kozben a minimalis jutalom / lepes...
OVERTAKE_SPEED = 0.6          # ...plusz ennyi * speed_factor ** SPEED_WEIGHT
                              # (25 km/h: 0.51 > kovetes 0.35; 50 km/h: 0.9 < sajat sav 1.0)
OVERTAKE_BONUS = 5.0          # egyszeri jutalom kesz elozesert
PASS_BEHIND_M = 5.0           # a megelozott auto ennyivel mogem kerult [m]
BACK_IN_LANE_M = 0.5          # a route-tol ez alatt: visszaert a savjaba [m]
COLLISION_PENALTY = -30       # jarmuvel utkozes (terminal)

# Mennyi ideje all EGYHUZAMBAN az auto [s]. Elindulaskor es minden epizod
# elejen nullazodik.
_low_speed_timer = 0.0

# reward_fn allapota, epizodonkent nullazodik:
_overtaking = False       # kiallt elozni, es meg nem ert vissza a route savjaba
_target_id = None         # akit eloz
_passed = False           # a _target_id mar PASS_BEHIND_M-rel mogotte van
_return_ok = False        # a bal savbol nezve jobbra szaggatott (szabad visszasorolni)
_overtaken_ids = set()    # akiert mar jart bonusz
_smooth_hold = 0          # visszasorolas utan ennyi lepesig nincs cikkcakk-buntetes


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
    """Sav-kozep tartas + balrol elozes. A forgalmat az env.surroundings adja
    (get_vehicle_surroundings, lepesenkent egyszer).

      - Nincs elottem senki: mint a reward_fn_og, 2 m-en tul Off-track.
      - Elottem van valaki, balra szaggatott es a bal sav szabad: kiallhat,
        a sav elhagyasa balra csak ekkor nem Off-track. Elozes kozben
        OVERTAKE_MIN + OVERTAKE_SPEED * sebesseg - kevesebb, mint a sajat
        savban, de tobb, mint lassan kovetni.
      - Megelozte (PASS_BEHIND_M-rel mogotte van), es jobbra szaggatott
        vonalon sorolt vissza a route savjaba: +OVERTAKE_BONUS.
      - Elottem van valaki, de nem elozhet: nincs sebessegbuntetes, csak a
        FOLLOW_GAP_S kovetesi tavolsagot kell tartani (fek nincs, gazelvetellel).
    """
    global _low_speed_timer, _overtaking, _target_id, _passed, _return_ok
    global _overtaken_ids, _smooth_hold

    terminal_reason = "Running..."
    speed_kmh = env.vehicle.get_speed()
    surr = env.surroundings
    if env.step_count == 0:
        _low_speed_timer = 0.0
        _overtaking, _target_id, _passed, _return_ok = False, None, False, False
        _overtaken_ids = set()
        _smooth_hold = 0

    # Elojeles tavolsag a route-tol [m]: + = balra, - = jobbra. A nagysaga a
    # distance_from_center, az elojele a route waypoint jobb vektorabol.
    wp = env.current_waypoint.transform
    loc = env.vehicle.get_transform().location
    r = wp.get_right_vector()
    side = (loc.x - wp.location.x) * r.x + (loc.y - wp.location.y) * r.y
    offset = env.distance_from_center if side < 0.0 else -env.distance_from_center

    lane_w = surr["lane_width"]
    has_lead = surr["front_id"] is not None
    can_overtake = has_lead and surr["left_marking"] == "Broken" and surr["left_free"]
    blocked = has_lead and not can_overtake

    # --- 1) Elozes allapot -------------------------------------------------
    # Kiallt: balra hagyja el a savot, mikozben elozhet. Innentol addig tart,
    # amig vissza nem er a route savjaba (a sav elhagyasa kozben az env mar a
    # bal savot latja sajatjanak, ezert kell megjegyezni).
    if not _overtaking and can_overtake and offset > OVERTAKE_START_M:
        _overtaking, _target_id, _passed, _return_ok = True, surr["front_id"], False, False
    if _overtaking:
        lon = surr["lon"].get(_target_id)
        if lon is not None and lon < -PASS_BEHIND_M:
            _passed = True
        # A bal savban (kozeppel a savhataron tul) a jobb oldali felfestes
        # az, amin vissza kell sorolni.
        if offset > lane_w / 2.0:
            _return_ok = surr["right_marking"] == "Broken"

    # --- 2) Epizod-vege feltetelek -----------------------------------------
    if EARLY_STOP and not env.terminal_state:
        _low_speed_timer = _low_speed_timer + 1.0 / env.fps if speed_kmh < 1.0 else 0.0

        if _low_speed_timer > STOP_TIMEOUT:
            env.terminal_state = True
            terminal_reason = "Vehicle stopped"

        # Elozes kozben a bal sav is ervenyes: balra egy savval tagabb a hatar.
        elif offset < -MAX_DISTANCE or offset > MAX_DISTANCE + (lane_w if _overtaking else 0.0):
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

    if _overtaking:
        state = "Overtaking"
        reward = OVERTAKE_MIN + OVERTAKE_SPEED * speed_factor ** SPEED_WEIGHT
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

        if blocked:
            # Nem elozhet: nincs sebessegbuntetes, csak a kovetesi tavolsagot
            # kell tartani - kozelebb aranyosan kevesebb.
            state = "Blocked"
            need = FOLLOW_GAP_S * speed_kmh / 3.6
            speed_term = min(surr["front_dist"] / (need + 1e-6), 1.0)
            env.blocked_time += 1.0 / env.fps
        else:
            state = "Can overtake" if can_overtake else "Free"
            speed_term = speed_factor ** SPEED_WEIGHT
        reward = speed_term * centering_factor * next_angle_factor * smoothness_factor

        # Elkezdett kiallni (elozhet, es balra tart): 0 m-en a kovetes, 1 m-en
        # mar a teljes elozesi jutalom jar, kozte linearisan - nincs "volgy".
        # Korabban itt a centering es a smoothness (a kiallast cikkcakknak
        # latta) 1 m-ig 0.35-rol 0.004-re vitte le a jutalmat.
        if can_overtake and offset > 0.0:
            k = min(offset / OVERTAKE_START_M, 1.0)
            reward = (1 - k) * speed_term * next_angle_factor \
                + k * (OVERTAKE_MIN + OVERTAKE_SPEED * speed_factor ** SPEED_WEIGHT)

    env.extra_info.extend([
        terminal_reason,
        "State:  % 19s" % state,
        "Overtakes:           % 7d" % env.overtakes,
        ""])
    return float(reward)
