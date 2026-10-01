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


def reward_fn_og(env):
    global _low_speed_timer

    terminal_reason = "Running..."
    speed_kmh = env.vehicle.get_speed()
    if env.step_count == 0:
        _low_speed_timer = 0.0

    # --- 1) Epizod-vege feltetelek -----------------------------------------
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

    surr = env.get_vehicle_surroundings()

    if speed_kmh <= 50.0:
        speed_factor = speed_kmh / 50.0
    else:
        speed_factor = 1.0 - (speed_kmh - 50.0) / 10.0
    speed_factor = float(np.clip(speed_factor, 0.0, 1.0))

    centering_factor = max(1.0 - env.distance_from_center / 2.0, 0.0)


    next_angle = env.vehicle.get_angle(env.current_waypoint)
    next_angle_factor = max(1.0 - abs(next_angle) / np.deg2rad(90), 0.0)

    std = np.std(env.distance_from_center_history)
    smoothness_factor = max(1.0 - abs(std) / 0.35, 0.0)

    env.extra_info.extend([terminal_reason, ""])


    return (speed_factor ** 1.9) * centering_factor * next_angle_factor * smoothness_factor


def reward_fn_1(env):
    global _low_speed_timer, _overtaking, _to_lane, _target_id, _passed, _return_ok
    global _overtaken_ids, _smooth_hold, _overtake_t, _passed_t, _prev_steer, _lag_t

    terminal_reason = "Running..."
    speed_kmh = env.vehicle.get_speed()
    surr = env.surroundings
    if env.step_count == 0:
        _low_speed_timer = 0.0
        _overtaking, _to_lane, _target_id, _passed, _return_ok = False, 0, None, False, False
        _overtaken_ids = set()
        _smooth_hold = 0
        _overtake_t, _passed_t = 0.0, None
        _prev_steer = env.vehicle.control.steer
        _lag_t = 0.0
    wp = env.current_waypoint.transform
    loc = env.vehicle.get_transform().location
    r = wp.get_right_vector()
    side = (loc.x - wp.location.x) * r.x + (loc.y - wp.location.y) * r.y
    offset = env.distance_from_center if side < 0.0 else -env.distance_from_center

    lane_w = surr["lane_width"]

    lane_idx = int(round(offset / lane_w))

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
    blocked = has_lead and not can_overtake

    if (not _overtaking and can_overtake and lane_idx == 0
            and offset * to_lane > 0.0 and abs(offset) > 1.0):
        _overtaking, _to_lane, _target_id, _passed, _return_ok = \
            True, to_lane, surr["front_id"], False, False
        _overtake_t, _passed_t = 0.0, None
    if _overtaking:
        _overtake_t += 1.0 / env.fps
        lon = surr["lon"].get(_target_id)
        if not _passed and lon is not None and lon < -5.0:
            _passed, _passed_t = True, _overtake_t
        if abs(offset) > lane_w / 2.0:
            _return_ok = surr["right_marking" if _to_lane > 0 else "left_marking"] == "Broken"

    # --- 2) Epizod-vege feltetelek -----------------------------------------
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

    # --- 3) Jutalom --------------------------------------------------------
    if speed_kmh <= 50.0:
        speed_factor = speed_kmh / 50.0
    else:
        speed_factor = 1.0 - (speed_kmh - 50.0) / 10.0
    speed_factor = float(np.clip(speed_factor, 0.0, 1.0))
    goal = _to_lane if _overtaking else (to_lane or 0)
    step = 1 if goal >= 0 else -1
    lane_d = min(abs(offset - n * lane_w) for n in range(0, goal + step, step))
    lane_factor = 1.0 - 0.3 * min(lane_d / (lane_w / 2.0), 1.0)
    linger = 0.0
    if _overtaking:
        linger = _overtake_t - 15.0
        if _passed:
            linger = max(linger, _overtake_t - _passed_t - 3.0 * abs(_to_lane))
    linger_factor = max(1.0 - max(linger, 0.0) / 4.0, 0.3)
    overtake_term = (0.3 + 0.6 * speed_factor ** 1.9) * lane_factor * linger_factor
    allowed_lane = lane_idx == 0 or (_overtaking and lane_idx * _to_lane > 0
                                     and abs(lane_idx) <= abs(_to_lane))

    lagging = False
    if not allowed_lane:
        state = "Off lane %+d" % lane_idx
        reward = -0.5
    elif _overtaking:
        state = "Overtaking %s%d" % ("L" if _to_lane > 0 else "R", abs(_to_lane))
        reward = overtake_term
        if abs(offset) < 0.5:
            _overtaking = False
            _smooth_hold = env.distance_from_center_history.maxlen
            if _passed and _return_ok and _target_id not in _overtaken_ids:
                _overtaken_ids.add(_target_id)
                env.overtakes += 1
                reward += 5.0
        env.overtake_reward += reward
    else:
        centering_factor = max(1.0 - env.distance_from_center / 2.0, 0.0)

        next_angle = env.vehicle.get_angle(env.current_waypoint)
        next_angle_factor = max(1.0 - abs(next_angle) / np.deg2rad(90), 0.0)
        if _smooth_hold > 0:
            _smooth_hold -= 1
            smoothness_factor = 1.0
        else:
            std = np.std(env.distance_from_center_history)
            smoothness_factor = max(1.0 - abs(std) / 0.35, 0.0)

        ttc_term = 0.0
        if blocked:
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
            state = "Free"
            speed_term = speed_factor ** 1.9
        reward = speed_term * centering_factor * next_angle_factor * smoothness_factor
        if lagging:
            reward = speed_term
        reward += ttc_term
        if can_overtake and offset * to_lane > 0.0:
            k = min(abs(offset) / 1.0, 1.0)
            reward = (1 - k) * speed_term * next_angle_factor + k * overtake_term
    _lag_t = _lag_t + 1.0 / env.fps if lagging else 0.0

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
    global _low_speed_timer, _overtaking, _to_lane, _target_id, _passed, _return_ok
    global _overtaken_ids, _smooth_hold, _overtake_t, _passed_t, _prev_steer, _lag_t

    terminal_reason = "Running..."
    speed_kmh = env.vehicle.get_speed()
    surr = env.surroundings
    if env.step_count == 0:
        _low_speed_timer = 0.0
        _overtaking, _to_lane, _target_id, _passed, _return_ok = False, 0, None, False, False
        _overtaken_ids = set()
        _smooth_hold = 0
        _overtake_t, _passed_t = 0.0, None
        _prev_steer = env.vehicle.control.steer
        _lag_t = 0.0
    wp = env.current_waypoint.transform
    loc = env.vehicle.get_transform().location
    r = wp.get_right_vector()
    side = (loc.x - wp.location.x) * r.x + (loc.y - wp.location.y) * r.y
    offset = env.distance_from_center if side < 0.0 else -env.distance_from_center

    lane_w = surr["lane_width"]

    lane_idx = int(round(offset / lane_w))

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
    blocked = has_lead and not can_overtake

    if (not _overtaking and can_overtake and lane_idx == 0
            and offset * to_lane > 0.0 and abs(offset) > 1.0):
        _overtaking, _to_lane, _target_id, _passed, _return_ok = \
            True, to_lane, surr["front_id"], False, False
        _overtake_t, _passed_t = 0.0, None
    if _overtaking:
        _overtake_t += 1.0 / env.fps
        lon = surr["lon"].get(_target_id)
        if not _passed and lon is not None and lon < -5.0:
            _passed, _passed_t = True, _overtake_t
        if abs(offset) > lane_w / 2.0:
            _return_ok = surr["right_marking" if _to_lane > 0 else "left_marking"] == "Broken"

    # --- 2) Epizod-vege feltetelek -----------------------------------------
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

    # --- 3) Jutalom --------------------------------------------------------
    if speed_kmh <= 50.0:
        speed_factor = speed_kmh / 50.0
    else:
        speed_factor = 1.0 - (speed_kmh - 50.0) / 10.0
    speed_factor = float(np.clip(speed_factor, 0.0, 1.0))
    goal = _to_lane if _overtaking else (to_lane or 0)
    step = 1 if goal >= 0 else -1
    lane_d = min(abs(offset - n * lane_w) for n in range(0, goal + step, step))
    lane_factor = 1.0 - 0.3 * min(lane_d / (lane_w / 2.0), 1.0)
    linger = 0.0
    if _overtaking:
        linger = _overtake_t - 15.0
        if _passed:
            linger = max(linger, _overtake_t - _passed_t - 3.0 * abs(_to_lane))
    linger_factor = max(1.0 - max(linger, 0.0) / 4.0, 0.3)
    overtake_term = (0.3 + 0.6 * speed_factor ** 1.9) * lane_factor * linger_factor
    allowed_lane = lane_idx == 0 or (_overtaking and lane_idx * _to_lane > 0
                                     and abs(lane_idx) <= abs(_to_lane))

    lagging = False
    if not allowed_lane:
        state = "Off lane %+d" % lane_idx
        reward = -0.5
    elif _overtaking:
        state = "Overtaking %s%d" % ("L" if _to_lane > 0 else "R", abs(_to_lane))
        reward = overtake_term
        if abs(offset) < 0.5:
            _overtaking = False
            _smooth_hold = env.distance_from_center_history.maxlen
            if _passed and _return_ok and _target_id not in _overtaken_ids:
                _overtaken_ids.add(_target_id)
                env.overtakes += 1
                reward += 5.0
        env.overtake_reward += reward
    else:
        centering_factor = max(1.0 - env.distance_from_center / 2.0, 0.0)

        next_angle = env.vehicle.get_angle(env.current_waypoint)
        next_angle_factor = max(1.0 - abs(next_angle) / np.deg2rad(90), 0.0)
        if _smooth_hold > 0:
            _smooth_hold -= 1
            smoothness_factor = 1.0
        else:
            std = np.std(env.distance_from_center_history)
            smoothness_factor = max(1.0 - abs(std) / 0.35, 0.0)

        ttc_term = 0.0
        if blocked:
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
            state = "Free"
            speed_term = speed_factor ** 1.9
        reward = speed_term * centering_factor * next_angle_factor * smoothness_factor
        if lagging:
            reward = speed_term
        reward += ttc_term
        if can_overtake and offset * to_lane > 0.0:
            k = min(abs(offset) / 1.0, 1.0)
            # A: a TTC-buntetes a kihuzodas elso 1 m-en fogy el.
            reward = (1 - k) * (speed_term * next_angle_factor + ttc_term) + k * overtake_term
    _lag_t = _lag_t + 1.0 / env.fps if lagging else 0.0

    steer = env.vehicle.control.steer
    reward -= 2.0 * abs(steer - _prev_steer)
    _prev_steer = steer

    env.extra_info.extend([
        terminal_reason,
        "State:  % 19s" % state,
        "Overtakes:           % 7d" % env.overtakes,
        ""])
    return float(reward)
