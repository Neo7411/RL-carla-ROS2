"""
Egyetlen, osszevont jutalomfuggveny a CARLA sav-koveteshez.

Hasznalat a train.py-ban - csak ennyi kell, semmi mas:
    from carla_env.reward import reward_fn
    ...
    env = CarlaRouteEnv(..., reward_fn=reward_fn, ...)

Minden beallitas itt lent van, a config.py-t nem kell hozzanyulni.
"""

import numpy as np

# --- Beallitasok (itt szerkesztheto minden) --------------------------------
EARLY_STOP = True             # korai epizod-vege engedelyezve?
MIN_SPEED = 20.0              # ez alatt aranyosan buntetunk [km/h]
MAX_SPEED = 70.0              # ezen felul terminal [km/h]
TARGET_SPEED = 50.0           # ideal sebesseg [km/h]
MAX_DISTANCE = 2.0            # max eltres a sav kozeptol [m]
MAX_STD_CENTER_LANE = 0.35    # max szoras a kozeptol valo tavolsagban [m]
MAX_ANGLE_CENTER_LANE = 90    # max szogeltres [fok]
PENALTY_REWARD = -10          # terminal buntetes
STOP_TIMEOUT = 5.0            # ennyi mp allas utan terminal [s]

# Alacsony sebesseggel toltott ido szamlaloja (epizodonkent nullazodik).
_low_speed_timer = 0.0


def reward_fn(env):
    """Egy lepes jutalma. A CarlaRouteEnv adja at magat (env) argumentumkent.

    Ket resze van:
      1) Terminal-ellenorzes: megall / lesodrodik / tulgyorsul -> PENALTY_REWARD.
      2) Kulonben a jutalom negy, [0, 1] koze normalt tenyezo szorzata,
         igy barmelyik "elrontasa" lehuzza a teljes jutalmat.
    """
    global _low_speed_timer

    terminal_reason = "Running..."
    speed_kmh = env.vehicle.get_speed()

    # --- 1) Epizod-vege feltetelek -----------------------------------------
    if EARLY_STOP and not env.terminal_state:
        _low_speed_timer += 1.0 / env.fps

        # Az auto beragadt: STOP_TIMEOUT-nal tovabb allt, mikozben mar elindult volna.
        if _low_speed_timer > STOP_TIMEOUT and speed_kmh < 1.0 and env.current_waypoint_index >= 1:
            env.terminal_state = True
            terminal_reason = "Vehicle stopped"

        # Lesodrodott a sav kozepetol.
        elif env.distance_from_center > MAX_DISTANCE:
            env.terminal_state = True
            terminal_reason = "Off-track"

        # Tul gyors.
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

    # --- 2) Jutalom tenyezok (mind 0..1) -----------------------------------
    # a) Sebesseg: MIN_SPEED alatt linearisan no, MIN..TARGET kozott 1.0,
    #    TARGET folott linearisan csokken MAX_SPEED-ig.
    if speed_kmh < MIN_SPEED:
        speed_factor = speed_kmh / MIN_SPEED
    elif speed_kmh > TARGET_SPEED:
        speed_factor = 1.0 - (speed_kmh - TARGET_SPEED) / (MAX_SPEED - TARGET_SPEED)
    else:
        speed_factor = 1.0
    speed_factor = float(np.clip(speed_factor, 0.0, 1.0))

    # b) Sav-kozepen tartas: minel kozelebb a kozephez, annal jobb.
    centering_factor = max(1.0 - env.distance_from_center / MAX_DISTANCE, 0.0)

    # c) Szogeltres: az auto iranya mennyire egyezik a kovetkezo waypoint iranyaval.
    angle = env.vehicle.get_angle(env.current_waypoint)
    angle_factor = max(1.0 - abs(angle) / np.deg2rad(MAX_ANGLE_CENTER_LANE), 0.0)

    # d) Simasag: a kozeptol valo tavolsag szorasa - a "cikkcakkozast" bunteti.
    std = np.std(env.distance_from_center_history)
    smoothness_factor = max(1.0 - abs(std) / MAX_STD_CENTER_LANE, 0.0)

    env.extra_info.extend([terminal_reason, ""])
    return speed_factor * centering_factor * angle_factor * smoothness_factor
