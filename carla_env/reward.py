import numpy as np

# --- Beallitasok (itt szerkesztheto minden) --------------------------------
EARLY_STOP = True             # korai epizod-vege engedelyezve?
MAX_SPEED = 100.0              # ezen felul terminal [km/h]
TARGET_SPEED = 80.0           # ideal sebesseg [km/h]
SPEED_WEIGHT = 1.5            # a sebesseg sulya (1.0 = nincs sulyozas)
MAX_DISTANCE = 2.0            # a centering_factor skalaja [m] (NEM terminal-hatar)
MAX_DISTANCE_HARD = 6.0       # ezen tul terminal akkor is, ha meg uttesten all [m]
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

        # Lehajtott az uttestrol (fu, jarda, arok). A szaggatott vonal atlepese
        # NEM ez: amig barmelyik Driving savon all, az uttesten van.
        #
        # Korabban a feltetel env.distance_from_center > 2.0 volt. A CARLA sav
        # ~3.5 m szeles, tehat a savhatar 1.75 m-nel van - a 2.0 m-es hatar
        # gyakorlatilag a szaggatott vonal atlepeset buntette -10-zel, miközben
        # a centering_factor ott mar ugyis 0.12-re esett. Az agensnek igy a
        # savvaltas dupla buntetes volt, pedig fizikailag semmi baj nem tortent.
        elif not env.on_driving_lane:
            env.terminal_state = True
            terminal_reason = "Off-road"

        # Szembemenes a forgalommal - ez valodi hiba, uttesten is.
        elif env.wrong_way:
            env.terminal_state = True
            terminal_reason = "Wrong way"

        # Vegso biztonsagi halo: az uttest sajat magaban nagy lehet
        # (kereszetezodes, parkolo), ezert a route-tol valo nagyon nagy
        # eltavolodas akkor is terminal, ha technikailag uton vagyunk.
        elif env.distance_from_center > MAX_DISTANCE_HARD:
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
    # a) Sebesseg: 0-tol TARGET_SPEED-ig VEGIG novekvo, folotte csokken.
    #
    #    A korabbi valtozat MIN_SPEED es TARGET_SPEED kozott lapos 1.0-t adott,
    #    vagyis 21 es 49 km/h kozott NULLA volt a gradiens: az agensnek semmi
    #    nem erte meg gyorsulni. A merés ezt igazolta - 34k lepes utan az
    #    epizodok atlagsebessege 16.5 km/h volt, es csak 36%-uk ment 20 folott.
    #    Lassan menni ugyanis biztonsagosabb (kisebb esely a lesodrodasra es a
    #    -10-es buntetesre), a jutalom-kulonbseg pedig alig 0.2 volt.
    #
    #    Igy viszont minden km/h szamit egeszen a celsebessegig.
    if speed_kmh <= TARGET_SPEED:
        speed_factor = speed_kmh / TARGET_SPEED
    else:
        speed_factor = 1.0 - (speed_kmh - TARGET_SPEED) / (MAX_SPEED - TARGET_SPEED)
    speed_factor = float(np.clip(speed_factor, 0.0, 1.0))

    # b) Sav-kozepen tartas: minel kozelebb a kozephez, annal jobb.
    #
    # A PADLO azert kell, mert a jutalom SZORZAT: nulla centering_factor
    # mellett a sebesseg es minden mas tenyezo is ertelmetlen (0-val szorzunk).
    # A savhataron (1.75 m) a faktor 0.12 volt, vagyis az agens gyakorlatilag
    # semmit nem kapott azert, hogy tovabbhajt - inkabb megallt. A 0.25-os
    # padlo megtartja a savkozep preferenciajat (1.0 vs 0.25 = 4x kulonbseg),
    # de a savvaltas utan is marad ertelme gyorsan haladni.
    centering_factor = max(1.0 - env.distance_from_center / MAX_DISTANCE, 0.25)

    # c) Szogeltres: az auto iranya mennyire egyezik a kovetkezo waypoint iranyaval.
    angle = env.vehicle.get_angle(env.current_waypoint)
    angle_factor = max(1.0 - abs(angle) / np.deg2rad(MAX_ANGLE_CENTER_LANE), 0.0)

    # d) Simasag: a kozeptol valo tavolsag szorasa - a "cikkcakkozast" bunteti.
    std = np.std(env.distance_from_center_history)
    smoothness_factor = max(1.0 - abs(std) / MAX_STD_CENTER_LANE, 0.0)

    env.extra_info.extend([terminal_reason, ""])

    # A sebesseget sulyozzuk (hatvanyozas: a 0..1 tartomanyban a nagyobb kitevo
    # jobban bunteti a lassusagot, tehat a sebesseg dominansabb lesz).
    #
    # Miert kell: a jutalom SZORZAT, es gyorsabban menni rontja a masik harom
    # tenyezot (nehezebb a sav kozepen maradni, no a szogeltres es a szoras).
    # Sulyozas nelkul ezek epp ~35-40 km/h-nal egyensulyozzak ki a sebesseget,
    # ezert az agens ott allt meg - helyesen, mert a reward szerint az volt az
    # optimum. Modellezve: w=1.0 -> 35 km/h, w=1.5 -> 40, w=2.0 -> 45, w=3.0 -> 50.
    return (speed_factor ** SPEED_WEIGHT) * centering_factor * angle_factor * smoothness_factor
