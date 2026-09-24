import carla
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

# --- Forgalom (csak reward_traffic_fn) --------------------------------------
COLLISION_PENALTY = -30       # jarmuvel utkozes (terminal) - a tobbi terminal PENALTY_REWARD
LEAD_RANGE_M = 40.0           # a sajat savban ennyin belul "van elottem valaki" [m]
LEAD_MIN_M = 5.0              # ennyinel kozelebb a kozelsegi buntetes maximalis [m]
PROXIMITY_PENALTY = 1.0       # a kozelsegi buntetes maximuma lepesenkent
NEAR_LAT_M = 5.5              # oldaliranyban ennyin belul: sajat vagy szomszed sav [m]
PASS_BEHIND_M = 5.0           # ha ennyivel mogem kerult, elhagytam (elozes kesz) [m]
OVERTAKE_REWARD = 3.0         # egyszeri jutalom minden elozesert...
OVERTAKE_SPEED_REWARD = 3.0   # ...plusz ennyi * speed_factor: gyors elozes tobbet er
LANE_LEAVE_REWARD = 0.1       # max lepesenkenti jutalom a sav elhagyasaert, ha van elottem valaki
LANE_WIDTH = 3.5              # [m]
BACK_IN_LANE_M = 0.5          # route_dev ez alatt: visszaert a route savjaba [m]

# Mennyi ideje all EGYHUZAMBAN az auto [s]. Elindulaskor es minden epizod
# elejen nullazodik.
_low_speed_timer = 0.0

# reward_traffic_fn allapota, epizodonkent nullazodik:
#   _ahead      azok a forgalmi autok, amik mar voltak elottem, a kozelemben
#   _passed     akiket mar elhagytam (egy autoert csak egyszer jar jutalom)
#   _maneuver   savot hagytam el forgalom miatt, es meg nem ertem vissza
#   _smooth_hold  ennyi ideig [s] meg nem buntetjuk a cikkcakkot (lasd lent)
_ahead, _passed = set(), set()
_maneuver = False
_smooth_hold = 0.0


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

    # A reset() vegen futo step(None) hivja elsokent (step_count meg 0) - itt
    # kezdodik az uj epizod, az elozo allasideje ne orokolodjon at.
    if env.step_count == 0:
        _low_speed_timer = 0.0

    # --- 1) Epizod-vege feltetelek -----------------------------------------
    if EARLY_STOP and not env.terminal_state:
        # Korabban a szamlalo MINDEN lepesben nott, es csak terminalkor
        # nullazodott: az epizod elso 5 mp-e utan barmilyen pillanatnyi
        # megallas (<1 km/h) azonnal terminalt. Most az egyhuzamban allva
        # toltott idot meri.
        _low_speed_timer = _low_speed_timer + 1.0 / env.fps if speed_kmh < 1.0 else 0.0

        # Az auto beragadt: STOP_TIMEOUT-nal tovabb allt egyhuzamban. Ez az
        # epizod elejen is ervenyes - korabban az "index >= 1" feltetel miatt
        # egy el sem indulo auto epizodja soha nem ert veget.
        if _low_speed_timer > STOP_TIMEOUT:
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


def _traffic_state(env):
    """A forgalom az egohoz kepest, egyetlen get_actors hivassal.

    Vissza: (lead_d, beside, n_passed, can_overtake)
      lead_d    a SAJAT savomban elottem levo legkozelebbi auto tavolsaga [m],
                None, ha LEAD_RANGE_M-en belul nincs ilyen
      beside    van-e auto a sajat vagy a szomszed savban, mellettem vagy
                elottem (elozes kozben ez tartja a "forgalmi" allapotot)
      n_passed  hany autot hagytam el EBBEN a lepesben: elottem volt a
                kozelemben, most mar PASS_BEHIND_M-rel mogottem van
      can_overtake  van-e azonos iranyu szomszed sav, amiben 8 m-rel mogottem
                es 20 m-rel elottem senki sincs (barmelyik oldal)
    """
    me = env.vehicle.get_transform()
    my_wp = env.world.map.get_waypoint(me.location)
    # Az azonos iranyu szomszed savok (bal, jobb) - ezekbe lehet kiloni.
    sides = [n for n in (my_wp.get_left_lane(), my_wp.get_right_lane())
             if n is not None and n.lane_type == carla.LaneType.Driving
             and n.lane_id * my_wp.lane_id > 0]
    if not env.traffic_ids:
        return None, False, 0, bool(sides)
    fwd, right = me.get_forward_vector(), me.get_right_vector()
    lead_d, beside, n_passed = None, False, 0
    blocked = set()   # a sides indexei, amikben van valaki az ablakban
    for actor in env.world.get_actors(env.traffic_ids):
        loc = actor.get_location()
        d = loc - me.location
        lon = d.x * fwd.x + d.y * fwd.y      # + = elottem
        lat = d.x * right.x + d.y * right.y  # + = jobbra
        if abs(lat) > NEAR_LAT_M or lon > LEAD_RANGE_M:
            continue
        if lon > 0.0:
            _ahead.add(actor.id)
        if -PASS_BEHIND_M < lon:
            beside = True
        elif actor.id in _ahead and actor.id not in _passed:
            _passed.add(actor.id)
            n_passed += 1
        # Sajat sav: a savomat lon meterrel elore kovetve (my_wp.next) az auto
        # fel savszelessegen belul van-e. A (road_id, lane_id) osszevetes nem
        # jo: az autopalya szakaszokra bontott, a hataron a road_id valtozik,
        # es az ugyanabban a savban levo auto "eltunt" (offline merve 28 m-nel).
        # A puszta oldaliranyu tavolsag sem: kanyarban 40 m-en tobb meter.
        if 0.0 < lon and (lead_d is None or lon < lead_d):
            lane_ahead = my_wp.next(lon) if lon > 0.1 else [my_wp]
            if any(w.transform.location.distance(loc) < 0.5 * w.lane_width for w in lane_ahead):
                lead_d = lon
        # Szomszed sav foglaltsaga, ugyanigy a savot kovetve (elore vagy
        # hatra). A 0.75 savszelesseg: a savhataron allo auto is foglal.
        if -8.0 <= lon <= 20.0:
            for i, n in enumerate(sides):
                if i in blocked:
                    continue
                pts = n.next(lon) if lon > 0.1 else (n.previous(-lon) if lon < -0.1 else [n])
                if any(w.transform.location.distance(loc) < 0.75 * w.lane_width for w in pts):
                    blocked.add(i)
    return lead_d, beside, n_passed, len(blocked) < len(sides)


def reward_traffic_fn(env):
    """Mint a reward_fn, plusz forgalom: a lassabb autokat nagy sebesseggel,
    barmelyik oldalrol ki kell kerulni.

      - Utkozes jarmuvel: COLLISION_PENALTY (terminal, reset).
      - Ha a sajat savomban elottem van valaki: minel kozelebb, annal nagyobb
        buntetes (LEAD_RANGE_M-tol LEAD_MIN_M-ig negyzetesen PROXIMITY_PENALTY-ig).
      - Minden elhagyott auto: OVERTAKE_REWARD + OVERTAKE_SPEED_REWARD * speed_factor.
      - Forgalmi helyzetben (van elottem valaki, vagy epp mellette megyek el
        a route savjabol kilepve) a savkozep elhagyasa NEM buntetett, sot kis
        jutalom jar erte (LANE_LEAVE_REWARD). Egyebkent minden a regi.
    """
    global _low_speed_timer, _ahead, _passed, _maneuver, _smooth_hold

    terminal_reason = "Running..."
    speed_kmh = env.vehicle.get_speed()

    # A reset() vegen futo step(None) hivja elsokent (step_count meg 0) - itt
    # kezdodik az uj epizod, az elozo allasideje ne orokolodjon at.
    if env.step_count == 0:
        _low_speed_timer = 0.0
        _ahead, _passed = set(), set()
        _maneuver, _smooth_hold = False, 0.0

    # --- 1) Epizod-vege feltetelek -----------------------------------------
    if EARLY_STOP and not env.terminal_state:
        # Korabban a szamlalo MINDEN lepesben nott, es csak terminalkor
        # nullazodott: az epizod elso 5 mp-e utan barmilyen pillanatnyi
        # megallas (<1 km/h) azonnal terminalt. Most az egyhuzamban allva
        # toltott idot meri.
        _low_speed_timer = _low_speed_timer + 1.0 / env.fps if speed_kmh < 1.0 else 0.0

        # Az auto beragadt: STOP_TIMEOUT-nal tovabb allt egyhuzamban. Ez az
        # epizod elejen is ervenyes - korabban az "index >= 1" feltetel miatt
        # egy el sem indulo auto epizodja soha nem ert veget.
        if _low_speed_timer > STOP_TIMEOUT:
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
        # Az utkozest a collision szenzor jelzi (env._on_collision), nem ez a
        # fuggveny - ezert itt nincs meg oka.
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

    # --- 2) Jutalom tenyezok (mind 0..1), mint a reward_fn-ben ----------------
    # a) Sebesseg: 0-tol TARGET_SPEED-ig vegig novekvo, folotte csokken.
    if speed_kmh <= TARGET_SPEED:
        speed_factor = speed_kmh / TARGET_SPEED
    else:
        speed_factor = 1.0 - (speed_kmh - TARGET_SPEED) / (MAX_SPEED - TARGET_SPEED)
    speed_factor = float(np.clip(speed_factor, 0.0, 1.0))

    # --- 3) Forgalom ------------------------------------------------------------
    lead_d, beside, n_passed, can_overtake = _traffic_state(env)

    # Forgalmi helyzet: van elottem valaki a sajat savomban ES van hova
    # kiloni, vagy mar kiloptem a route savjabol, es epp mellette/elotte
    # megyek el valakinek. Ha a szomszed sav foglalt (vagy nincs), nincs
    # forgalmi helyzet: a savelhagyas ugyanugy buntetett, mint maskor - es
    # amint a sav felszabadul, ujra ez az ag lep eletbe.
    traffic = (lead_d is not None and can_overtake) or (beside and env.route_dev > BACK_IN_LANE_M)

    # Manover: a forgalom miatti savelhagyastol addig tart, amig vissza nem
    # erek a route savjaba. A simasagi tenyezo (a kozeptavolsag 30 lepeses
    # szorasa) az egesz manovert - a visszasorolast is - cikkcakknak latna,
    # ezert alatta, es utana meg egy ablaknyi ideig nem buntetjuk.
    dt = 1.0 / env.fps
    if traffic:
        _maneuver = True
    elif _maneuver and env.route_dev < BACK_IN_LANE_M:
        _maneuver = False
        _smooth_hold = env.distance_from_center_history.maxlen * dt
    _smooth_hold = max(_smooth_hold - dt, 0.0)

    # b) Sav-kozepen tartas. Forgalmi helyzetben nincs buntetes, sot a route
    #    savjabol valo kilepesert kis jutalom jar (egy savszelessegnel a max).
    #    Kulonben a regi: minel kozelebb a kozephez, annal jobb (0.25-os padlo).
    if traffic:
        centering_factor = 1.0
        lane_leave = LANE_LEAVE_REWARD * min(env.route_dev / LANE_WIDTH, 1.0)
    else:
        centering_factor = max(1.0 - env.distance_from_center / MAX_DISTANCE, 0.25)
        lane_leave = 0.0

    # c) Szogeltres: az auto iranya mennyire egyezik a kovetkezo waypoint iranyaval.
    angle = env.vehicle.get_angle(env.current_waypoint)
    angle_factor = max(1.0 - abs(angle) / np.deg2rad(MAX_ANGLE_CENTER_LANE), 0.0)

    # d) Simasag: a kozeptol valo tavolsag szorasa - a "cikkcakkozast" bunteti,
    #    de a forgalom miatti manovert nem (lasd fent).
    if _maneuver or _smooth_hold > 0.0:
        smoothness_factor = 1.0
    else:
        std = np.std(env.distance_from_center_history)
        smoothness_factor = max(1.0 - abs(std) / MAX_STD_CENTER_LANE, 0.0)

    # e) Kozelsegi buntetes a sajat savomban elottem levo autora, negyzetesen
    #    no LEAD_MIN_M-ig. Ha lehet elozni, mar LEAD_RANGE_M-tol indul - a vegen
    #    nagyobb, mint amit egy lepes haladasert kaphat, tehat meg kell elozni.
    #    Ha nem lehet, csak 12 m alatt: a biztonsagos kovetes nem buntetett,
    #    csak a raforas.
    proximity = 0.0
    if lead_d is not None:
        start = LEAD_RANGE_M if can_overtake else 12.0
        p = np.clip((start - lead_d) / (start - LEAD_MIN_M), 0.0, 1.0)
        proximity = PROXIMITY_PENALTY * p ** 2

    # f) Elozes: minden elhagyott autoert egyszer, a sebesseggel aranyosan tobb.
    overtake = n_passed * (OVERTAKE_REWARD + OVERTAKE_SPEED_REWARD * speed_factor)

    env.extra_info.extend([
        terminal_reason,
        "Elozesek:            % 7d" % len(_passed),
        "Elottem:          %s" % ("% 7.1f m" % lead_d if lead_d is not None else "      -"),
        "Elozhet:           %s" % ("   igen" if can_overtake else "    nem"),
        "Forgalmi helyzet:  %s" % ("   igen" if traffic else "    nem"),
        ""])

    # A sebesseg tovabbra is a fo tenyezo (SPEED_WEIGHT hatvany, szorzat), a
    # forgalmi tagok hozzaadodnak.
    base = (speed_factor ** SPEED_WEIGHT) * centering_factor * angle_factor * smoothness_factor
    return float(base + lane_leave - proximity + overtake)
