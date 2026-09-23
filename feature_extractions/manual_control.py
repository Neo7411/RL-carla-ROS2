"""
Kezi vezetes: billentyuzet VAGY PS5 (DualSense) kontroller.

Miert kell: az autopilot rendszeresen koccan, es a karambol utani kepek
hasznalhatatlan tanitoadatot adnak (az auto all, a kepek szinte azonosak).
Kezi vezetessel te donthetsz arrol, hova menjen az auto, es kikerulheted a
problemas helyzeteket.

Mindket tanito ugyanezt a modult hasznalja (camera/train.py, lidar/train.py),
igy a vezetes pontosan ugyanugy viselkedik mindkettoben.

BILLENTYUZET
    W / FEL      - gaz
    S / LE       - fek
    A / BAL      - balra
    D / JOBB     - jobbra
    SPACE        - kezifek
    Q            - hatramenet be/ki

PS5 KONTROLLER (DualSense)
    R2           - gaz            (analog: minel jobban nyomod, annal jobban)
    L2           - fek            (analog)
    bal analog X - kormany        (analog)
    X (kereszt)  - kezifek
    kor          - hatramenet be/ki
"""

import pygame


# ---------------------------------------------------------------------------
# PS5 DualSense tengely- es gombkiosztas
# ---------------------------------------------------------------------------
# FIGYELEM: ezek a szamok illesztoprogramtol fuggenek. Linuxon a DualSense
# ket kulonbozo modban jelentkezhet be, es a kiosztas elcsuszhat. Ha nem jo
# tengelyre mozdul az auto, inditsd a tanitot es nezd a konzolt: a modul
# kiirja a talalt kontroller nevet es a tengelyek szamat.
#
# Az alabbi ertekek az SDL2 alapertelmezett DualSense terkepezesehez valok.
# Ezt a kiosztast a te DualSense-eden MERTEM (pygame 2.6.1 / SDL 2.28.4,
# 6 tengely, 13 gomb). Nyugalomban a 2. es az 5. tengely all -1.0-n, vagyis
# azok a ravaszok; a 0/1 a bal kar, a 3/4 a jobb kar.
AXIS_STEER = 0          # bal analog, vizszintes
AXIS_L2 = 2             # bal ravaszt (fek)
AXIS_R2 = 5             # jobb ravaszt (gaz)

BUTTON_CROSS = 0        # X - kezifek
BUTTON_CIRCLE = 1       # kor - hatramenet

# A ravaszok NYUGALMI allapotban -1.0-t adnak, teljesen benyomva +1.0-t.
# Ezert kell a (v + 1) / 2 atszamitas, kulonben a nyugalmi allapot -1 lenne,
# vagyis "teljes gaz visszafele".
def _trigger_to_unit(value):
    """Ravasz nyers erteke (-1..1) -> 0..1."""
    return (value + 1.0) * 0.5


# A bal analog kar holtjateka. Enelkul a kar sosem all pontosan 0-n, es az
# auto lassan elhuzna oldalra magatol.
STICK_DEADZONE = 0.12

# Billentyuzetnel a kormany nem ugrik azonnal a szelso allasba, hanem ennyi
# ido alatt (masodperc) er oda. Enelkul a vezetes kapkodo es az auto
# kiszamithatatlanul rangat - a CARLA sajat manual_control.py-ja is igy
# csinalja.
STEER_RAMP_TIME = 0.35
# Elengedes utan ennyi ido alatt all vissza kozepre. Gyorsabb, mint a
# kitekeres: igy az auto magatol kiegyenesedik, ha elengeded a gombot.
STEER_RETURN_TIME = 0.20


class VehicleController(object):
    """
    Billentyuzet + kontroller -> carla.VehicleControl.

    Hasznalat a tanito fo ciklusaban:

        controller = VehicleController()
        ...
        controller.handle_event(event)          # az esemeny ciklusban
        ...
        controller.apply(vehicle, dt)           # minden kepkockaban
    """

    def __init__(self, control_factory=None):
        # A carla modult nem importaljuk a fajl tetejen: igy ez a modul carla
        # nelkul is betolthető (teszthez), az import csak a tenyleges
        # vezerles-keszitesnel tortenik.
        self._control_factory = control_factory

        self.steer = 0.0
        self.reverse = False
        self.handbrake = False

        self.joystick = None
        self._init_joystick()

    # ------------------------------------------------------------------
    # Kontroller
    # ------------------------------------------------------------------

    def _init_joystick(self):
        """Elso csatlakoztatott kontroller megkeresese. Ha nincs, nem baj."""
        if not pygame.get_init():
            pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            print("[control] nincs kontroller - billentyuzetes vezetes "
                  "(W/A/S/D vagy nyilak, SPACE kezifek, Q hatramenet)")
            return

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"[control] kontroller: {self.joystick.get_name()} "
              f"({self.joystick.get_numaxes()} tengely, "
              f"{self.joystick.get_numbuttons()} gomb)")
        print("[control] R2 gaz, L2 fek, bal kar kormany, X kezifek, kor hatramenet")
        self._check_trigger_axes()

    def _check_trigger_axes(self):
        """
        Ellenorzi, hogy a ravaszok tenyleg ott vannak-e, ahol varjuk.

        A ravasz nyugalomban -1.0-t ad, egy analog kar pedig 0 korulit. Ha a
        vart tengelyen 0 korul all az ertek, akkor a kiosztas elcsuszott (mas
        driver, mas kontroller) - ilyenkor szolunk, mert kulonben az auto
        magatol gyorsulna vagy fekezne.
        """
        # A friss ertekekhez kell egy pump: kozvetlenul init utan a SDL meg
        # nem toltotte fel a tengelyeket.
        for _ in range(5):
            pygame.event.pump()

        n = self.joystick.get_numaxes()
        for name, axis in (("R2 (gaz)", AXIS_R2), ("L2 (fek)", AXIS_L2)):
            if axis >= n:
                print(f"[control] FIGYELEM: a {name} tengely ({axis}) nem letezik "
                      f"ezen a kontrolleren ({n} tengely van)")
                continue
            value = self.joystick.get_axis(axis)
            if value > -0.5:
                print(f"[control] FIGYELEM: a {name} tengely ({axis}) nyugalomban "
                      f"{value:+.2f}, pedig -1.00 korul kellene lennie. "
                      f"A kiosztas valoszinuleg elcsuszott - allitsd at az "
                      f"AXIS_* konstansokat a manual_control.py-ban.")

    @property
    def has_joystick(self):
        return self.joystick is not None

    # ------------------------------------------------------------------
    # Esemenyek
    # ------------------------------------------------------------------

    def handle_event(self, event):
        """
        A tanito esemeny ciklusabol hivando.

        Csak a KAPCSOLOKAT kezeli itt (hatramenet), mert azok egyszeri
        esemenyek. A folyamatos jeleket (gaz, fek, kormany) az apply()
        olvassa ki kozvetlenul - egy lenyomva tartott gombra ugyanis nem
        erkezik ismetelt esemeny.
        """
        if event.type == pygame.KEYUP and event.key == pygame.K_q:
            self.reverse = not self.reverse
            return True
        if event.type == pygame.JOYBUTTONDOWN and event.button == BUTTON_CIRCLE:
            self.reverse = not self.reverse
            return True
        # Kontroller menet kozbeni be/kihuzasa
        if event.type in (pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED):
            self.joystick = None
            self._init_joystick()
        return False

    # ------------------------------------------------------------------
    # Vezerles kiszamitasa
    # ------------------------------------------------------------------

    def _read_joystick(self, dt):
        """(throttle, brake, steer, handbrake) a kontrollerrol."""
        throttle = _trigger_to_unit(self.joystick.get_axis(AXIS_R2))
        brake = _trigger_to_unit(self.joystick.get_axis(AXIS_L2))

        steer = self.joystick.get_axis(AXIS_STEER)
        if abs(steer) < STICK_DEADZONE:
            steer = 0.0
        else:
            # A holtjatek utan ujraskalazzuk, hogy a teljes kitekeres
            # megmaradjon: enelkul a maximum csak 1 - deadzone lenne.
            sign = 1.0 if steer > 0 else -1.0
            steer = sign * (abs(steer) - STICK_DEADZONE) / (1.0 - STICK_DEADZONE)

        handbrake = bool(self.joystick.get_button(BUTTON_CROSS))
        return throttle, brake, steer, handbrake

    def _read_keyboard(self, dt):
        """(throttle, brake, steer, handbrake) a billentyuzetrol."""
        keys = pygame.key.get_pressed()

        throttle = 1.0 if (keys[pygame.K_w] or keys[pygame.K_UP]) else 0.0
        brake = 1.0 if (keys[pygame.K_s] or keys[pygame.K_DOWN]) else 0.0

        left = keys[pygame.K_a] or keys[pygame.K_LEFT]
        right = keys[pygame.K_d] or keys[pygame.K_RIGHT]

        # Fokozatos kormanyzas (lasd STEER_RAMP_TIME). A billentyu nem ad
        # analog erteket, ezert az idovel epitjuk fel.
        if left and not right:
            self.steer = max(-1.0, self.steer - dt / STEER_RAMP_TIME)
        elif right and not left:
            self.steer = min(1.0, self.steer + dt / STEER_RAMP_TIME)
        else:
            # Elengedve visszaall kozepre.
            back = dt / STEER_RETURN_TIME
            if abs(self.steer) <= back:
                self.steer = 0.0
            else:
                self.steer -= back if self.steer > 0 else -back

        handbrake = keys[pygame.K_SPACE]
        return throttle, brake, self.steer, handbrake

    def apply(self, vehicle, dt):
        """
        Kiolvassa a bemenetet es ratolja az autora.

        vehicle : carla.Vehicle actor (a tanitokban ego.player)
        dt      : eltelt ido masodpercben (a fokozatos kormanyzashoz)

        return: (throttle, brake, steer) a HUD-nak
        """
        if self.joystick is not None:
            throttle, brake, steer, handbrake = self._read_joystick(dt)
            # A kontroller analog kormanyat nem kell epiteni, de eltaroljuk,
            # hogy a ket bemeneti mod kozott valtva ne ugorjon az ertek.
            self.steer = steer
        else:
            throttle, brake, steer, handbrake = self._read_keyboard(dt)

        self.handbrake = handbrake

        control = self._make_control()
        control.throttle = float(throttle)
        control.brake = float(brake)
        control.steer = float(max(-1.0, min(1.0, steer)))
        control.hand_brake = bool(handbrake)
        control.reverse = self.reverse
        vehicle.apply_control(control)

        return throttle, brake, steer

    def _make_control(self):
        if self._control_factory is not None:
            return self._control_factory()
        import carla
        return carla.VehicleControl()

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def hud_lines(self, throttle, brake, steer):
        """A HUD extra_info listajaba illesztheto sorok."""
        mode = "kontroller" if self.joystick is not None else "billentyuzet"
        return [
            "",
            "Kezi vezetes (%s)" % mode,
            "Gaz:          % 11.2f" % throttle,
            "Fek:          % 11.2f" % brake,
            "Kormany:      % 11.2f" % steer,
            "Hatramenet:   % 11s" % ("IGEN" if self.reverse else "nem"),
            "Kezifek:      % 11s" % ("IGEN" if self.handbrake else "nem"),
        ]
