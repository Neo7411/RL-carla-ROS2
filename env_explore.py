import carla
import random
import numpy as np
import pygame

kamera_kep_surface = None


def kamera_callback(image):
    global kamera_kep_surface
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3][:, :, ::-1]
    kamera_kep_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))


def get_vehicle_surroundings(vehicle, radius=30.0):
    """
    Csak a járművet kell átadni. Visszaadja:
      - van-e jármű előtte / mögötte / balra / jobbra (radius méteren belül)
      - a bal és jobb oldali szomszéd sáv típusát
    """
    world = vehicle.get_world()
    carla_map = world.get_map()

    tr = vehicle.get_transform()
    loc = tr.location
    fwd = tr.get_forward_vector()
    right = tr.get_right_vector()

    ego_wp = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Any)
    half_lane = (ego_wp.lane_width / 2.0) if ego_wp else 1.75

    flags = {"front": False, "behind": False, "left": False, "right": False}

    for other in world.get_actors().filter("vehicle.*"):
        if other.id == vehicle.id:
            continue
        o = other.get_location()
        if loc.distance(o) > radius:
            continue

        dx, dy = o.x - loc.x, o.y - loc.y
        lon = dx * fwd.x + dy * fwd.y      # + előre, - hátra
        lat = dx * right.x + dy * right.y  # + jobbra, - balra

        if abs(lat) < half_lane:           # a saját sávomban van
            flags["front" if lon > 0 else "behind"] = True
        else:
            flags["right" if lat > 0 else "left"] = True

    # --- Bal és jobb oldali felfestés (szaggatott / folytonos) ---
    left_marking = str(ego_wp.left_lane_marking.type) if ego_wp else None
    right_marking = str(ego_wp.right_lane_marking.type) if ego_wp else None

    return {"vehicles": flags, "left_marking": left_marking, "right_marking": right_marking}


def main():
    actor_lista = []
    pygame.init()
    W, H = 800, 600
    display = pygame.display.set_mode((W, H), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA PyGame HUD")
    font = pygame.font.SysFont("monospace", 18, bold=True)
    clock = pygame.time.Clock()

    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        vilag = client.load_world("Town04")
        bp_lib = vilag.get_blueprint_library()
        spawn_pontok = vilag.get_map().get_spawn_points()

        tm = client.get_trafficmanager(8000)
        tm.set_global_distance_to_leading_vehicle(2.5)
        tm.set_hybrid_physics_mode(True)
        tm.set_hybrid_physics_radius(50.0)

        ego_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        ego_bp.set_attribute("role_name", "ego")
        ego_bp.set_attribute("color", "255,0,0")
        ego_spawn = random.choice(spawn_pontok)
        ego = vilag.spawn_actor(ego_bp, ego_spawn)
        actor_lista.append(ego)
        ego.set_autopilot(True, tm.get_port())

        kam_bp = bp_lib.find("sensor.camera.rgb")
        kam_bp.set_attribute("image_size_x", str(W))
        kam_bp.set_attribute("image_size_y", str(H))
        kam_bp.set_attribute("fov", "90")
        kamera = vilag.spawn_actor(
            kam_bp, carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=ego)
        actor_lista.append(kamera)
        kamera.listen(kamera_callback)

        pontok = [p for p in spawn_pontok if p.location.distance(ego_spawn.location) > 5.0]
        random.shuffle(pontok)
        for p in pontok[:30]:
            npc = vilag.try_spawn_actor(random.choice(bp_lib.filter("vehicle.*")), p)
            if npc:
                npc.set_autopilot(True, tm.get_port())
                actor_lista.append(npc)

        fut = True
        while fut:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    fut = False

            if kamera_kep_surface is not None:
                display.blit(kamera_kep_surface, (0, 0))

            info = get_vehicle_surroundings(ego)
            print(info)
            v = info["vehicles"]

            sorok = [
                f"FRONT: {v['front']} | BEHIND: {v['behind']}",
                f"LEFT:  {v['left']} | RIGHT:  {v['right']}",
                "",
                f"BAL VONAL:  {info['left_marking']}",
                f"JOBB VONAL: {info['right_marking']}",
            ]
            for i, s in enumerate(sorok):
                display.blit(font.render(s, True, (255, 255, 0), (0, 0, 0)), (20, 20 + i * 25))

            pygame.display.flip()
    finally:
        for a in actor_lista:
            if a is not None:
                a.destroy()
        pygame.quit()


if __name__ == "__main__":
    main()