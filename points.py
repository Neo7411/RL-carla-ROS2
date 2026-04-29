import carla
import time

def main():
    # 1. Connect to the CARLA server
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0) # Wait up to 10 seconds for a connection
        print("Successfully connected to CARLA server.")
    except Exception as e:
        print(f"Failed to connect to CARLA server: {e}")
        return

    # 2. Get the current world and map
    world = client.load_world("Town04")
    carla_map = world.get_map()

    # 3. Retrieve all recommended spawn points
    spawn_points = carla_map.get_spawn_points()
    print(f"Found {len(spawn_points)} spawn points on the map: {carla_map.name}\n")

    # 4. Iterate through spawn points, calculate Lat/Lon, and visualize
    debug = world.debug
    life_time = 120.0 # How long the visualization stays on screen (in seconds)

    for i, spawn_point in enumerate(spawn_points):
        location = spawn_point.location
        
        # Convert CARLA Cartesian coordinates (X, Y, Z) to GeoLocation (Lat, Lon, Alt)
        geo_location = carla_map.transform_to_geolocation(location)
        lat = geo_location.latitude
        lon = geo_location.longitude

        # Print to terminal
        print(f"Spawn Point {i:03d} | X: {location.x:.2f}, Y: {location.y:.2f}, Z: {location.z:.2f} | Lat: {lat:.6f}, Lon: {lon:.6f}")

        # Draw a red point at the exact spawn location in the simulator
        debug.draw_point(
            location, 
            size=0.1, 
            color=carla.Color(r=255, g=0, b=0), 
            life_time=life_time
        )

        # Draw green text (Lat/Lon) slightly above the point so it is readable
        text_location = carla.Location(x=location.x, y=location.y, z=location.z + 1.5)
        text = f"SP {i}\nLat: {lat:.5f}\nLon: {lon:.5f}"
        
        debug.draw_string(
            text_location, 
            text, 
            draw_shadow=True, 
            color=carla.Color(r=0, g=255, b=0), 
            life_time=life_time
        )

    print(f"\nVisualization complete! Points will remain in the simulator for {life_time} seconds.")

if __name__ == '__main__':
    main()