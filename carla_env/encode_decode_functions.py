import numpy as np

from carla_env.wrappers import vector, get_displacement_vector


def create_encode_state_fn():
    # Encode current state
    def encode_state(env):
        # dict for current CARLA state
        encoded_state = {}

        # Nyers szegmentalt kep. uint8, (80,160,3), NEM normalizalva -
        # az SB3 NatureCNN belul oszt 255-tel.
        encoded_state['seg_camera'] = np.asarray(env.observation, dtype=np.uint8)

        vehicle_measures = []

        # ask current vechile measures steer, throttle, speed, angle, waypoint
        vehicle_measures.append(env.vehicle.control.steer)
        vehicle_measures.append(env.vehicle.control.throttle)
        vehicle_measures.append(env.vehicle.get_speed())
        vehicle_measures.append(env.vehicle.get_angle(env.current_waypoint))

        # Append to dict
        encoded_state['vehicle_measures'] = vehicle_measures

        # actual vehicle maneuver
        encoded_state['maneuver'] = env.current_road_maneuver.value
        next_waypoints_state = env.route_waypoints[env.current_waypoint_index: env.current_waypoint_index + 15]
        waypoints = [vector(way[0].transform.location) for way in next_waypoints_state]
        vehicle_location = vector(env.vehicle.get_location())
        theta = np.deg2rad(env.vehicle.get_transform().rotation.yaw)
        relative_waypoints = np.zeros((15, 2))
        for i, w_location in enumerate(waypoints):
            relative_waypoints[i] = get_displacement_vector(vehicle_location, w_location, theta)[:2]
        if len(waypoints) < 15:
            start_index = len(waypoints)
            reference_vector = relative_waypoints[start_index-1] - relative_waypoints[start_index-2]
            for i in range(start_index, 15):
                relative_waypoints[i] = relative_waypoints[i-1] + reference_vector
        encoded_state['waypoints'] = relative_waypoints
        return encoded_state

    return encode_state