import random
import math as mt


def spawn_vehicle_route(world, blueprint_library, traffic_manager, route_map, route_name, weather_type):
    route = []
    spawn_points = world.get_map().get_spawn_points()

    # Get the route indices of the route_name
    route_indices = route_map[route_name]
    print(f"{route_name}: {route_indices}")

    # Get the distance between all points in route_indices, in meters
    total_distance = 0
    for i in range(len(route_indices) - 1):
        loc1 = spawn_points[route_indices[i]].location
        loc2 = spawn_points[route_indices[i + 1]].location
        # print(f"Idx: {route_indices[i]} -> {route_indices[i+1]} = {loc1.distance(loc2)} meters")

        total_distance += loc1.distance(loc2)
    print(f"Total distance: {mt.ceil(total_distance)} meters")

    # spawn the vehicle at the starting point
    vehicle = spawn_vehicle(world, blueprint_library, start_point=route_indices[0])
    print("Vehicle spawned!")

    for ind in route_indices:
        route.append(spawn_points[ind].location)

    traffic_manager.ignore_lights_percentage(vehicle, 100)  # Ignore all the red ligths
    traffic_manager.random_left_lanechange_percentage(vehicle, 0)
    traffic_manager.random_right_lanechange_percentage(vehicle, 0)
    traffic_manager.auto_lane_change(vehicle, False)
    traffic_manager.set_path(vehicle, route)  # Set the path of the vehicle
    if weather_type == "NightCloudy":
        traffic_manager.update_vehicle_lights(vehicle, True)

    return vehicle, spawn_points[route_indices[0]]


def spawn_vehicle(world, blueprint_library, start_point=random.randint(0, 30)):
    # Get the blueprint for the vehicle - Tesla Model 3
    bp = blueprint_library.filter('model3')[0]

    # Get the spawn point
    transform = world.get_map().get_spawn_points()[start_point]

    vehicle = world.spawn_actor(bp, transform)

    vehicle.set_autopilot(True)

    return vehicle

def spawn_npc(world, blueprint_library, start_point):
    # Get the blueprint for the vehicle - Tesla Model 3
    bp = blueprint_library.filter('model3')[0]

    # Get the spawn point
    transform = world.get_map().get_spawn_points()[start_point]

    vehicle = world.spawn_actor(bp, transform)

    vehicle.set_autopilot(True)

    return vehicle