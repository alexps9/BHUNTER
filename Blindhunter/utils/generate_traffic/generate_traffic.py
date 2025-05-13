import carla
import random


def gennerate_vehicles(world, traffic_manager, blueprint_library, vehicle_spawn_point, number_of_vehicles=70):
    vehicles_list = []

    spawn_points = world.get_map().get_spawn_points()

    while len(vehicles_list) < number_of_vehicles:
        # i = len(vehicles_list)
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
        spawn_point = random.choice(spawn_points)

        if spawn_point != vehicle_spawn_point:
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
            vehicle.set_autopilot(True, traffic_manager.get_port())
            traffic_manager.ignore_lights_percentage(vehicle, 80)  # Ignore all the red ligths
            vehicles_list.append(vehicle)
            # print(f"Spawned vehicle {i+1}/{number_of_vehicles}")
            spawn_points.remove(spawn_point)

    print(f"Traffic vehicles spawned: {len(vehicles_list)}/{number_of_vehicles}")

    return vehicles_list


def gennerate_pedestrians(world, blueprint_library, number_of_pedestrians=50):
    pedestrians_list = []
    controllers_list = []
    spawn_points = []

    # Retrieve pedestrian blueprints (filtered for walkers)
    pedestrian_blueprints = blueprint_library.filter('walker.pedestrian.*')

    # Retrieve the pedestrian controller blueprint
    walker_controller_bp = blueprint_library.find('controller.ai.walker')

    # Get the spawn points to pedestrians
    while len(spawn_points) < number_of_pedestrians + 50:
        spawn_point = carla.Transform()

        loc = world.get_random_location_from_navigation()
        if (loc != None):
            spawn_point.location = loc
            spawn_points.append(spawn_point)

    while len(pedestrians_list) < number_of_pedestrians:
        # i = len(pedestrians_list)
        # Choose a pedestrian blueprint and a spawn point randomly
        pedestrian_bp = random.choice(pedestrian_blueprints)
        spawn_point = random.choice(spawn_points)

        # Spawn the pedestrian
        pedestrian = world.try_spawn_actor(pedestrian_bp, spawn_point)
        if pedestrian:
            pedestrians_list.append(pedestrian)
            # print(f"Spawned pedestrian {i+1}/{number_of_pedestrians}")

            # Spawn the controller for the pedestrian
            controller = world.try_spawn_actor(walker_controller_bp, carla.Transform(), pedestrian)
            if controller:
                controllers_list.append(controller)

                world.tick()

                controller.start()

                # Get a random location for the pedestrian to walk to
                target_location = world.get_random_location_from_navigation()

                if target_location != None:
                    # Validate the location is within bounds and reachable
                    controller.go_to_location(target_location)
                    controller.set_max_speed(1 + random.random())  # Random speed between 1 and 2 m/s

        # Remove used spawn point to avoid reusing it
        spawn_points.remove(spawn_point)

    world.set_pedestrians_cross_factor(0.0)  # Percentage of pedestrians crossing the road
    print(f"Pedestrians spawned: {len(pedestrians_list)}/{number_of_pedestrians}")

    return pedestrians_list, controllers_list







