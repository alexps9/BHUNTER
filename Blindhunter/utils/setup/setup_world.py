import glob
import os
import sys
import carla


def server_settings(world):
    """
    Set the server settings for the given world object.

    Parameters:
    - world: the world object on which to apply the settings
    """
    settings = world.get_settings()
    settings.no_rendering_mode = True  # No rendering mode
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)


def set_traffic_light_timings(world, green_time=10.0, yellow_time=2.0, red_time=1.0):
    """
    Set custom timings for traffic lights in the CARLA world.

    :param world: The CARLA world object.
    :param green_time: Duration of the green light in seconds.
    :param yellow_time: Duration of the yellow light in seconds.
    :param red_time: Duration of the red light in seconds.
    """

    traffic_lights = world.get_actors().filter('traffic.traffic_light')

    for light in traffic_lights:
        # Set the timing for each phase
        light.set_green_time(green_time)
        light.set_yellow_time(yellow_time)
        light.set_red_time(red_time)


def setup_carla(map):
    '''
    try:
        sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except IndexError:
        pass
    '''
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()

    if not world.get_map().name == 'Carla/Maps/' + map:
        print(f"Current world: {world.get_map().name}... loading {map}...")
        client.load_world(map)
    print(f"Current world: {world.get_map().name}")

    # To edit the traffic
    traffic_manager = client.get_trafficmanager(8000)

    blueprint_library = world.get_blueprint_library()

    server_settings(world)
    set_traffic_light_timings(world)

    return world, blueprint_library, traffic_manager