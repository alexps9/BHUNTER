import yaml
import carla
def load_sensor_config(config_path):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: The file '{config_path}' was not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file. Details: {e}")
        return None

def apply_sensor_semantic(world, blueprint_library, vehicle, sensor_config):
    if str(sensor_config['name']) == 'kitti_lidar':
        print("switch lidar to semantic!")
        sensor_bp = blueprint_library.find("sensor.lidar.ray_cast_semantic")

    else:
        sensor_bp = blueprint_library.find(sensor_config['bp_name'])
    if 'options' in sensor_config:
        for key, value in sensor_config['options'].items():
            sensor_bp.set_attribute(str(key), str(value))
    location = carla.Location(
        x=sensor_config['location']['x'],
        y=sensor_config['location']['y'],
        z=sensor_config['location']['z']
    )
    rotation = carla.Rotation(
        roll=sensor_config['rotation']['roll'],
        pitch=sensor_config['rotation']['pitch'],
        yaw=sensor_config['rotation']['yaw']
    )
    transform = carla.Transform(location, rotation)

    try:
        sensor = world.spawn_actor(sensor_bp, transform, attach_to=vehicle)
        print(f"Sensor '{sensor_config['name']}' successfully added.")
        return sensor
    except Exception as e:
        print(f"Failed to add sensor '{sensor_config['name']}': {e}")
        return None

def apply_sensor(world, blueprint_library, vehicle, sensor_config):
    sensor_bp = blueprint_library.find(sensor_config['bp_name'])
    if 'options' in sensor_config:
        for key, value in sensor_config['options'].items():
            sensor_bp.set_attribute(str(key), str(value))
    location = carla.Location(
        x=sensor_config['location']['x'],
        y=sensor_config['location']['y'],
        z=sensor_config['location']['z']
    )
    rotation = carla.Rotation(
        roll=sensor_config['rotation']['roll'],
        pitch=sensor_config['rotation']['pitch'],
        yaw=sensor_config['rotation']['yaw']
    )
    transform = carla.Transform(location, rotation)

    try:
        sensor = world.spawn_actor(sensor_bp, transform, attach_to=vehicle)
        print(f"Sensor '{sensor_config['name']}' successfully added.")
        return sensor
    except Exception as e:
        print(f"Failed to add sensor '{sensor_config['name']}': {e}")
        return None


def setup_vehicle_sensors(world, vehicle, config_path):
    blueprint_library = world.get_blueprint_library()
    sensor_configs = load_sensor_config(config_path)['sensors']

    sensors = []
    for sensor_config in sensor_configs:
        sensor = apply_sensor(world, blueprint_library, vehicle, sensor_config)
        if sensor:
            sensors.append(sensor)

    return sensors

"""
=== an example of sensor config===
sensors:
  #-----------camera----------
  - name: "kitti_camera_rgb"
    bp_name: 'sensor.camera.rgb'
    location:
      x: 0.0
      y: 0.0
      z: 1.6
    rotation:
      yaw: 0.0
      pitch: 0.0
      roll: 0.0
    options:
      "image_size_x": "1248"
      "image_size_y": "384"
      "fov": "90"

  #-----------lidar----------
  - name: "kitti_lidar"
    bp_name: 'sensor.lidar.ray_cast'
    location:
      x: 0.0
      y: 0.0
      z: 1.6
    rotation:
      yaw: 0.0
      pitch: 0.0
      roll: 0.0
    options:
      "channels": "40"
      "range": "70.0"
      "points_per_second": "720000"
      "rotation_frequency": "10"
      "upper_fov": "7"
      "lower_fov": "-16"

  #-----------depth camera----------
  - name: "kitti_depth_camera"
    bp_name: 'sensor.camera.depth'
    location:
      x: 0.0
      y: 0.0
      z: 1.6
    rotation:
      yaw: 0.0
      pitch: 0.0
      roll: 0.0
    options:
      "image_size_x": "1248"
      "image_size_y": "384"
      "fov": "90"

=============================
"""