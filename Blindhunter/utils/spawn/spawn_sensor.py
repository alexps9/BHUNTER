import carla


def set_attributes_lidar(sensor_bp, attributes):
    attrs = attributes.get("real_lidar", attributes["real_lidar"])

    for attr, value in attrs.items():
        # print(f"LIDAR: {attr} - {value}")
        sensor_bp.set_attribute(str(attr), str(value))


def set_atributes_rgb(camera_bp, attributes):
    attrs = attributes.get("real_rgb", attributes["real_rgb"])

    for attr, value in attrs.items():
        # print(f"RBG: {attr} - {value}")
        camera_bp.set_attribute(str(attr), str(value))


def spawn_sensores(sensor, world, blueprint_library, vehicle, attributes):
    sensor_bp = blueprint_library.find(sensor)

    if sensor.startswith('sensor.camera'):

        # Set attributes for the camera (common for rgb and depth)
        attrs = attributes.get("rgb_and_depth", attributes["rgb_and_depth"])
        for attr, value in attrs.items():
            # print(f"RBG and Depth: {attr} - {value}")
            sensor_bp.set_attribute(str(attr), str(value))

        # Set attributes for the rgb camera
        if sensor == 'sensor.camera.rgb':
            set_atributes_rgb(sensor_bp, attributes)

        # Posição da camera em metros
        sensor_transform = carla.Transform(carla.Location(x=0.9, z=1.3))

    elif sensor.startswith('sensor.lidar'):
        # Set attributes for the lidar
        set_attributes_lidar(sensor_bp, attributes)

        # Posição do sensor em metros
        sensor_transform = carla.Transform(carla.Location(z=2.5))

    camera = world.spawn_actor(sensor_bp, sensor_transform, attach_to=vehicle)

    return camera

