import carla
import numpy as np
import math as mt
from numpy.matlib import repmat
import ctypes


def spawn_camera(camera, world, blueprint_library, vehicle, img_width, img_height, camera_transform):
    camera_bp = blueprint_library.find(camera)

    camera_bp.set_attribute('image_size_x', f"{img_width}")
    camera_bp.set_attribute('image_size_y', f"{img_height}")
    camera_bp.set_attribute('fov', f"{90}")
    if camera == 'sensor.camera.depth':  # Remove any distortion from the depth camera
        camera_bp.set_attribute('lens_circle_falloff', '0')
        camera_bp.set_attribute('lens_circle_multiplier', '0')
        camera_bp.set_attribute('lens_x_size', '0')
        camera_bp.set_attribute('lens_y_size', '0')
        camera_bp.set_attribute('lens_k', '0')
        camera_bp.set_attribute('lens_kcube', '0')

    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    return camera


def spawn_cameras(world, blueprint_library, vehicle, IMG_WIDTH, IMG_HEIGHT):
    # Depth & RGB FRONT
    front_camera_transform = carla.Transform(carla.Location(z=2.5))
    front_depth_camera = spawn_camera('sensor.camera.depth', world, blueprint_library, vehicle, IMG_WIDTH, IMG_HEIGHT,
                                      front_camera_transform)
    front_rgb_camera = spawn_camera('sensor.camera.rgb', world, blueprint_library, vehicle, IMG_WIDTH, IMG_HEIGHT,
                                    front_camera_transform)
    # Depth & RGB RIGHT
    right_camera_transform = carla.Transform(carla.Location(z=2.5), carla.Rotation(yaw=90.0))
    right_depth_camera = spawn_camera('sensor.camera.depth', world, blueprint_library, vehicle, IMG_WIDTH, IMG_HEIGHT,
                                      right_camera_transform)
    right_rgb_camera = spawn_camera('sensor.camera.rgb', world, blueprint_library, vehicle, IMG_WIDTH, IMG_HEIGHT,
                                    right_camera_transform)
    # Depth & RGB LEFT
    left_camera_transform = carla.Transform(carla.Location(z=2.5), carla.Rotation(yaw=-90.0))
    left_depth_camera = spawn_camera('sensor.camera.depth', world, blueprint_library, vehicle, IMG_WIDTH, IMG_HEIGHT,
                                     left_camera_transform)
    left_rgb_camera = spawn_camera('sensor.camera.rgb', world, blueprint_library, vehicle, IMG_WIDTH, IMG_HEIGHT,
                                   left_camera_transform)
    # Depth & RGB BACK
    back_camera_transform = carla.Transform(carla.Location(z=2.5), carla.Rotation(yaw=180.0))
    back_depth_camera = spawn_camera('sensor.camera.depth', world, blueprint_library, vehicle, IMG_WIDTH, IMG_HEIGHT,
                                     back_camera_transform)
    back_rgb_camera = spawn_camera('sensor.camera.rgb', world, blueprint_library, vehicle, IMG_WIDTH, IMG_HEIGHT,
                                   back_camera_transform)

    return front_depth_camera, front_rgb_camera, right_depth_camera, right_rgb_camera, left_depth_camera, left_rgb_camera, back_depth_camera, back_rgb_camera


def get_intrinsic_extrinsic_matrix(camera_depth, image_depth):
    """
    This function is used to calculate the intrinsic and extrinsic matrices of a camera based on its depth and image depth.

    :param camera_depth: The `camera_depth` parameter refers to the depth camera information.
    :param image_depth: The `image_depth` parameter typically refers to the depth information of an image.
    """

    # -------- Extrinsic matrix
    camera2vehicle_matrix = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)

    pitch = camera_depth.get_transform().rotation.pitch / 180.0 * mt.pi
    yaw = camera_depth.get_transform().rotation.yaw / 180.0 * mt.pi
    roll = camera_depth.get_transform().rotation.roll / 180.0 * mt.pi
    loc_x = camera_depth.get_transform().location.x
    loc_y = - camera_depth.get_transform().location.y
    loc_z = camera_depth.get_transform().location.z
    sin_y, sin_p, sin_r = mt.sin(yaw), mt.sin(pitch), mt.sin(roll)
    cos_y, cos_p, cos_r = mt.cos(yaw), mt.cos(pitch), mt.cos(roll)

    transform_matrix = np.array([
        [cos_y * cos_p, cos_y * sin_p * sin_r + sin_y * cos_r, -cos_y * sin_p * cos_r + sin_y * sin_r, loc_x],
        [-sin_y * cos_p, -sin_y * sin_p * sin_r + cos_y * cos_r, sin_y * sin_p * cos_r + cos_y * sin_r, loc_y],
        [sin_p, -cos_p * sin_r, cos_p * cos_r, loc_z],
        [0.0, 0.0, 0.0, 1.0]
    ])

    camera2world_matrix = transform_matrix @ camera2vehicle_matrix

    # -------- Intrinsics matrix
    focal_lengthX = image_depth.width / (2.0 * mt.tan(90 * mt.pi / 360.0))
    centerX = image_depth.width / 2
    centerY = image_depth.height / 2

    intrinsic_matrix = [[focal_lengthX, 0, centerX],
                        [0, focal_lengthX, centerY],
                        [0, 0, 1]]

    return intrinsic_matrix, camera2world_matrix


def _to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA numpy array."""

    if not isinstance(image, carla.Image):
        raise ValueError("Argument must be a carla.sensor.Image")
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))

    return array


def _depth_to_array(image):
    """
    Convert an image containing CARLA encoded depth-map to a 2D array containing
    the depth value of each pixel normalized between [0.0, 1.0].
    """

    array = _to_bgra_array(image)
    array = array.astype(np.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    normalized_depth = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)

    return normalized_depth


def point2D_to_point3D(image_depth, image_rgb, intrinsic_matrix):
    """
    This function converts a 2D point to a 3D point using image depth, image RGB, and intrinsic matrix.

    :param image_depth: The `image_depth` is a 2D image representing the depth information of the scene.
    :param image_rgb: The `image_rgb` is a 2D image containing color information of the scene.
    :param intrinsic_matrix: The intrinsic matrix is a 3x3 matrix to represents the internal parameters of the depth camera.
    """

    intrinsic_matrix_inv = np.linalg.inv(intrinsic_matrix)

    pixel_length = image_depth.width * image_depth.height

    # Return a array (height, width, 4) with the BGRA values of each pixel
    normalized_depth = _depth_to_array(image_depth)
    normalized_depth = np.reshape(normalized_depth, pixel_length)

    color = image_rgb.reshape(pixel_length, 3)

    u_coord = repmat(np.r_[image_depth.width - 1:-1:-1],
                     image_depth.height, 1).reshape(pixel_length)
    v_coord = repmat(np.c_[image_depth.height - 1:-1:-1],
                     1, image_depth.width).reshape(pixel_length)

    depth_in_meters = normalized_depth * 1000

    # get only the points with depth less than 90 meters
    max_depth_indexes = np.where(depth_in_meters > 90)

    depth_in_meters = np.delete(depth_in_meters, max_depth_indexes)
    u_coord = np.delete(u_coord, max_depth_indexes)
    v_coord = np.delete(v_coord, max_depth_indexes)
    color = np.delete(color, max_depth_indexes, axis=0)

    # Convert the 2D pixel coordinates to 3D points
    p2d = np.array([u_coord, v_coord, np.ones_like(u_coord)])
    p3d = np.dot(intrinsic_matrix_inv, p2d) * depth_in_meters

    # Return [[X...], [Y...], [Z...]] and [[R...], [G...], [B...]] normalized
    return p3d, color


def downsample(points, colors, leaf_size):
    """
    The function `downsample` takes in arrays of points and colors, passes them to a C function for
    downsampling, and returns the downsampled points and colors.

    :param points: The `points` parameter is expected to be a numpy array containing the coordinates of points in a point cloud.
                   Each row of the array represents a point in 3D space with its x, y, and z coordinates.
    :param colors: The `colors` parameter is expected to be a numpy array containing the RGB of points in a point cloud.
                   Each row of the array represents a point in 3D space with its R, G, and B colors.
    :param leaf_size: The `leaf_size` parameter is the size of the leaf for the downsampling algorithm.
    :return: The `downsample` function returns two numpy arrays: `output_points` and `output_colors`,
             which contain the downsampled points and colors of the point cloud, respectively.
    """

    pcl_lib = ctypes.cdll.LoadLibrary("./utils/ground_truth/build/libpcl_downsample.so")

    # To pass the numpy array to the C function
    ND_POINTER = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C")

    # Define the prototype of the function
    pcl_lib.pcl_downSample.argtypes = [ND_POINTER, ND_POINTER, ctypes.c_float, ctypes.c_size_t, ND_POINTER, ND_POINTER]
    pcl_lib.pcl_downSample.restype = ctypes.c_int

    # put the arrays in a contiguous memory to pass to the C function
    points = np.ascontiguousarray(points, dtype=np.float64)
    colors = np.ascontiguousarray(colors, dtype=np.float64)

    # To get the output points and colors of the downsampled point cloud
    output_points = np.zeros((points.shape[0], 3)).astype(np.float64)
    output_colors = np.zeros((colors.shape[0], 3)).astype(np.float64)

    # Call the C function to downsample the point cloud
    pcl_lib.pcl_downSample(points, colors, ctypes.c_float(float(leaf_size)), ctypes.c_size_t(points.size // 3),
                           output_points, output_colors)

    return output_points, output_colors













