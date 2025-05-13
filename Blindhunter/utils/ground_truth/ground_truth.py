import pdb
from sklearn.cluster import DBSCAN
import carla
import numpy as np
import math as mt
from numpy.matlib import repmat
import ctypes
from scipy.ndimage import minimum_filter
import cv2
from sklearn.cluster import KMeans

interesting_labels = [14]


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
    # Depth RIGHT
    right_camera_transform = carla.Transform(carla.Location(z=2.5), carla.Rotation(yaw=90.0))
    right_depth_camera = spawn_camera('sensor.camera.depth', world, blueprint_library, vehicle, IMG_WIDTH, IMG_HEIGHT,
                                      right_camera_transform)
    # Depth LEFT
    left_camera_transform = carla.Transform(carla.Location(z=2.5), carla.Rotation(yaw=-90.0))
    left_depth_camera = spawn_camera('sensor.camera.depth', world, blueprint_library, vehicle, IMG_WIDTH, IMG_HEIGHT,
                                     left_camera_transform)
    # Depth BACK
    back_camera_transform = carla.Transform(carla.Location(z=2.5), carla.Rotation(yaw=180.0))
    back_depth_camera = spawn_camera('sensor.camera.depth', world, blueprint_library, vehicle, IMG_WIDTH, IMG_HEIGHT,
                                     back_camera_transform)

    return front_depth_camera, front_rgb_camera, right_depth_camera, left_depth_camera, back_depth_camera


def get_intrinsic_extrinsic_matrix_1(camera_depth, image_depth):
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

    return intrinsic_matrix, camera2world_matrix, transform_matrix


def get_intrinsic_extrinsic_matrix(camera_depth, image_depth):
    """
    This function is used to calculate the intrinsic and extrinsic matrices of a camera based on its depth and image depth.

    :param camera_depth: The `camera_depth` parameter refers to the depth camera information.
    :param image_depth: The `image_depth` parameter typically refers to the depth information of an image.
    """

    # -------- Extrinsic matrix     (x,y,z)-->(z,x,y)
    camera2vehicle_matrix = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    depth_camera_transform = camera_depth.get_transform()
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

    transform_matrix_ = np.array(depth_camera_transform.get_matrix())

    camera2world_matrix = transform_matrix_ @ camera2vehicle_matrix

    # -------- Intrinsics matrix
    focal_lengthX = image_depth.width / (2.0 * mt.tan(mt.radians(90 / 2.0)))
    centerX = image_depth.width / 2
    centerY = image_depth.height / 2

    intrinsic_matrix = [[focal_lengthX, 0, centerX],
                        [0, focal_lengthX, centerY],
                        [0, 0, 1]]

    return intrinsic_matrix, camera2world_matrix


def get_intrinsic_matrix(camera):
    width = int(camera.attributes['image_size_x'])
    height = int(camera.attributes['image_size_y'])
    fov = float(camera.attributes['fov'])

    k = np.identity(3)
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    k[0, 0] = k[1, 1] = width / (2.0 * np.tan(fov * np.pi / 360.0))

    return k


def get_intrinsic_extrinsic_matrix_(camera_depth, image_depth, vehicle):
    """
    This function calculates the intrinsic and extrinsic matrices of a camera based on its depth, image depth,
    and the vehicle's position and orientation.

    :param camera_depth: The `camera_depth` parameter refers to the depth camera information.
    :param image_depth: The `image_depth` parameter refers to the image's depth information.
    :param vehicle: The `vehicle` parameter is a Carla.Vehicle object representing the vehicle information.
    :return: Intrinsic and extrinsic matrices.
    """

    # -------- Extrinsic matrix     (x,y,z)-->(z,x,y)
    camera2vehicle_matrix = np.array([[0, 0, 1, 0],
                                      [1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 0, 1]], dtype=np.float64)

    # Camera's rotation and location in the vehicle's coordinate system
    pitch = camera_depth.get_transform().rotation.pitch / 180.0 * mt.pi
    yaw = camera_depth.get_transform().rotation.yaw / 180.0 * mt.pi
    roll = camera_depth.get_transform().rotation.roll / 180.0 * mt.pi
    loc_x = camera_depth.get_transform().location.x
    loc_y = -camera_depth.get_transform().location.y
    loc_z = camera_depth.get_transform().location.z

    sin_y, sin_p, sin_r = mt.sin(yaw), mt.sin(pitch), mt.sin(roll)
    cos_y, cos_p, cos_r = mt.cos(yaw), mt.cos(pitch), mt.cos(roll)

    # Transform matrix: Camera to Vehicle
    transform_matrix = np.array([
        [cos_y * cos_p, cos_y * sin_p * sin_r + sin_y * cos_r, -cos_y * sin_p * cos_r + sin_y * sin_r, loc_x],
        [-sin_y * cos_p, -sin_y * sin_p * sin_r + cos_y * cos_r, sin_y * sin_p * cos_r + cos_y * sin_r, loc_y],
        [sin_p, -cos_p * sin_r, cos_p * cos_r, loc_z],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # Vehicle's rotation and location in the world coordinate system
    vehicle_rotation = vehicle.get_transform().rotation
    vehicle_location = vehicle.get_transform().location

    # Vehicle rotation in radians
    vehicle_yaw = vehicle_rotation.yaw / 180.0 * mt.pi
    vehicle_pitch = vehicle_rotation.pitch / 180.0 * mt.pi
    vehicle_roll = vehicle_rotation.roll / 180.0 * mt.pi

    sin_vy, sin_vp, sin_vr = mt.sin(vehicle_yaw), mt.sin(vehicle_pitch), mt.sin(vehicle_roll)
    cos_vy, cos_vp, cos_vr = mt.cos(vehicle_yaw), mt.cos(vehicle_pitch), mt.cos(vehicle_roll)

    # Vehicle to World Transform (Rotation and Translation)
    vehicle2world_matrix = np.array([
        [cos_vy * cos_vp, cos_vy * sin_vp * sin_vr + sin_vy * cos_vr, -cos_vy * sin_vp * cos_vr + sin_vy * sin_vr,
         vehicle_location.x],
        [-sin_vy * cos_vp, -sin_vy * sin_vp * sin_vr + cos_vy * cos_vr, sin_vy * sin_vp * cos_vr + cos_vy * sin_r,
         vehicle_location.y],
        [sin_vp, -cos_vp * sin_vr, cos_vp * cos_vr, vehicle_location.z],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # Combine all matrices: Camera to Vehicle, Vehicle to World
    camera2world_matrix = vehicle2world_matrix @ transform_matrix @ camera2vehicle_matrix

    # -------- Intrinsic matrix
    focal_lengthX = image_depth.width / (2.0 * mt.tan(mt.radians(90 / 2.0)))  # Assuming a 90° FOV
    centerX = image_depth.width / 2
    centerY = image_depth.height / 2

    intrinsic_matrix = [[focal_lengthX, 0, centerX],
                        [0, focal_lengthX, centerY],
                        [0, 0, 1]]

    return intrinsic_matrix, camera2world_matrix


'''
def get_intrinsic_extrinsic_matrix(camera_depth, image_depth):
    """
    Calculate the intrinsic and extrinsic matrices of a camera based on its depth and image depth.

    :param camera_depth: CARLA camera object providing position and orientation.
    :param image_depth: Depth image from CARLA, used for image dimensions.
    :return: Intrinsic matrix (3x3) and Extrinsic matrix (4x4).
    """

    # -------- Extrinsic matrix: Camera to World Transformation --------
    pitch = camera_depth.get_transform().rotation.pitch / 180.0 * mt.pi
    yaw = camera_depth.get_transform().rotation.yaw / 180.0 * mt.pi
    roll = camera_depth.get_transform().rotation.roll / 180.0 * mt.pi
    loc_x = camera_depth.get_transform().location.x
    loc_y = camera_depth.get_transform().location.y
    loc_z = camera_depth.get_transform().location.z

    sin_y, sin_p, sin_r = mt.sin(yaw), mt.sin(pitch), mt.sin(roll)
    cos_y, cos_p, cos_r = mt.cos(yaw), mt.cos(pitch), mt.cos(roll)

    # Build the transformation matrix
    transform_matrix = np.array([
        [cos_y * cos_p, cos_y * sin_p * sin_r - sin_y * cos_r, cos_y * sin_p * cos_r + sin_y * sin_r, loc_x],
        [sin_y * cos_p, sin_y * sin_p * sin_r + cos_y * cos_r, sin_y * sin_p * cos_r - cos_y * sin_r, loc_y],
        [-sin_p, cos_p * sin_r, cos_p * cos_r, loc_z],
        [0.0, 0.0, 0.0, 1.0]
    ])
    camera2world_matrix = transform_matrix

    # -------- Intrinsic matrix: Camera Internal Parameters --------
    fov = 90.0  # Field of View, adjust if camera's FOV is different
    focal_lengthX = image_depth.width / (2.0 * mt.tan(mt.radians(fov / 2.0)))
    centerX = image_depth.width / 2.0
    centerY = image_depth.height / 2.0

    intrinsic_matrix = np.array([
        [focal_lengthX, 0, centerX],
        [0, focal_lengthX, centerY],
        [0, 0, 1]
    ])

    return intrinsic_matrix, camera2world_matrix
'''


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


def point2D_to_point3D_(image_depth, intrinsic_matrix, segmentation_image):
    """
    Converts 2D points to 3D points using depth image, intrinsic matrix, and segmentation image.
    Filters only the points corresponding to interesting labels in the segmentation image.

    :param image_depth: Depth image from CARLA containing depth information.
    :param intrinsic_matrix: Intrinsic matrix (3x3) representing camera's internal parameters.
    :param segmentation_image: Semantic segmentation image where the red channel encodes labels.
    :param interesting_labels: List of labels of interest (e.g., vehicles, pedestrians).
    :return: Filtered 3D points and corresponding colors.
    """
    # Compute inverse of intrinsic matrix
    intrinsic_matrix_inv = np.linalg.inv(intrinsic_matrix)

    # Get depth information as a 2D numpy array
    normalized_depth = _depth_to_array(image_depth)  # Depth as 2D numpy array
    height, width = image_depth.height, image_depth.width
    normalized_depth = normalized_depth.reshape((height, width))

    # Extract red channel from segmentation image for label information
    segmentation_labels = np.frombuffer(segmentation_image.raw_data, dtype=np.uint8)
    segmentation_labels = segmentation_labels.reshape((height, width, 4))[:, :, 2]  # Red channel

    # Create pixel coordinate grid
    u_coord, v_coord = np.meshgrid(np.arange(width), np.arange(height))

    # Flatten all arrays for processing
    u_coord = u_coord.flatten()
    v_coord = v_coord.flatten()
    depth_in_meters = normalized_depth.flatten() * 1000  # Convert to meters
    segmentation_labels = segmentation_labels.flatten()

    # Mask to select pixels corresponding to interesting labels
    label_mask = np.isin(segmentation_labels, interesting_labels)

    # Apply mask to filter depth and coordinates
    u_coord = u_coord[label_mask]
    v_coord = v_coord[label_mask]
    depth_in_meters = depth_in_meters[label_mask]

    # Remove points with depth > 90 meters (optional)
    valid_depth_mask = depth_in_meters <= 90
    u_coord = u_coord[valid_depth_mask]
    v_coord = v_coord[valid_depth_mask]
    depth_in_meters = depth_in_meters[valid_depth_mask]

    # Convert 2D pixel coordinates to 3D points
    p2d = np.array([u_coord, v_coord, np.ones_like(u_coord)])
    p3d = np.dot(intrinsic_matrix_inv, p2d) * depth_in_meters

    # Assign colors for the selected labels (optional visualization)
    color = np.full((p3d.shape[1], 3), [0, 255, 0])  # Green color for all points

    return p3d, color


def separate_objects_by_known_depth(filtered_depth, distance, delta=3.0):
    """
    Separate points in a depth map based on known distances to objects.

    :param distance: the distance between objects in meters.
    :param filtered_depth: 1D array of depth values (in meters)
    :param delta: Allowed deviation range for matching (in meters)
    :return: List of masks, each corresponding to points belonging to an object
    """

    mask = (filtered_depth >= distance - delta) & (filtered_depth <= distance + delta)
    return mask


def get_occlusion_from_depth(image_depth, segmentation_image, distance, base_depth_threshold=0.3, delta=1.0,
                             max_depth=100.0):
    """
    Compute the occlusion rate for different vehicle instances from a depth image.

    :param image_depth: Depth image object containing height, width, and depth data.
    :param segmentation_image: Segmentation image object containing label information.
    :param base_depth_threshold: Base depth difference threshold to determine occlusion (as a proportion of depth).
    :param delta: Allowed deviation range for matching (in meters).
    :param max_depth: Maximum valid depth value to filter out invalid points.
    :return: A dictionary mapping instance IDs to their occlusion rates.
    """
    # Extract depth data as a 2D numpy array and convert to meters
    normalized_depth = _depth_to_array(image_depth)  # Custom function to extract depth as an array
    height, width = image_depth.height, image_depth.width
    normalized_depth = normalized_depth.reshape((height, width)) * 1000  # Convert depth to meters

    threshold = 10

    # Filter out invalid depth values
    normalized_depth[(normalized_depth < 0) | (normalized_depth > max_depth)] = np.inf

    # Extract segmentation labels
    segmentation_labels = np.frombuffer(segmentation_image.raw_data, dtype=np.uint8)
    segmentation_labels = segmentation_labels.reshape((height, width, 4))

    # Extract channels
    red_channel = segmentation_labels[:, :, 2]  # Semantic labels
    green_channel = segmentation_labels[:, :, 1]
    blue_channel = segmentation_labels[:, :, 0]

    # Compute instance IDs from green and blue channels
    instance_ids = green_channel * 256 + blue_channel

    # Define the semantic label for vehicles in the Red channel (e.g., 10 represents vehicles)
    vehicle_label = 14  # Adjust this value based on your label definition

    # Create a mask for vehicle pixels based on the Red channel
    vehicle_mask = (red_channel == vehicle_label)

    if vehicle_mask.sum() == 0:
        return 0.0

    # Filter instance IDs to only include vehicle pixels
    vehicle_instance_ids = instance_ids[vehicle_mask]

    # Get unique instance IDs for vehicles
    unique_vehicle_ids = np.unique(vehicle_instance_ids)

    # Dictionary to store occlusion rates for each vehicle instance
    max_distance = 0
    total_occlusion_count = 0
    total_count = 0

    for instance_id in unique_vehicle_ids:
        if instance_id == 0:  # Exclude background or invalid labels
            continue

        # Mask for the current vehicle instance
        instance_mask = (instance_ids == instance_id) & vehicle_mask

        # Masked depth for the current instance
        instance_depth = np.where(instance_mask, normalized_depth, np.inf)

        average_depth = np.mean(instance_depth[instance_depth < np.inf])

        if average_depth > max_distance:
            max_distance = average_depth
            candidate_id = instance_id
            total_count = np.sum(instance_mask)  # Total valid pixels for the instance

        depth_map = instance_depth.astype(np.float32)
        smoothed_depth = cv2.GaussianBlur(depth_map, (5, 5), 0)

        gradient_up = cv2.Sobel(smoothed_depth, cv2.CV_64F, 0, 1, ksize=3)
        gradient_down = -gradient_up
        gradient_left = cv2.Sobel(smoothed_depth, cv2.CV_64F, 1, 0, ksize=3)
        gradient_right = -gradient_left

        gradients = np.sqrt(gradient_up ** 2 + gradient_down ** 2 + gradient_left ** 2 + gradient_right ** 2).flatten()
        dynamic_threshold = np.percentile(gradients[instance_mask.flatten()], 95)

        condition_up = gradient_up > dynamic_threshold
        condition_down = gradient_down > dynamic_threshold
        condition_left = gradient_left > dynamic_threshold
        condition_right = gradient_right > dynamic_threshold

        condition_sum = (
                condition_up.astype(int)
                + condition_down.astype(int)
                + condition_left.astype(int)
                + condition_right.astype(int)
        )

        occlusion_mask = condition_sum >= 2
        occlusion_mask[~instance_mask] = False

        # Compute the occlusion rate for the current instance
        occluded_count = np.sum(occlusion_mask)

        total_occlusion_count += occluded_count
        print("Total occlusion count:", occluded_count)

    occlusion_rate = total_occlusion_count / (total_occlusion_count + total_count + 1e-6)

    return occlusion_rate


def point2D_to_point3D(image_depth, intrinsic_matrix):
    """
    This function converts a 2D point to a 3D point using image depth, image RGB, and intrinsic matrix.

    :param image_depth: The `image_depth` is a 2D image representing the depth information of the scene.
    :param intrinsic_matrix: The intrinsic matrix is a 3x3 matrix to represents the internal parameters of the depth camera.
    """

    intrinsic_matrix_inv = np.linalg.inv(intrinsic_matrix)

    pixel_length = image_depth.width * image_depth.height

    # Return a array (height, width, 4) with the BGRA values of each pixel
    normalized_depth = _depth_to_array(image_depth)
    normalized_depth = np.reshape(normalized_depth, pixel_length)

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

    # Convert the 2D pixel coordinates to 3D points
    p2d = np.array([u_coord, v_coord, np.ones_like(u_coord)])
    p3d = np.dot(intrinsic_matrix_inv, p2d) * depth_in_meters

    # Add the 0,0,0 point to the point cloud (90º) -> (4 red dots)
    color = np.full((p3d.shape[1], 3), np.array([0, 255, 0]))  # Green
    '''
    p3d = np.hstack([p3d, [[0], [0], [0]]])
    color = np.vstack([color, [255, 0, 0]]) # RED
    '''
    # Return [[X...], [Y...], [Z...]] and [[R...], [G...], [B...]]
    return p3d, color


def downsample(points, colors, leaf_size):
    """
    The function `downsample` takes in arrays of points and colors, passes them to a C function for
    downsampling, and returns the downsampled points and colors.

    :param points: Numpy array containing the coordinates of points in a point cloud.
                   Each row of the array represents a point in 3D space with its x, y, and z coordinates.
    :param colors: Numpy array containing the RGB of points in a point cloud.
                   Each row of the array represents a point in 3D space with its R, G, and B colors.
    :param leaf_size: Is the size of the leaf for the downsampling algorithm.

    :return: 'output_points' and 'output_colors' which contain the downsampled points and colors
             of the point cloud, respectively.
    """

    pcl_lib = ctypes.cdll.LoadLibrary("/home/adsec/blindhunter/grid_map/utils/ground_truth/build/libpcl_downsample.so")

    # To pass the numpy array to the C function
    ND_POINTER = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C")

    # Define the prototype of the function
    pcl_lib.pcl_downSample.argtypes = [ND_POINTER, ND_POINTER, ctypes.c_float, ctypes.c_size_t, ND_POINTER, ND_POINTER]
    pcl_lib.pcl_downSample.restype = ctypes.c_int

    # put the arrays in a contiguous memory to pass to the C function
    points = np.ascontiguousarray(points, dtype=np.float64)
    colors = np.ascontiguousarray(colors, dtype=np.float64)

    # To get the output points and colors of the downsampled point cloud on a clear array
    output_points = np.zeros((points.shape[0], 3)).astype(np.float64)
    output_colors = np.zeros((colors.shape[0], 3)).astype(np.float64)

    # Call the C function to downsample the point cloud
    pcl_lib.pcl_downSample(points, colors, ctypes.c_float(float(leaf_size)), ctypes.c_size_t(points.size // 3),
                           output_points, output_colors)

    # Delete the [0,0,0] points from the point cloud
    output_points = np.delete(output_points, np.where(np.all(output_points == [0, 0, 0], axis=1)), axis=0)
    output_colors = np.delete(output_colors, np.where(np.all(output_colors == [0, 0, 0], axis=1)), axis=0)

    return output_points, output_colors













