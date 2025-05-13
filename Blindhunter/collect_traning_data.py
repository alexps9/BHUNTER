import carla
import sys
import os
import math as mt
import random
import json
import copy
import queue
import argparse
import cv2
import numpy as np
import open3d as o3d
import time
import subprocess
from scipy.spatial import cKDTree
from grid_map.utils.setup import setup_world, environment
from grid_map.utils.spawn import spawn_sensor, spawn_vehicle
from grid_map.utils.ground_truth import ground_truth as ground_truth
from grid_map.utils.gennerate_traffic import gennerate_traffic
from grid_map.utils.spawn import sensor_resemble as sensor_resemble
from datetime import datetime
from math import pi
from queue import Empty
from matplotlib import cm
from carla_data_descriptor import CarlaDataDescriptor
import CMM_CARLA_Config as CFG
from numpy.linalg import pinv, inv
from CMM_CARLA_Config import *
from PIL import Image
import math
import logging

""" OUTPUT FOLDER GENERATION """
PHASE = "training4"
OUTPUT_FOLDER = os.path.join("/home/adsec/SimTrans/lyh/nas/kitti_carla/", PHASE)
folders = ['calib1', 'image_2', 'label_2', 'velodyne1', 'ImageSets', 'label_1', 'image_1', 'calib2','velodyne2','image_depth', 'planes']


def maybe_create_dir(path):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_distance(loc1, loc2):
    return math.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2 + (loc1.z - loc2.z)**2)

for folder in folders:
    directory = os.path.join(OUTPUT_FOLDER, folder)
    maybe_create_dir(directory)

""" DATA SAVE PATHS """
INDEX_PATH = os.path.join(OUTPUT_FOLDER, 'ImageSets')
LIDAR01_PATH = os.path.join(OUTPUT_FOLDER, 'velodyne1/{0:06}.bin')
LIDAR01_PATH_PLY = os.path.join(OUTPUT_FOLDER, 'velodyne1/{0:06}.ply')
LABEL01_PATH = os.path.join(OUTPUT_FOLDER, 'label_1/{0:06}.txt')
IMAGE01_PATH = os.path.join(OUTPUT_FOLDER, 'image_1/{0:06}.png')
CALIBRATION01_PATH = os.path.join(OUTPUT_FOLDER, 'calib1/{0:06}.txt')
GROUNDPLANE_PATH = os.path.join(OUTPUT_FOLDER, 'planes/{0:06}.txt')

LABEL02_PATH = os.path.join(OUTPUT_FOLDER, 'label_2/{0:06}.txt')
IMAGE02_PATH = os.path.join(OUTPUT_FOLDER, 'image_2/{0:06}.png')
CALIBRATION02_PATH = os.path.join(OUTPUT_FOLDER, 'calib2/{0:06}.txt')
LIDAR02_PATH = os.path.join(OUTPUT_FOLDER, 'velodyne2/{0:06}.bin')
LIDAR02_PATH_PLY = os.path.join(OUTPUT_FOLDER, 'velodyne2/{0:06}.ply')
DEPTH_IMAGE_PATH = os.path.join(OUTPUT_FOLDER, 'image_depth/{0:06}.png')


CP_DISTANCE = 51.2

weather_types = {
    "DayClear": {
        'precip_deposits': 0,
        'cloudiness': 0,
        'precipitation': 0,
        'sun_angle': 10,
        'fog': 0,
        'wetness': 5,
        'wind': 5,
        'air_pollution': 0,
        'dust_storm': 0
    },
    "DayCloudy": {
        'precip_deposits': 10,
        'cloudiness': 50,
        'precipitation': 10,
        'sun_angle': 10,
        'fog': 10,
        'wetness': 10,
        'wind': 10,
        'air_pollution': 0,
        'dust_storm': 0
    },
    "DayRain": {
        'precip_deposits': 70,
        'cloudiness': 80,
        'precipitation': 50,
        'sun_angle': 40,
        'fog': 0,
        'wetness': 70,
        'wind': 70,
        'air_pollution': 0,
        'dust_storm': 0
    },
    "NightCloudy": {
        'precip_deposits': 10,
        'cloudiness': 20,
        'precipitation': 10,
        'sun_angle': -10,
        'fog': 10,
        'wetness': 5,
        'wind': 5,
        'air_pollution': 0,
        'dust_storm': 0
    }
}


def server_settings(world):
    """
    Set the server settings for the given world object.

    Parameters:
    - world: the world object on which to apply the settings
    """
    settings = world.get_settings()
    # settings.no_rendering_mode = True  # No rendering mode
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.10
    world.apply_settings(settings)


def spawn_vehicle(world, blueprint, spawn_point: carla.Transform):
    # Get the blueprint for the vehicle - Tesla Model 3
    vehicle = world.try_spawn_actor(blueprint, spawn_point)
    if vehicle is None:
        raise Exception("Vehicle spawn failed: vehicle is None. Please check spawn point or blueprint.")

    vehicle.set_autopilot(True)

    return vehicle


def set_all_traffic_lights_to_green(world):
    traffic_lights = world.get_actors().filter('traffic.traffic_light')

    for traffic_light in traffic_lights:
        traffic_light.set_state(carla.TrafficLightState.Green)
        traffic_light.set_green_time(99999.0)
        traffic_light.freeze(True)

    print(f"All {len(traffic_lights)} traffic lights set to green.")


def spawn_npc(world, blueprint, spawn_point: carla.Transform):
    vehicle = world.try_spawn_actor(blueprint, spawn_point)
    if vehicle is None:
        raise Exception("Vehicle spawn failed: vehicle is None. Please check spawn point or blueprint.")

    return vehicle


def spawn_vehicle_route(world, blueprint, traffic_manager, route, weather_type, spawn_point, type_name: str):
    routes = []
    spawn_points = world.get_map().get_spawn_points()
    route_indices = route
    total_distance = 0
    for i in range(len(route_indices) - 1):
        loc1 = spawn_points[route_indices[i]].location
        loc2 = spawn_points[route_indices[i + 1]].location
        # print(f"Idx: {route_indices[i]} -> {route_indices[i+1]} = {loc1.distance(loc2)} meters")

        total_distance += loc1.distance(loc2)
    print(f"Total distance: {mt.ceil(total_distance)} meters")

    vehicle = spawn_vehicle(world, blueprint, spawn_points[route_indices[0]])
    print(f"{type_name} spawned!")

    traffic_manager.ignore_lights_percentage(vehicle, 100)
    traffic_manager.ignore_lights_percentage(vehicle, 0)
    traffic_manager.random_left_lanechange_percentage(vehicle, 0)
    traffic_manager.random_right_lanechange_percentage(vehicle, 0)
    traffic_manager.auto_lane_change(vehicle, False)
    for ind in route_indices:
        routes.append(spawn_points[ind].location)

    traffic_manager.set_path(vehicle, routes)

    if weather_type == "NightCloudy":
        traffic_manager.update_vehicle_lights(vehicle, True)

    return vehicle, spawn_points[route_indices[0]], spawn_points[route_indices[-1]]


def set_weather(world, weather_type: str):
    attrs = weather_types.get(weather_type, weather_types[weather_type])

    weather = carla.WeatherParameters(
        precipitation_deposits=attrs['precip_deposits'],
        cloudiness=attrs['cloudiness'],
        precipitation=attrs['precipitation'],
        sun_altitude_angle=attrs['sun_angle'],
        fog_density=attrs['fog'],
        wetness=attrs['wetness'],
        wind_intensity=attrs['wind'],
        mie_scattering_scale=attrs['air_pollution'],
        dust_storm=attrs['dust_storm']
    )

    world.set_weather(weather)
    print(f"Weather: {weather_type}")


def lidar_transformation(lidar_data, lidar):
    """
    Transforms raw lidar data into a point cloud and projects it into the world coordinate system,
    and reflects the points across the lidar's origin x, y coordinates in world space.

    :param lidar_data:
    :param lidar: The CARLA lidar sensor object to retrieve its transform for coordinate conversion.

    :return:
    1. 'lidar_pcl': An Open3D PointCloud object representing the lidar point cloud in the world coordinate system.
    2. 'center_lidar': A numpy array containing the coordinates of the lidar origin in the world coordinate system.
    """
    # Get the lidar transform matrix (lidar to world)
    transform = lidar.get_transform()
    lidar_to_world = np.array(transform.get_matrix())

    # Create an Open3D PointCloud object
    lidar_pcl = o3d.geometry.PointCloud()

    # Retrieve raw lidar data
    raw_data = lidar_data.raw_data

    # Convert raw lidar data to NumPy array
    point_cloud_array = np.frombuffer(raw_data, dtype=np.float32)
    point_cloud_array = point_cloud_array.reshape(-1, 6)[:, :4]
    point_cloud_array = np.delete(point_cloud_array, 3, 1)  # Remove intensity values, keep x, y, z

    # Apply coordinate transformation (x, y, z) → (y, x, -z)
    transformed_points = np.zeros_like(point_cloud_array)
    transformed_points[:, 0] = point_cloud_array[:, 0]  # y → x
    transformed_points[:, 1] = point_cloud_array[:, 1]  # x → y
    transformed_points[:, 2] = point_cloud_array[:, 2]  # -z → z

    # Transform lidar points to world coordinates
    # Add a column of 1s for homogeneous coordinates
    points_homogeneous = np.hstack((transformed_points, np.ones((transformed_points.shape[0], 1))))
    points_world = np.dot(lidar_to_world, points_homogeneous.T).T[:, :3]  # Transform and remove the homogeneous coordinate

    # Get the lidar's origin coordinates in world space
    origin_world = np.array([lidar.get_location().x, lidar.get_location().y, lidar.get_location().z])

    # Reflect points_world across the origin's x, y in world space
   # points_world[:, 0] = 2 * origin_world[0] - points_world[:, 0]  # Reflect x
   # points_world[:, 1] = 2 * origin_world[1] - points_world[:, 1]  # Reflect y

    # Assign transformed points to the Open3D PointCloud
    lidar_pcl.points = o3d.utility.Vector3dVector(points_world)

    # Get the lidar's origin in world coordinates (translation part of the transformation matrix)
    center_lidar = origin_world

    return lidar_pcl, center_lidar


def lidar_transformation_1(extrinsic, image_queue_lidar):
    """
    The function 'lidar_transformation' transforms raw lidar data into a point cloud, applies various
    rotations and translations to fit into ground truth point cloud. Turn into the world coordinates.

    :param extrinsic: Represents the extrinsic calibration matrix that describes the transformation between
                      the lidar sensor and the camera coordinate systems. It is used to calculate the rotation
                      around the Z-axis based on the ground truth camera data.
    :param image_queue_lidar: Is the queue that holds a byte array that contains the raw lidar data.

    :return: The function 'lidar_transformation' returns two values:
    1. 'lidar_pcl': An Open3D PointCloud object that represents the transformed lidar point cloud.
    2. 'center_lidar': A numpy array containing the coordinates of the origin of the lidar point cloud after transformation.
    """

    lidar_pcl = o3d.geometry.PointCloud()

    lidar_data = image_queue_lidar.get()
    raw_data = lidar_data.raw_data

    point_cloud_array = np.frombuffer(raw_data, dtype=np.float32)
    point_cloud_array = np.reshape(point_cloud_array, (int(point_cloud_array.shape[0] / 4), 4))
    point_cloud_array = np.delete(point_cloud_array, 3, 1)

    # Fix the lidar point cloud transformation to world coordinates
    yaw_90 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])  # Yaw = 90º
    point_cloud_array = np.dot(point_cloud_array, yaw_90)

    point_cloud_array[:, 2] = -point_cloud_array[:, 2]  # Z = -Z

    lidar_pcl.points = o3d.utility.Vector3dVector(point_cloud_array)

    point_cloud_color = np.full((len(lidar_pcl.points), 3), np.array([0, 0, 255]))  # All points are BLUE
    point_cloud_array = np.vstack([np.array(lidar_pcl.points), [0, 0, 0]])  # Add point 0,0,0 (origin)
    point_cloud_color = np.vstack([point_cloud_color, [255, 0, 0]])  # The (0,0,0) point is RED

    # Put the center of the point cloud in the origin
    centroid = np.mean(np.array(point_cloud_array), axis=0)
    point_cloud_array = point_cloud_array - centroid

    lidar_pcl.points = o3d.utility.Vector3dVector(point_cloud_array)
    lidar_pcl.colors = o3d.utility.Vector3dVector(point_cloud_color)

    center_lidar = point_cloud_array[-1]  # Get the origin coordinates of the lidar point cloud

    return lidar_pcl, center_lidar


""" def update_image(vis, image):
    # Convert OpenCV image to Open3D image
    open3d_img = o3d.geometry.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    vis.clear_geometries()
    vis.add_geometry(open3d_img)

    return False """


def get_ground_truth(queue_list, depth_camera_list):
    """
    The function 'get_ground_truth' processes depth images from multiple cameras to generate a point cloud.

    :param queue_list: Dictionary containing queues for different types of depth images.
    :param depth_camera_list: Dictionary containing depth camera objects for front, right, left, and back cameras.

    :return: The function `ground_truth` returns three values:
    1. `points`: A numpy array containing the 3D points in the world space for all four cameras.
    2. `colors`: A numpy array containing the color information (RGB) corresponding to each 3D point.
    3. `front_extrinsic_matrix`: The extrinsic matrix corresponding to the front camera.
    """

    front_depth_image = queue_list['image_queue_depth_front'].get()

    # Get the intrinsic and extrinsic matrix of the 4 cameras
    front_intrinsic_matrix, front_extrinsic_matrix = ground_truth.get_intrinsic_extrinsic_matrix(
        depth_camera_list['front_depth_camera'], front_depth_image)

    # Get the points [[X...], [Y...], [Z...]] and the colors [[R...], [G...], [B...]]
    front_points_3D, front_color = ground_truth.point2D_to_point3D(front_depth_image, front_intrinsic_matrix)

    # To multiply by the extrinsic matrix (same shape as the extrinsic_matrix matrix)
    front_p3d = np.concatenate((front_points_3D, np.ones((1, front_points_3D.shape[1]))))

    # Get the 3D points in the world
    front_p3d_world = np.dot(front_extrinsic_matrix, front_p3d)[:3]

    # Reshape the array to (height * width, 3) -> X, Y and Z for each point
    front_p3d_world = np.transpose(front_p3d_world)

    points = front_p3d_world
    colors = front_color

    # Put the center of the point cloud in the origin
    centroid = np.mean(points, axis=0)
    points = points - centroid

    return points, colors, front_extrinsic_matrix


def occupancy_grid_map(points, voxel_size=1, max_range_X_Y=360, min_range_Z=-6, max_range_Z=6):
    """
    A function that generates an occupancy grid map based on the input points, voxel size, and grid dimensions.
    It initializes the grid, converts the point cloud to voxel coordinates, and marks the occupied voxels.

    Parameters:
    - points: numpy array, representing the input points
    - voxel_size: float, the size of each voxel
    - max_range_X_Y: int, the maximum range in X and Y axes
    - min_range_Z: int, the minimum range in the Z axis
    - max_range_Z: float, the maximum range in the Z axis

    Returns:
    - occupancy_grid: numpy array, the final occupancy grid map
    """

    # Define grid bounds and size
    min_bound = np.array([-max_range_X_Y, -max_range_X_Y, min_range_Z])
    max_bound = np.array([max_range_X_Y, max_range_X_Y, max_range_Z])
    grid_size = np.ceil((max_bound - min_bound) / voxel_size).astype(int)

    # Initialize the occupancy grid
    occupancy_grid = np.zeros(grid_size, dtype=np.int8)

    # Convert point cloud to voxel coordinates and filter out-of-bounds indices
    voxel_indices = np.floor((points - min_bound) / voxel_size).astype(int)
    mask = (
            (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < grid_size[0]) &
            (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < grid_size[1]) &
            (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < grid_size[2])
    )
    voxel_indices = voxel_indices[mask]

    # Mark the voxels as occupied using unique indices
    unique_indices = np.unique(voxel_indices, axis=0)
    occupancy_grid[unique_indices[:, 0], unique_indices[:, 1], unique_indices[:, 2]] = 1

    return occupancy_grid


def ensure_directory_exists(directory_path):
    os.makedirs(directory_path, exist_ok=True)

"""
============ data collection ===========
"""

def save_groundplanes(planes_fname, player_measurements, lidar_height:float=1.6):
    from math import cos, sin
    """ Saves the groundplane vector of the current frame.
        The format of the ground plane file is first three lines describing the file (number of parameters).
        The next line is the three parameters of the normal vector, and the last is the height of the normal vector,
        which is the same as the distance to the camera in meters.
    """
    ego_transform = player_measurements.get_transform()
    rotation = ego_transform.rotation
    pitch, roll = rotation.pitch, rotation.roll
    # Since measurements are in degrees, convert to radians
    pitch = degrees_to_radians(pitch)
    roll = degrees_to_radians(roll)
    # Rotate normal vector (y) wrt. pitch and yaw
    normal_vector = [cos(pitch)*sin(roll),
                     -cos(pitch)*cos(roll),
                     sin(pitch)
                     ]
    normal_vector = map(str, normal_vector)
    with open(planes_fname, 'w') as f:
        f.write("# Plane\n")
        f.write("Width 4\n")
        f.write("Height 1\n")
        f.write("{} {}\n".format(" ".join(normal_vector), lidar_height))
    logging.info("Wrote plane data to %s", planes_fname)



def isInRoadsideRange(location):

    locationX = location.x
    locationY = location.y
    minX = CFG.Lidar['LocationX'] - 51.2
    maxX = CFG.Lidar['LocationX'] + 51.2
    minY = CFG.Lidar['LocationY'] - 51.2
    maxY = CFG.Lidar['LocationY'] + 51.2

    if locationX >= minX and locationX<= maxX and locationY >= minY and locationY <= maxY:
        inRangeFlag = True
    else:
        inRangeFlag = False

    return inRangeFlag

def isInRange(location, sensor_location):

    locationX = location.x
    locationY = location.y
    minX = sensor_location.x - 51.2
    maxX = sensor_location.x + 51.2
    minY = sensor_location.y - 51.2
    maxY = sensor_location.y + 51.2

    if locationX >= minX and locationX<= maxX and locationY >= minY and locationY <= maxY:
        inRangeFlag = True
    else:
        inRangeFlag = False

    return inRangeFlag


def dis2sensor(actor_location):
    dis = mt.sqrt(np.square(actor_location.x - CFG.Lidar['LocationX']) + np.square(actor_location.y - CFG.Lidar['LocationY']))

    return round(dis, 2)


def creat_kitti_datapoint(actor, sensor, actor_type, cur_N):
    if actor:
        datapoint = CarlaDataDescriptor()
        bbox_2d = [0, 0, 0, 0] # no camera so far

        rotation_y = get_relative_rotation_y(actor.get_transform().rotation.yaw, sensor.rotation.yaw)
        # id = actor.id
        datapoint.set_bbox(bbox_2d)
        datapoint.set_3d_object_dimensions(actor.bounding_box.extent)
        datapoint.set_type(actor_type)
        datapoint.set_3d_object_location(actor, sensor, actor_type)
        datapoint.set_rotation_y(rotation_y)
        datapoint.set_id(actor.id)

        if cur_N is None:
            datapoint.set_occlusion(3)
            datapoint.occluded = 3
        elif cur_N > 100:
            datapoint.set_occlusion(0)
            datapoint.occluded = 0
        elif cur_N < 5:
            datapoint.set_occlusion(2)
            datapoint.occluded = 2
        else:
            datapoint.set_occlusion(1)
            datapoint.occluded = 1
        '''
        truncation = float(1- cur_N/15) if cur_N < 15 else 0
        datapoint.set_truncated(truncation)
        '''
        # print(actor)
        # print(datapoint)
        return datapoint
    else:
        return None


def get_relative_rotation_y(actor_yaw, sensor_yaw):
    """ Returns the relative rotation of the agent to the camera in yaw
    The relative rotation is the difference between the camera rotation (on car) and the agent rotation"""

    rot_actor = degrees_to_radians(actor_yaw) # -180 ~ 180 wst clockwise
    # rot_actor = -1 * rot_actor # 180 ~ -180 wst clockwise
    rot_sensor = degrees_to_radians(sensor_yaw)
    rot_y_lidar = rot_actor - rot_sensor # rt wrt lidar coordinate
    rot_y_camera = rot_y_lidar - 0.5*pi
    if rot_y_camera < -1.0*pi:
        rot_y_camera = rot_y_camera + 2.0 * pi
    if rot_y_camera > 1.0*pi:
        rot_y_camera -= 2.0*pi
    # print('vehicle rotation :{}'.format(rot_y_camera))
    # the difference of the x-axis direction between carla and kitti
    kitti_ry = round(rot_y_camera, 2)

    return kitti_ry


def degrees_to_radians(degrees):
    return np.round(degrees * mt.pi / 180.0, 2)


def save_kitti_label_data(filename, datapoints):
    with open(filename, 'w') as f:
        out_str = "\n".join([str(point) for point in datapoints if point])
        f.write(out_str)
    # logging.info("Wrote kitti label data to %s", filename)



def proj_to_camera(pos_vector):
    # transform the points to camera
    TR_velodyne = np.array([[0, -1, 0],
                            [0, 0, -1],
                            [1, 0, 0]])
    # Add translation vector from velo to camera. This is 0 because the position of camera and lidar is equal in our configuration.

    transformed_3d_pos = np.dot(TR_velodyne, pos_vector)
    return transformed_3d_pos


def save_lidar_data(filename, lidar_measurement, format="bin"):
    """ Saves lidar data to given filename, according to the lidar data format.
        bin is used for KITTI-data format, while .ply is the regular point cloud format
        In Unreal, the coordinate system of the engine is defined as, which is the same as the lidar points
        z
        ^   ^ x
        |  /
        | /
        |/____> y
        This is a left-handed coordinate system, with x being forward, y to the right and z up
        See also https://github.com/carla-simulator/carla/issues/498
        However, the lidar coordinate system from KITTI is defined as
              z
              ^   ^ x
              |  /
              | /
        y<____|/
        Which is a right handed coordinate sylstem
        Therefore, we need to flip the y axis of the lidar in order to get the correct lidar format for kitti.

        This corresponds to the following changes from Carla to Kitti
            Carla: X   Y   Z
            KITTI: X  -Y   Z
        NOTE: We do not flip the coordinate system when saving to .ply.
    """
    # logging.info("Wrote lidar data to %s", filename)

    if format == "bin":
        data = np.copy(np.frombuffer(lidar_measurement.raw_data, dtype=np.float32))
        data = data.reshape(-1, 6)[:, :4]

        data[:, 1] = -1 * data[:, 1]
        lidar_array = np.array(data).astype(np.float32)
        # logging.debug("Lidar min/max of x: {} {}".format(
        #               lidar_array[:, 0].min(), lidar_array[:, 0].max()))
        # logging.debug("Lidar min/max of y: {} {}".format(
        #               lidar_array[:, 1].min(), lidar_array[:, 0].max()))
        # logging.debug("Lidar min/max of z: {} {}".format(
        #               lidar_array[:, 2].min(), lidar_array[:, 0].max()))
        lidar_array.tofile(filename)
    else:
        lidar_measurement.save_to_disk(filename)


def save_calibration_matrices(sensor_data, filename, intrinsic_mat,  lidar_cam_mat):
    """ Saves the calibration matrices to a file.
        AVOD (and KITTI) refers to P as P=K*[R;t], so we will just store P.
        The resulting file will contain:
        3x4    p0-p3      Camera P matrix. Contains extrinsic
                          and intrinsic parameters. (P=K*[R;t])
        3x3    r0_rect    Rectification matrix, required to transform points
                          from velodyne to camera coordinate frame.
        3x4    tr_velodyne_to_cam    Used to transform from velodyne to cam
                                     coordinate frame according to:
                                     Point_Camera = P_cam * R0_rect *
                                                    Tr_velo_to_cam *
                                                    Point_Velodyne.
        3x4    tr_imu_to_velo        Used to transform from imu to velodyne coordinate frame. This is not needed since we do not export
                                     imu data.
    """
    """
        in this carla dataset the calibration matrices are set as follows
        3x4 P0: 1 0 0 0 0 1 0 0 0 0 1 0
        3x4 P1: 1 0 0 0 0 1 0 0 0 0 1 0
        3x4 P2: 1 0 0 0 0 1 0 0 0 0 1 0
        3x4 P3: 1 0 0 0 0 1 0 0 0 0 1 0
        3x3 R0_rect: 1 0 0 0 1 0 0 0 1
        3x4 Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0
        3x4 Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0
    """
    # KITTI format demands that we flatten in row-major order
    ravel_mode = 'C'
    P0 = intrinsic_mat
    P0 = np.column_stack((P0, np.array([0, 0, 0])))
    P0 = np.ravel(P0, order=ravel_mode)
    R0 = np.identity(3)
    R_velodyne = np.array([[0, -1, 0],
                            [0, 0, -1],
                            [1, 0, 0]])

    # Add translation vector from velo to camera.
    T_velodyne = np.array([lidar_cam_mat[1, 3], -lidar_cam_mat[2, 3], lidar_cam_mat[0, 3]])
    TR_velodyne = np.column_stack((R_velodyne, T_velodyne))
    TR_imu_to_velo = np.identity(3)
    TR_imu_to_velo = np.column_stack((TR_imu_to_velo, np.array([0, 0, 0])))
    """
    A new calibration matrix for cooperative perception:
    sensor location and pose (SLaP)
    [x, y, z, pitch, yaw, roll]
    """
    SLaP = np.array([sensor_data.transform.location.x, sensor_data.transform.location.y, sensor_data.transform.location.z,
                    sensor_data.transform.rotation.pitch, sensor_data.transform.rotation.yaw, sensor_data.transform.rotation.roll])


    def write_flat(f, name, arr):
        f.write("{}: {}\n".format(name, ' '.join(
            map(str, arr.flatten(ravel_mode).squeeze()))))

    # All matrices are written on a line with spacing
    with open(filename, 'w') as f:
        for i in range(4):  # Avod expects all 4 P-matrices even though we only use the first
            write_flat(f, "P" + str(i), P0)
        write_flat(f, "R0_rect", R0)
        write_flat(f, "Tr_velo_to_cam", TR_velodyne)
        write_flat(f, "TR_imu_to_velo", TR_imu_to_velo)
        write_flat(f, "SLaP", SLaP)


    # logging.info("Wrote all calibration matrices to %s", filename)


def save_index_data(OUTPUT_FOLDER, id):
    """ Appends the id of the given record to the files """
    for name in ['train.txt', 'val.txt', 'trainval.txt']:
        path = os.path.join(OUTPUT_FOLDER, name)
        with open(path, 'a') as f:
            f.write("{0:06}".format(id) + '\n')
        # logging.info("Wrote reference files to %s", path)

# Sensor callback.
# This is where you receive the sensor data and
# process it as you liked and the important part is that,
# at the end, it should include an element into the sensor queue.

'''
def lidar_sensor_callback(sensor_data, world, sensor_queue, sensor_name, vehicle):
    # Do stuff with the sensor_data data like save it to disk
    # Then you just need to add to the queue
    frame = sensor_data.frame
    sensor_location = vehicle.get_location()
    dis2Roadside = dis2sensor(sensor_location)

    if frame%5 == 0 and dis2Roadside <= CP_DISTANCE:
        actor_list = world.get_actors()
        vehicle_list = actor_list.filter('vehicle.*')
        pedestrians_list = actor_list.filter('walker.pedestrian.*')
        # print('vehicle_list length: ', len(actor_list))
        lidar_list = actor_list.filter('sensor.lidar.ray_cast')
        # print('lidar_list: ', lidar.get_location())
        label_data = []
        for actor in vehicle_list:
            if isInRange(actor.get_location(), sensor_location):
                kitti_datapoint = creat_kitti_datapoint(actor, sensor_data.transform, 'Car')
                if kitti_datapoint:
                    label_data.append(kitti_datapoint)

        for actor in pedestrians_list:
            if isInRange(actor.get_location(), sensor_location):
                kitti_datapoint = creat_kitti_datapoint(actor, sensor_data.transform, 'Pedestrian')
                if kitti_datapoint:
                    label_data.append(kitti_datapoint)

        # print('label_data', label_data)
        # print('frame: ', frame)

        if sensor_name == "lidar01":
            label_filename = LABEL01_PATH.format(frame)
            calib_filename = CALIBRATION01_PATH.format(frame)
            if LIDAR_DATA_FORMAT == "bin":
                lidar_filename = LIDAR01_PATH.format(frame)
            else:
                lidar_filename = LIDAR01_PATH_PLY.format(frame)

        if sensor_name == "lidar02":
            label_filename = LABEL02_PATH.format(frame)
            calib_filename = CALIBRATION02_PATH.format(frame)
            if LIDAR_DATA_FORMAT == "bin":
                lidar_filename = LIDAR02_PATH.format(frame)
            else:
                lidar_filename = LIDAR02_PATH_PLY.format(frame)

        # print('label_filename', )
        save_kitti_label_data(label_filename, label_data)
        save_calibration_matrices(sensor_data, calib_filename)
        # save_index_data(INDEX_PATH, frame)
        save_lidar_data(lidar_filename, sensor_data, LIDAR_DATA_FORMAT)

    sensor_queue.put((sensor_data.frame, sensor_name))

def roadside_lidar_sensor_callback(sensor_data, world, sensor_queue, sensor_name, vehicle):
    # Do stuff with the sensor_data data like save it to disk
    # Then you just need to add to the queue
    frame = sensor_data.frame
    sensor_location = vehicle.get_location()
    dis2Roadside = dis2sensor(sensor_location)
    if frame%5 == 0 and dis2Roadside <= CP_DISTANCE:
        actor_list = world.get_actors()
        vehicle_list = actor_list.filter('vehicle.*')
        pedestrians_list = actor_list.filter('walker.pedestrian.*')
        # print('vehicle_list length: ', len(actor_list))
        lidar_list = actor_list.filter('sensor.lidar.ray_cast')
        # print('lidar_list: ', lidar.get_location())
        label_data = []
        for actor in vehicle_list:
            if isInRoadsideRange(actor.get_location()):
                kitti_datapoint = creat_kitti_datapoint(actor, sensor_data.transform, 'Car')
                if kitti_datapoint:
                    label_data.append(kitti_datapoint)

        for actor in pedestrians_list:
            if isInRoadsideRange(actor.get_location()):
                kitti_datapoint = creat_kitti_datapoint(actor, sensor_data.transform, 'Pedestrian')
                if kitti_datapoint:
                    label_data.append(kitti_datapoint)

        # print('label_data', label_data)
        # print('frame: ', frame)

        if sensor_name == "lidar01":
            label_filename = LABEL01_PATH.format(frame)
            calib_filename = CALIBRATION01_PATH.format(frame)
            if LIDAR_DATA_FORMAT == "bin":
                lidar_filename = LIDAR01_PATH.format(frame)
            else:
                lidar_filename = LIDAR01_PATH_PLY.format(frame)

        # print('label_filename', )
        save_kitti_label_data(label_filename, label_data)
        save_calibration_matrices(sensor_data, calib_filename)
        save_index_data(INDEX_PATH, frame)
        save_lidar_data(lidar_filename, sensor_data, LIDAR_DATA_FORMAT)

    sensor_queue.put((sensor_data.frame, sensor_name))

def cam_sensor_callback(sensor_data,  world, sensor_queue, sensor_name, vehicle):
    # Do stuff with the sensor_data data like save it to disk
    # Then you just need to add to the queue
    frame = sensor_data.frame
    sensor_location = vehicle.get_location()
    dis2Roadside = dis2sensor(sensor_location)
    if sensor_name == "cam01":
        img_filename = IMAGE01_PATH.format(frame)

    if sensor_name == "cam02":
        img_filename = IMAGE02_PATH.format(frame)
    if frame%5 == 0 and dis2Roadside <= CP_DISTANCE:
        sensor_data.save_to_disk(img_filename)
    sensor_queue.put((sensor_data.frame, sensor_name))
'''

def lidar_sensor_callback(sensor_data, world, sensor_queue, sensor_name, vehicle):
    frame = sensor_data.frame
    actor_list = world.get_actors()
    vehicle_list = actor_list.filter('vehicle.*')
    pedestrians_list = actor_list.filter('walker.pedestrian.*')

    label_data = []
    ego_vehicle_id = vehicle.id

    for actor in vehicle_list:
        if actor.id == ego_vehicle_id:
            continue
        if isInRange(actor.get_location(), vehicle.get_location()):
            kitti_datapoint = creat_kitti_datapoint(actor, sensor_data.transform, 'Car')
            if kitti_datapoint:
                label_data.append(kitti_datapoint)

    for actor in pedestrians_list:
        if isInRange(actor.get_location(), vehicle.get_location()):
            kitti_datapoint = creat_kitti_datapoint(actor, sensor_data.transform, 'Pedestrian')
            if kitti_datapoint:
                label_data.append(kitti_datapoint)

    label_filename = LABEL01_PATH.format(frame)

    if LIDAR_DATA_FORMAT == "bin":
        lidar_filename = LIDAR01_PATH.format(frame)
    else:
        lidar_filename = LIDAR01_PATH_PLY.format(frame)

    save_kitti_label_data(label_filename, label_data)

    save_lidar_data(lidar_filename, sensor_data, LIDAR_DATA_FORMAT)

    sensor_queue.put((sensor_data.frame, sensor_name))

def _depth_to_array(image):
    """
    Convert an image containing CARLA encoded depth-map to a 2D array containing
    the depth value of each pixel in meters.
    """
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))

    array = array.astype(np.float32)
    normalized_depth = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)

    max_depth = 1000.0
    depth_in_meters = normalized_depth * max_depth

    return depth_in_meters


def cam_sensor_callback(sensor_data, world, sensor_queue, sensor_name, vehicle, camera, lidar, ego):
    frame = sensor_data.frame
    '''
    calib_filename = CALIBRATION01_PATH.format(frame)
    plane_filename = GROUNDPLANE_PATH.format(frame)

    actor_list = world.get_actors()
    vehicle_list = actor_list.filter('vehicle.*')
    pedestrians_list = actor_list.filter('walker.pedestrian.*')

    ego_vehicle_id = vehicle.id

    label_data = []

    for actor in vehicle_list:
        if actor.id == ego_vehicle_id:
            continue
        if isInRange(actor.get_location(), vehicle.get_location()):
            kitti_datapoint = creat_kitti_datapoint(actor, sensor_data.transform, 'Car')
            if kitti_datapoint:
                label_data.append(kitti_datapoint)

    for actor in pedestrians_list:
        if isInRange(actor.get_location(), vehicle.get_location()):
            kitti_datapoint = creat_kitti_datapoint(actor, sensor_data.transform, 'Pedestrian')
            if kitti_datapoint:
                label_data.append(kitti_datapoint)

    camera_intrinsic = ground_truth.get_intrinsic_matrix(camera)
    cam_transform = camera.get_transform()
    lidar_transform = lidar.get_transform()
    veh_cam_mat = cam_transform.get_inverse_matrix()
    lidar_veh_mat = lidar_transform.get_matrix()
    lidar_to_cam_matrix = np.dot(veh_cam_mat, lidar_veh_mat)
    save_groundplanes(plane_filename, ego)
    save_calibration_matrices(sensor_data, calib_filename, camera_intrinsic, lidar_to_cam_matrix)
    label_filename = LABEL01_PATH.format(frame)
    save_kitti_label_data(label_filename, label_data)
    '''
    if sensor_name == "cam01":
        img_filename = IMAGE01_PATH.format(frame)

        sensor_data.save_to_disk(img_filename)


    elif sensor_name == "cam02":
        depth_array = _depth_to_array(sensor_data)
        depth_filename = IMAGE02_PATH.format(frame)
        np.save(depth_filename, depth_array)
        depth_visualization = (depth_array * 255).astype(np.uint8)
        depth_image_filename = DEPTH_IMAGE_PATH.format(frame)
        Image.fromarray(depth_visualization).save(depth_image_filename)
        print(f"Depth data saved for frame {frame}")

    elif sensor_name == "lidar01":
        if LIDAR_DATA_FORMAT == "bin":
            lidar_filename = LIDAR01_PATH.format(frame)
        else:
            lidar_filename = LIDAR01_PATH_PLY.format(frame)
        save_lidar_data(lidar_filename, sensor_data, LIDAR_DATA_FORMAT)

    sensor_queue.put((sensor_data.frame, sensor_name))


def generate_lidar_bp(arg, world, blueprint_library):
    """Generates a CARLA blueprint based on the script parameters"""
    if arg.semantic:
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
    else:
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        if arg.no_noise:
            lidar_bp.set_attribute('dropoff_general_rate', '0.0')
            lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
            lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
        else:

            lidar_bp.set_attribute('noise_stddev', '0.01')
            # lidar_bp.set_attribute('dropoff_general_rate', '0.05')
            # lidar_bp.set_attribute('dropoff_intensity_limit', '0.9')
            # lidar_bp.set_attribute('dropoff_zero_intensity', '0.1')

    lidar_bp.set_attribute('upper_fov', str(arg.upper_fov))
    lidar_bp.set_attribute('lower_fov', str(arg.lower_fov))
    lidar_bp.set_attribute('channels', str(arg.channels))
    # lidar_bp.set_attribute('atmosphere_attenuation_rate', str(0.05))
    lidar_bp.set_attribute('range', str(arg.range))
    lidar_bp.set_attribute('rotation_frequency', str(10))
    lidar_bp.set_attribute('points_per_second', str(arg.points_per_second))
    return lidar_bp

def downsample_within_bbox(fused_pcl: o3d.geometry.PointCloud, candidate: carla.Actor, voxel_size=0.2, scale=1.0):
    """
    Filters points in the Open3D PointCloud object that are inside the candidate bounding box and applies downsampling.
    This method projects the 3D object to the sensor's facing direction and uses the 2D shape's area for visibility.

    :param fused_pcl: Open3D PointCloud object representing the fused point cloud
    :param candidate: CARLA Actor object with a bounding box
    :param sensor_transform: Transform of the sensor, indicating its position and orientation
    :param voxel_size: Voxel size for downsampling
    :return: Downsampled Open3D PointCloud object containing points inside the candidate bounding box
    """
    # Get bounding box of the candidate in local coordinates
    bbox = candidate.bounding_box
    bb_extent = np.array([bbox.extent.x, bbox.extent.y, bbox.extent.z]) * scale
    bb_center = np.array([bbox.location.x, bbox.location.y, bbox.location.z])

    # Transform bounding box center to world coordinates
    candidate_transform = candidate.get_transform()
    bb_center_world = candidate_transform.transform(carla.Location(*bb_center))

    # Convert bb_center_world to NumPy array
    bb_center_world_np = np.array([bb_center_world.x, bb_center_world.y, bb_center_world.z])

    # Extract points from the Open3D PointCloud
    points = np.asarray(fused_pcl.points)

    # Filter points inside the bounding box
    filtered_points = []
    for point in points:
        # Transform point to bounding box local coordinates
        relative_point = np.dot(
            np.array(candidate_transform.get_inverse_matrix())[:3, :3],  # Use rotation matrix
            point - bb_center_world_np,
        )
        if (
                -bb_extent[0] <= relative_point[0] <= bb_extent[0] and
                -bb_extent[1] <= relative_point[1] <= bb_extent[1] and
                -bb_extent[2] <= relative_point[2] <= bb_extent[2]
        ):
            filtered_points.append(point)

    # Convert filtered points back to Open3D PointCloud
    filtered_pcl = o3d.geometry.PointCloud()
    filtered_pcl.points = o3d.utility.Vector3dVector(filtered_points)

    # Apply voxel downsampling
    downsampled_pcl = filtered_pcl.voxel_down_sample(voxel_size=voxel_size)

    return downsampled_pcl

class CarlaRunner:
    # run the sim based on a seed
    def __init__(self):
        self.client = None
        self.world = None
        self.actor_list = []
        self.npc_list = []
        self.sensor_list = []
        self.ego = None
        self.candidate = None
        self.traffic_manager = None
        self.blueprint_library = None
        self._seed = None
        self.weather_type = None
        self.controllers_list = []
        self.frame = 2000
        self.mutated_frame = 0
        self._score = 100
        self.spectator = None
        self.candidate_location = None
        self.ego_location = None

    def run_sim(self, seed):
        self._seed = seed
        # set client
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(30.0)
        # set weather and map
        environment = seed["environment"]
        weather_type = environment["weather_type"]
        self.weather_type = weather_type
        map = environment["map"]
        print(map)
        world = self.client.load_world(map)
        self.world = world
        server_settings(world)
        set_weather(world, weather_type)
        self.blueprint_library = world.get_blueprint_library()
        self.traffic_manager = self.client.get_trafficmanager()
        set_all_traffic_lights_to_green(world)

        # set ego
        ego = seed["ego"]
        ego_route = []
        for _t in ego["route"]:
            point = _t["point"]
            ego_route.append(point)
        ego_spawn_point = carla.Transform(
            carla.Location(x=ego["spawn_point"]["x"], y=ego["spawn_point"]["y"], z=ego["spawn_point"]["z"]),
            carla.Rotation(pitch=ego["spawn_point"]["pitch"], yaw=ego["spawn_point"]["yaw"],
                           roll=ego["spawn_point"]["roll"]))
        ego_bp = self.blueprint_library.find(ego["type"])
        self.ego, ego_start_point, ego_end_point = spawn_vehicle_route(world, ego_bp, self.traffic_manager, ego_route,
                                                                       weather_type, ego_spawn_point, "ego")
        self.actor_list.append(self.ego)
        image_queue_depth = queue.Queue()
        image_queue_lidar = queue.Queue()
        image_queue_rgb = queue.Queue()

        # set sensors
        sensor_config = seed["Sensor"]
        self.sensor_list = sensor_resemble.setup_vehicle_sensors(world, self.ego, sensor_config)
        for sensor in self.sensor_list:
            if sensor.type_id == 'sensor.camera.rgb':
                camera_rgb = sensor
                camera_rgb.listen(image_queue_rgb.put)
            elif sensor.type_id == 'sensor.camera.depth':
                camera_depth = sensor
                camera_depth.listen(image_queue_depth.put)
            elif sensor.type_id == 'sensor.lidar.ray_cast':
                camera_lidar = sensor
                camera_lidar.listen(image_queue_lidar.put)
            else:
                print('Unknown sensor type ')

            print(sensor.type_id)

        # set candidate
        candidate = seed["candidate"]
        candidate_route = []

        for _t in candidate["route"]:
            point = _t["point"]
            candidate_route.append(point)

        candidate_spawn_point = carla.Transform(
            carla.Location(x=candidate["spawn_point"]["x"], y=candidate["spawn_point"]["y"],
                           z=candidate["spawn_point"]["z"]),
            carla.Rotation(pitch=candidate["spawn_point"]["pitch"], yaw=candidate["spawn_point"]["yaw"],
                           roll=candidate["spawn_point"]["roll"]))

        candidate_bp = self.blueprint_library.find(candidate["type"])
        self.candidate, candidate_start_point, candidate_end_point = spawn_vehicle_route(world, candidate_bp,
                                                                                         self.traffic_manager,
                                                                                         candidate_route, weather_type,
                                                                                         candidate_spawn_point,
                                                                                         "candidate")
        self.actor_list.append(self.candidate)

        # set npcs
        npcs = seed["npcs"]
        for npc_data in npcs:
            npc_name = next(iter(npc_data))

            npc_spawn_point = carla.Transform(
                carla.Location(
                    x=npc_data[npc_name]["spawn_point"]["x"],
                    y=npc_data[npc_name]["spawn_point"]["y"],
                    z=npc_data[npc_name]["spawn_point"]["z"]
                ),
                carla.Rotation(
                    pitch=npc_data[npc_name]["spawn_point"]["pitch"],
                    yaw=npc_data[npc_name]["spawn_point"]["yaw"],
                    roll=npc_data[npc_name]["spawn_point"]["roll"]
                )
            )

            npc_type = self.blueprint_library.find(npc_data[npc_name]["type"])
            npc_instance = spawn_npc(world, npc_type, npc_spawn_point)

            npc_trigger_time = npc_data[npc_name]["triggering_time"]

            self.npc_list.append((npc_trigger_time, npc_instance))

        self.npc_list.sort(key=lambda x: x[0])
        pcl_downsampled = o3d.geometry.PointCloud()
        frame = 0

        # sim loop
        while frame < self.frame:
            world.tick()

            frame += 1

            # set spectator
            spectator = world.get_spectator()
            ego_transform = self.ego.get_transform()
            spectator_location = ego_transform.location + carla.Location(z=5)
            spectator_rotation = ego_transform.rotation
            spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))

            for npc in self.npc_list:
                if npc[0] == frame:
                    npc[1].set_autopilot(True)
                    self.traffic_manager.ignore_lights_percentage(npc[1], 100)
                    self.traffic_manager.random_left_lanechange_percentage(npc[1], 0)
                    self.traffic_manager.random_right_lanechange_percentage(npc[1], 0)
                    self.traffic_manager.auto_lane_change(npc[1], False)

            queue_list = {
                "image_queue_depth_front": image_queue_depth,
                "image_queue_rgb": image_queue_rgb
            }

            depth_camera_list = {"front_depth_camera": camera_depth}

            points, colors, extrinsic = get_ground_truth(queue_list, depth_camera_list)

            downsampled_points, downsampled_colors = ground_truth.downsample(points, colors, 0.2)
            # downsampled_points, downsampled_colors = points,colors
            '''
            # Get the center of the 4 ground truth cameras (Red points)
            red_indices = np.where(downsampled_colors[:, 0] == 255)[0]
            red_center_points = downsampled_points[red_indices]
            # Get the center of the red points (Coords of the cameras)
            groundtruth_center = np.mean(red_center_points, axis=0)

            # Add the center of the point cloud (RED)
            pcl_downsampled.points = o3d.utility.Vector3dVector(np.vstack([downsampled_points, groundtruth_center]))
            pcl_downsampled.colors = o3d.utility.Vector3dVector(np.vstack([downsampled_colors, [255, 0, 0]]))

            # LIDAR TRANSFORMATION
            lidar_pcl, center_lidar = lidar_transformation(extrinsic, image_queue_lidar)

            # Fit the lidar point cloud to the ground truth point cloud
            translation_to_center = center_lidar - groundtruth_center
            pcl_downsampled.points = o3d.utility.Vector3dVector(
                np.array(pcl_downsampled.points) + translation_to_center)

            # DELETE THE RED POINTS
            ground_truth_red_indices = np.where(np.asarray(pcl_downsampled.colors)[:, 0] == 255)[0]
            ground_truth_points = np.delete(np.asarray(pcl_downsampled.points), ground_truth_red_indices, axis=0)
            pcl_downsampled.points = o3d.utility.Vector3dVector(ground_truth_points)
            pcl_downsampled.colors = o3d.utility.Vector3dVector([])

            lidar_red_indices = np.where(np.asarray(lidar_pcl.colors)[:, 0] == 255)[0]
            lidar_points = np.delete(np.asarray(lidar_pcl.points), lidar_red_indices, axis=0)
            lidar_pcl.points = o3d.utility.Vector3dVector(lidar_points)
            lidar_pcl.colors = o3d.utility.Vector3dVector([])

            fused_pointcloud = np.vstack((np.asarray(pcl_downsampled.points), np.asarray(lidar_pcl.points)))
            fused_pcl = o3d.geometry.PointCloud()
            fused_pcl.points = o3d.utility.Vector3dVector(fused_pointcloud)

            # Voxel occupancy grid
            fused_voxel_occupancy_grid = occupancy_grid_map(fused_pointcloud)
            '''
            # Get the center of the 4 ground truth cameras (Red points)
            red_indices = np.where(downsampled_colors[:, 0] == 255)[0]
            red_center_points = downsampled_points[red_indices]
            # Get the center of the red points (Coords of the cameras)
            groundtruth_center = np.mean(red_center_points, axis=0)

            # Add the center of the point cloud (RED)
            pcl_downsampled.points = o3d.utility.Vector3dVector(np.vstack([downsampled_points, groundtruth_center]))
            pcl_downsampled.colors = o3d.utility.Vector3dVector(np.vstack([downsampled_colors, [255, 0, 0]]))

            # LIDAR TRANSFORMATION
            lidar_pcl, center_lidar = lidar_transformation(extrinsic, image_queue_lidar)

            # Fit the lidar point cloud to the ground truth point cloud
            translation_to_center = center_lidar - groundtruth_center
            pcl_downsampled.points = o3d.utility.Vector3dVector(
                np.array(pcl_downsampled.points) + translation_to_center)
            # lidar_pcl.points = o3d.utility.Vector3dVector(np.array(pcl_downsampled.points) - translation_to_center)

            # DELETE THE RED POINTS
            ground_truth_red_indices = np.where(np.asarray(pcl_downsampled.colors)[:, 0] == 255)[0]
            ground_truth_points = np.delete(np.asarray(pcl_downsampled.points), ground_truth_red_indices, axis=0)
            pcl_downsampled.points = o3d.utility.Vector3dVector(ground_truth_points)
            pcl_downsampled.colors = o3d.utility.Vector3dVector([])

            lidar_red_indices = np.where(np.asarray(lidar_pcl.colors)[:, 0] == 255)[0]
            lidar_points = np.delete(np.asarray(lidar_pcl.points), lidar_red_indices, axis=0)
            lidar_pcl.points = o3d.utility.Vector3dVector(lidar_points)
            lidar_pcl.colors = o3d.utility.Vector3dVector([])

            fused_pointcloud = np.vstack((np.asarray(pcl_downsampled.points), np.asarray(lidar_pcl.points)))
            # fused_pointcloud =np.asarray(lidar_pcl.points)
            fused_pcl = o3d.geometry.PointCloud()
            fused_pcl.points = o3d.utility.Vector3dVector(fused_pointcloud)

            # Voxel occupancy grid
            fused_voxel_occupancy_grid = occupancy_grid_map(fused_pcl.points)

            # Compute visibility score
            candidate_location = np.array([
                self.candidate.get_location().x,
                self.candidate.get_location().y,
                self.candidate.get_location().z
            ])
            translation_offset = center_lidar - candidate_location
            cur_score, candidate_voxel_grid = self.compute_visibility_score(fused_voxel_occupancy_grid,
                                                                            translation_offset)
            if cur_score < self._score:
                self._score = cur_score
                self.mutated_frame = frame
                self.candidate_location = self.candidate.get_location()
                self.ego_location = self.ego.get_location()

            # stop route
            ego_location = ego_transform.location
            end_location = ego_end_point.location
            distance_to_end = ego_location.distance(end_location)
            if distance_to_end < 1:
                self.traffic_manager.vehicle_percentage_speed_difference(ego, 100)
                self.traffic_manager.vehicle_percentage_speed_difference(candidate, 100)
                break
            image = image_queue_rgb.get()
            ensure_directory_exists('_out/rgb/')
            ensure_directory_exists('_out/fused_pointcloud/')
            ensure_directory_exists('_out/fused_voxel_grid/')
            ensure_directory_exists('_out/candidate_voxel_grid/')
            image.save_to_disk('_out/rgb/' + time.strftime('%Y%m%d_%H%M%S') + '_%06d' % image.frame + '.png')
            # Save the fused point cloud
            o3d.io.write_point_cloud(
                f'./_out/fused_pointcloud/' + time.strftime('%Y%m%d_%H%M%S') + '_%06d' % image.frame + '.ply',
                fused_pcl)
            np.savez_compressed(
                '_out/fused_pointcloud/' + time.strftime('%Y%m%d_%H%M%S') + '_%06d' % image.frame + '.npz',
                fused_pointcloud)

            # Save the fused voxel occupancy grid
            np.savez_compressed(
                '_out/fused_voxel_grid/' + time.strftime('%Y%m%d_%H%M%S') + '_%06d' % image.frame + '.npz',
                fused_voxel_occupancy_grid)

            np.savez_compressed(
                '_out/candidate_voxel_grid/' + time.strftime('%Y%m%d_%H%M%S') + '_%06d' % image.frame + '.npz',
                candidate_voxel_grid)

        print(f"mutated frame: {self.mutated_frame}, score: {self._score}")

    def compute_visibility_score(self, sensor_occupancy_grid, translation_offset, voxel_size=1, visibility_threshold=1,
                                 max_range_X_Y=360, min_range_Z=-6, max_range_Z=6):
        """
        Computes the visibility score of a vehicle's bounding box by measuring the distance from its voxels to the sensor's occupied voxels.

        Parameters:
        - sensor_occupancy_grid: numpy array, the occupancy grid representing the sensor's field of view in world coordinates
        - voxel_size: float, the size of each voxel in meters
        - visibility_threshold: float, the maximum distance for a voxel to be considered visible
        - sensor_min_bound: numpy array, the minimum bound of the sensor's occupancy grid in world coordinates
        - max_range_X_Y: int, the maximum range in the X and Y axes for the candidate occupancy grid
        - min_range_Z: int, the minimum range in the Z axis for the candidate occupancy grid
        - max_range_Z: float, the maximum range in the Z axis for the candidate occupancy grid

        Returns:
        - visible_percentage: float, the percentage of the vehicle's bounding box that is within the visibility threshold of the sensor's occupied voxels
        - candidate_voxel_grid: numpy array, the occupancy grid of the candidate vehicle
        """

        candidate_pointcloud = o3d.geometry.PointCloud()

        bbox = self.candidate.bounding_box
        extent = bbox.extent
        transform = self.candidate.get_transform()

        x_offsets = np.arange(-extent.x, extent.x, voxel_size)
        y_offsets = np.arange(-extent.y, extent.y, voxel_size)
        z_offsets = np.arange(-extent.z, extent.z, voxel_size)
        local_points = np.array([[x, y, z] for x in x_offsets for y in y_offsets for z in z_offsets])

        pitch = mt.radians(transform.rotation.pitch)
        yaw = mt.radians(transform.rotation.yaw)
        roll = mt.radians(transform.rotation.roll)
        cos_y, sin_y = mt.cos(yaw), mt.sin(yaw)
        cos_p, sin_p = mt.cos(pitch), mt.sin(pitch)
        cos_r, sin_r = mt.cos(roll), mt.sin(roll)

        rotation_matrix = np.array([
            [cos_y * cos_p, cos_y * sin_p * sin_r - sin_y * cos_r, cos_y * sin_p * cos_r + sin_y * sin_r],
            [sin_y * cos_p, sin_y * sin_p * sin_r + cos_y * cos_r, sin_y * sin_p * cos_r - cos_y * sin_r],
            [-sin_p, cos_p * sin_r, cos_p * cos_r]
        ])

        world_points = []
        location = np.array([transform.location.x, transform.location.y, transform.location.z]) + translation_offset
        for pt in local_points:
            rotated_point = rotation_matrix @ pt
            world_point = rotated_point + location
            world_points.append(world_point)

        candidate_pointcloud.points = o3d.utility.Vector3dVector(world_points)

        candidate_voxel_grid = occupancy_grid_map(
            np.asarray(candidate_pointcloud.points), voxel_size=1.0,
            max_range_X_Y=max_range_X_Y, min_range_Z=min_range_Z, max_range_Z=max_range_Z
        )

        sensor_indices = np.argwhere(sensor_occupancy_grid == 1)
        candidate_indices = np.argwhere(candidate_voxel_grid == 1)

        tree = cKDTree(sensor_indices)
        close_pairs = []
        for pos in candidate_indices:
            distances, indices = tree.query(pos, distance_upper_bound=visibility_threshold)

            if np.isfinite(distances):
                close_pairs.append((pos, sensor_indices[indices]))

        visible_voxels = len(close_pairs)
        total_voxels = len(candidate_pointcloud.points)

        visible_percentage = (visible_voxels / total_voxels) * 100 if total_voxels > 0 else 0
        print("Visible Percentage:", visible_percentage)

        return visible_percentage, candidate_voxel_grid

    def clean(self):
        for sensor in self.sensor_list:
            sensor.stop()
            sensor.destroy()

        for actor in self.actor_list:
            actor.destroy()

        for controller in self.controllers_list:
            if controller.is_alive:
                controller.stop()

        for npc in self.npc_list:
            npc[1].destroy()

        print(f"All cleaned up!")

    def get_reasonable_place(self):
        # Step2: sample the reasonable places to mutate
        ego_location = self.ego_location
        candidate_location = self.candidate_location
        vector = candidate_location - ego_location
        distance = np.sqrt(vector.x ** 2 + vector.y ** 2)
        reasonable_places = []

        unit_vector = carla.Location(x=vector.x / distance, y=vector.y / distance)
        left_shift = carla.Location(x=-unit_vector.y * 3, y=unit_vector.x * 3)
        right_shift = carla.Location(x=unit_vector.y * 3, y=-unit_vector.x * 3)
        vertex1 = ego_location + left_shift
        vertex2 = ego_location + right_shift
        vertex3 = candidate_location + left_shift
        vertex4 = candidate_location + right_shift

        num_samples = 10

        for _ in range(num_samples):
            i = random.uniform(0, 1)
            point_left = vertex1 + (vertex3 - vertex1) * i
            point_right = vertex2 + (vertex4 - vertex2) * i

            j = random.uniform(0, 1)
            sample_point = point_left + (point_right - point_left) * j

            reasonable_places.append(sample_point)

        return reasonable_places

    def excuate_mutation(self, reasonable_places):
        """
        Performs mutation on the original seed by randomly selecting a point in reasonable_places
        and placing an NPC at that point, then saving it in the specified seed format.
        """
        mutated_seed = copy.deepcopy(self._seed)
        chosen_point = random.choice(reasonable_places)
        map = self.world.get_map()
        spawn_point = map.get_waypoint(chosen_point, project_to_road=True, lane_type=(carla.LaneType.Driving))
        if spawn_point is None:
            print("can not find a place to mutate")
            return mutated_seed, self._score
        transform = spawn_point.transform
        loc = transform.location
        rotation = transform.rotation

        npc = {
            "spawn_point": {
                "x": loc.x,
                "y": loc.y,
                "z": 3,
                "pitch": rotation.pitch,
                "yaw": rotation.yaw,
                "roll": rotation.roll
            },
            "type": "vehicle.tesla.model3",
            "controller": "vehicle_controller",
            "triggering_time": self.mutated_frame
        }

        npc_name = f"npc{len(mutated_seed['npcs']) + 1}"
        mutated_seed["npcs"].append({npc_name: npc})
        print("Mutation completed.")
        return mutated_seed, self._score


    def run_data_collect_kitti(self, seed):
        self._seed = seed
        # set client
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(30.0)
        # set weather and map
        environment = seed["environment"]
        weather_type = environment["weather_type"]
        self.weather_type = weather_type
        #map = environment["map"]
        map = "Town01"
        print(map)
        world = self.client.load_world(map)
        self.world = world
        server_settings(world)
        set_weather(world, weather_type)
        self.blueprint_library = world.get_blueprint_library()
        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)
        set_all_traffic_lights_to_green(world)
        start_record_full = time.time()

        # time_stop = 0.5
        nbr_frame = 5000  # MAX = 10000
        nbr_vehicles = 30

        # set ego
        ego = seed["ego"]
        ego_route = []
        for _t in ego["route"]:
            point = _t["point"]
            ego_route.append(point)
        ego_spawn_point = world.get_map().get_spawn_points()[23]
        ego_bp = self.blueprint_library.find(ego["type"])
        self.ego = world.spawn_actor(ego_bp, ego_spawn_point)
        self.actor_list.append(self.ego)
        self.ego.set_autopilot(True)

        image_queue_depth = queue.Queue()
        image_queue_lidar = queue.Queue()
        image_queue_rgb = queue.Queue()
        sensor_list = []

        # setup for the sensors
        sensor_config = seed["Sensor"]
        self.sensor_list = sensor_resemble.setup_vehicle_sensors(world, self.ego, sensor_config)
        for sensor in self.sensor_list:
            if sensor.type_id == 'sensor.camera.rgb':
                camera01 = sensor
            elif sensor.type_id == 'sensor.camera.depth':
                camera02 = sensor
            elif sensor.type_id == 'sensor.lidar.ray_cast_semantic':
                lidar01 = sensor
            else:
                print('Unknown sensor type ')

            print(sensor.type_id)

        lidar01.listen(image_queue_lidar.put)
        camera01.listen(image_queue_rgb.put)
        camera02.listen(image_queue_depth.put)

        self.actor_list.append(lidar01)
        sensor_list.append(lidar01)

        self.actor_list.append(camera01)
        sensor_list.append(camera01)

        self.actor_list.append(camera02)
        sensor_list.append(camera02)

        # spawn some npc vehicles
        blueprints = world.get_blueprint_library().filter('vehicle.*')
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('mkz_2020')]
        blueprints = [x for x in blueprints if not x.id.endswith('cooper_s_2021')]
        blueprints = [x for x in blueprints if not x.id.endswith('crown')]
        blueprints = [x for x in blueprints if not x.id.endswith('mkz_2017')]
        blueprints = [x for x in blueprints if not x.id.endswith('model3')]


        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        print("Number of spawn points : ", number_of_spawn_points)
        spawn_points.remove(spawn_points[23])  # remove ego spawn point
        ego_location = spawn_points[23].location

        distance_threshold = 50.0

        filtered_spawn_points = [point for point in spawn_points if
                                 get_distance(point.location, ego_location) <= distance_threshold]

        filtered_spawn_points.remove(spawn_points[23])
        selected_spawn_points = random.sample(filtered_spawn_points, 10)

        if nbr_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif nbr_vehicles >= number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            print(msg % (nbr_vehicles, number_of_spawn_points))
            nbr_vehicles = number_of_spawn_points - 1

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for n, transform in enumerate(selected_spawn_points):
                if n >= nbr_vehicles:
                        break
                blueprint = random.choice(blueprints)
                if blueprint.has_attribute('color'):
                        color = random.choice(blueprint.get_attribute('color').recommended_values)
                        blueprint.set_attribute('color', color)
                if blueprint.has_attribute('driver_id'):
                        driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                        blueprint.set_attribute('driver_id', driver_id)
                blueprint.set_attribute('role_name', 'autopilot')

                # # prepare the light state of the cars to spawn
                # light_state = vls.NONE
                # car_lights_on = False
                # if car_lights_on:
                #         light_state = vls.Position | vls.LowBeam | vls.LowBeam

                # spawn the cars and set their autopilot and light state all together
                batch.append(SpawnActor(blueprint, transform)
                        .then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port())))
                        # .then(SetVehicleLightState(FutureActor, light_state)))

        # batch.append(SpawnActor(bp_ego, start_pose)
        #         .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in self.client.apply_batch_sync(batch, True):
                if response.error:
                        print(response.error)
                else:
                        self.npc_list.append(response.actor_id)


        pcl_downsampled = o3d.geometry.PointCloud()
        frame = 0
        dt0 = datetime.now()
        frame_CP = 0
     #   pdb.set_trace()
        # run loop
        while frame < self.frame:
            frame += 1
            world.tick()
            w_frame = world.get_snapshot().frame
            follow(self.ego.get_transform(), world)
            lidar_data = image_queue_lidar.get()
            image_data = image_queue_rgb.get()

            lidar_pcl, center_lidar = lidar_transformation(lidar_data, lidar01)
            lidar_red_indices = np.where(np.asarray(lidar_pcl.colors)[:, 0] == 255)[0]
            lidar_points = np.delete(np.asarray(lidar_pcl.points), lidar_red_indices, axis=0)
            lidar_pcl.points = o3d.utility.Vector3dVector(lidar_points)
            lidar_pcl.colors = o3d.utility.Vector3dVector([])

            '''
            spectator = world.get_spectator()
            ego_transform = self.ego.get_transform()
            spectator_location = ego_transform.location + carla.Location(z=5)
            spectator_rotation = ego_transform.rotation
            spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))
            '''
            img_filename = IMAGE01_PATH.format(frame)

            lidar_filename = LIDAR01_PATH.format(frame)


            sensor_data = image_data

            image_data.save_to_disk(img_filename)

            data = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.float32))
            data = np.reshape(data, (int(data.shape[0] / 4), 4))

            data[:, 1] = -1 * data[:, 1]
            lidar_array = np.array(data).astype(np.float32)
            # logging.debug("Lidar min/max of x: {} {}".format(
            #               lidar_array[:, 0].min(), lidar_array[:, 0].max()))
            # logging.debug("Lidar min/max of y: {} {}".format(
            #               lidar_array[:, 1].min(), lidar_array[:, 0].max()))
            # logging.debug("Lidar min/max of z: {} {}".format(
            #               lidar_array[:, 2].min(), lidar_array[:, 0].max()))
            lidar_array.tofile(lidar_filename)

            #save_lidar_data(lidar_filename, lidar_data, LIDAR_DATA_FORMAT)

            calib_filename = CALIBRATION01_PATH.format(frame)
            plane_filename = GROUNDPLANE_PATH.format(frame)

            actor_list = world.get_actors()
            vehicle_list = actor_list.filter('vehicle.*')
            pedestrians_list = actor_list.filter('walker.pedestrian.*')

            ego_vehicle_id = self.ego.id

            label_data = []

            for actor in vehicle_list:
                cur_N_points = downsample_within_bbox(lidar_pcl, actor)
                cur_N = len(np.asarray(cur_N_points.points))
                if actor.id == ego_vehicle_id:
                    continue
                if isInRange(actor.get_location(), self.ego.get_location()):
                    kitti_datapoint = creat_kitti_datapoint(actor, sensor_data.transform, 'Car', cur_N)
                    if kitti_datapoint:
                        label_data.append(kitti_datapoint)

            for actor in pedestrians_list:
                cur_N_points = downsample_within_bbox(lidar_pcl, actor)
                cur_N = len(np.asarray(cur_N_points.points))
                if isInRange(actor.get_location(),  self.ego.get_location()):
                    kitti_datapoint = creat_kitti_datapoint(actor, sensor_data.transform, 'Pedestrian', cur_N)
                    if kitti_datapoint:
                        label_data.append(kitti_datapoint)

            camera_intrinsic = ground_truth.get_intrinsic_matrix(camera01)
            cam_transform = camera01.get_transform()
            lidar_transform = lidar01.get_transform()
            veh_cam_mat = cam_transform.get_inverse_matrix()
            lidar_veh_mat = lidar_transform.get_matrix()
            lidar_to_cam_matrix = np.dot(veh_cam_mat, lidar_veh_mat)
            save_groundplanes(plane_filename, self.ego)
            save_calibration_matrices(sensor_data, calib_filename, camera_intrinsic, lidar_to_cam_matrix)
            label_filename = LABEL01_PATH.format(frame)
            save_kitti_label_data(label_filename, label_data)

            frame_CP += 1
            print("\nRecorded frame %d, CP at World's frame: %d" % (frame_CP / 5, w_frame))


            process_time = datetime.now() - dt0
            sys.stdout.write('\r' + 'FPS: ' + str(1.0 / process_time.total_seconds()))
            sys.stdout.flush()
            dt0 = datetime.now()


def stop_carla():
    """
    Checks if any CARLA processes are running and terminates them.
    Returns True if processes were found and terminated, False if no processes were found.
    """
    try:
        result = subprocess.run(['pkill', '-f', 'CarlaUE4'], check=True)
        print("Existing CARLA processes terminated.")
        return True

    except subprocess.CalledProcessError:
        print("No existing CARLA processes were found.")
        return False


def run_carla():
    try:
        process = subprocess.Popen(['/opt/carla/CarlaUE4.sh'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(12)
        if process.poll() is None:
            print("CARLA started successfully.")
            return True
        else:
            print("CARLA failed to start.")
            return False

    except Exception as e:
        print("Failed to start CARLA:", str(e))
        return False


def main():
    while stop_carla():
        print("Retrying to terminate CARLA processes...")
    run_carla()

    try:
        with open("/home/adsec/blindhunter/data/seed/mutated_seed.json", "r") as file:
            seed = json.load(file)

        carla_runner = CarlaRunner()
        carla_runner.run_data_collect_kitti(seed)
        carla_runner.clean()
        del carla_runner


    except Exception as e:
        print("An error occurred:", e)
    finally:
        sys.exit(0)

def follow(transform, world):    # Transforme carla.Location(x,y,z) from sensor to world frame
    rot = transform.rotation
    rot.pitch = -25
    world.get_spectator().set_transform(carla.Transform(transform.transform(carla.Location(x=-15,y=0,z=5)), rot))

if __name__ == '__main__':
    main()






