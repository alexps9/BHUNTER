import os
import numpy as np
import random
import open3d as o3d
import matplotlib.pyplot as plt
import carla
import queue
import math as mt
import cv2
import subprocess
from scipy.spatial import cKDTree
from grid_map.utils.setup import setup_world, environment
from grid_map.utils.spawn import spawn_sensor, spawn_vehicle
from grid_map.utils.ground_truth import ground_truth as ground_truth
from grid_map.utils.gennerate_traffic import gennerate_traffic
from grid_map.utils.spawn import sensor_resemble as sensor_resemble
from scenario_runner.srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from scenario_runner.srunner.scenariomanager.actorcontrols.waypoint_control import WaypointVehicleControl
import copy
import json
import sys
import time
import pdb
import math
import argparse
import hashlib
import multiprocessing
import shutil
from carla_data_descriptor import CarlaDataDescriptor, CarlaDataDescriptorTracking
import CMM_CARLA_Config as CFG
from numpy.linalg import pinv, inv
from CMM_CARLA_Config import *
from PIL import Image
import math
import logging
from math import pi
import bisect
import scipy
import skfuzzy
from scipy import signal as signal

""" OUTPUT FOLDER GENERATION FOR KITTI DETECTION"""
occlusion_levels = [(i / 10) for i in range(0, 11)]

tracking_levels = [str(x) for x in [0, 0.3, 0.6]]


folders = [
    'calib1', 'image_2', 'label_2', 'velodyne1', 'ImageSets',
    'label_1', 'image_1', 'calib2', 'velodyne2', 'image_depth', 'planes'
]

""" OUTPUT FOLDER GENERATION FOR KITTI Tracking"""

def maybe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_distance(loc1, loc2):
    return math.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2 + (loc1.z - loc2.z) ** 2)

""" DATA SAVE PATHS """
def get_data_paths(output_folder):
    return {
        "INDEX_PATH": os.path.join(output_folder, 'ImageSets'),
        "LIDAR01_PATH": os.path.join(output_folder, 'velodyne1/{0:06}.bin'),
        "LIDAR01_PATH_PLY": os.path.join(output_folder, 'velodyne1/{0:06}.ply'),
        "LABEL01_PATH": os.path.join(output_folder, 'label_1/{0:06}.txt'),
        "IMAGE01_PATH": os.path.join(output_folder, 'image_1/{0:06}.png'),
        "CALIBRATION01_PATH": os.path.join(output_folder, 'calib1/{0:06}.txt'),
        "GROUNDPLANE_PATH": os.path.join(output_folder, 'planes/{0:06}.txt'),
        "LABEL02_PATH": os.path.join(output_folder, 'label_2/{0:06}.txt'),
        "IMAGE02_PATH": os.path.join(output_folder, 'image_2/{0:06}.png'),
        "CALIBRATION02_PATH": os.path.join(output_folder, 'calib2/{0:06}.txt'),
        "LIDAR02_PATH": os.path.join(output_folder, 'velodyne2/{0:06}.bin'),
        "LIDAR02_PATH_PLY": os.path.join(output_folder, 'velodyne2/{0:06}.ply'),
        "DEPTH_IMAGE_PATH": os.path.join(output_folder, 'image_depth/{0:06}.png'),
    }


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

lane_type_pool = {
    "vehicle": carla.LaneType.Driving,
    "walker": carla.LaneType.Sidewalk,
    "cyclist": carla.LaneType.Biking,
    "static": carla.LaneType.Parking
}

# 12 = pedestrian, 14 = vehicle, 21 = dynamic.
interesting_tags = [12, 14, 21]

noise_limit = 2.0

vehicle_pools = [
    'vehicle.audi.a2',
    'vehicle.citroen.c3',
    'vehicle.chevrolet.impala',
    'vehicle.dodge.charger_police_2020',
    'vehicle.micro.microlino',
    'vehicle.dodge.charger_police',
    'vehicle.audi.tt',
    'vehicle.jeep.wrangler_rubicon',
    'vehicle.mercedes.coupe',
    'vehicle.mercedes.coupe_2020',
    'vehicle.dodge.charger_2020',
    'vehicle.ford.ambulance',
    'vehicle.lincoln.mkz_2020',
    'vehicle.mini.cooper_s_2021',
    'vehicle.toyota.prius',
    'vehicle.ford.crown',
    'vehicle.carlamotors.carlacola',
    'vehicle.nissan.patrol_2021',
    'vehicle.mercedes.sprinter',
    'vehicle.audi.etron',
    'vehicle.seat.leon',
    'vehicle.volkswagen.t2_2021',
    'vehicle.tesla.cybertruck',
    'vehicle.lincoln.mkz_2017',
    'vehicle.ford.mustang',
    'vehicle.carlamotors.firetruck',
    'vehicle.volkswagen.t2',
    'vehicle.mitsubishi.fusorosa',
    'vehicle.tesla.model3',
    'vehicle.nissan.patrol',
    'vehicle.nissan.micra',
    'vehicle.mini.cooper_s'
]

van_pools = ['vehicle.ford.ambulance',
'vehicle.mercedes.sprinter',
'vehicle.tesla.cybertruck',
'vehicle.carlamotors.firetruck',
'vehicle.volkswagen.t2_2021',
'vehicle.volkswagen.t2',
'vehicle.mitsubishi.fusorosa',
'vehicle.nissan.patrol_2021',
'vehicle.nissan.patrol'
]

def server_settings(world, rendering: bool = False):
    """
    Set the server settings for the given world object.

    Parameters:
    - world: the world object on which to apply the settings
    """
    settings = world.get_settings()
    settings.no_rendering_mode = rendering  # No rendering mode
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.10
    world.apply_settings(settings)




def set_weather(world, weather: dict):
    weather_params = carla.WeatherParameters(
        cloudiness=weather.get('cloudiness', 0),
        precipitation=weather.get('precipitation', 0),
        precipitation_deposits=weather.get('precipitation_deposits', 0),
        wind_intensity=weather.get('wind_intensity', 0),
        fog_density=weather.get('fog_density', 0),
        wetness=weather.get('wetness', 0),
        sun_azimuth_angle=weather.get('sun_azimuth_angle', 0),
        sun_altitude_angle=weather.get('sun_altitude_angle', 45)
    )

    world.set_weather(weather_params)
    print(f"Weather updated: {weather}")

def set_all_traffic_lights_to_green(world):
    traffic_lights = world.get_actors().filter('traffic.traffic_light')

    for traffic_light in traffic_lights:
        traffic_light.set_state(carla.TrafficLightState.Green)
        traffic_light.set_green_time(99999.0)
        traffic_light.freeze(True)

    print(f"All {len(traffic_lights)} traffic lights set to green.")


def on_tick_wrapper(snapshot):
    CarlaDataProvider.on_carla_tick()


def generate_pointcloud_from_obb(obb, num_points=300, z_min=0.15, z_max=10):
    valid_points = []

    while len(valid_points) < num_points:

        points = np.random.uniform(0, 1.0, (num_points, 3))
        points *= obb.extent* 0.95
        points = (obb.R @ points.T).T + obb.center



        filtered_points = points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max)]


        valid_points.extend(filtered_points)


    valid_points = np.array(valid_points[:num_points])


    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(valid_points)

    return pcl

def generate_surface_pointcloud_from_obb(obb, num_points=4000):

    extent_x, extent_y, extent_z = obb.extent
    extent_z *= 0.95

    num_points_per_face = num_points

    points = []

    for _ in range(num_points_per_face):
        face_id = np.random.randint(0, 4)

        if face_id == 0:
            x = extent_x / 2
            y, z = np.random.uniform(-extent_y / 2, extent_y / 2), np.random.uniform(-extent_z / 2, extent_z / 2)
        elif face_id == 1:
            x = -extent_x / 2
            y, z = np.random.uniform(-extent_y / 2, extent_y / 2), np.random.uniform(-extent_z / 2, extent_z / 2)
        elif face_id == 2:
            y = extent_y / 2
            x, z = np.random.uniform(-extent_x / 2, extent_x / 2), np.random.uniform(-extent_z / 2, extent_z / 2)
        elif face_id == 3:
            y = -extent_y / 2
            x, z = np.random.uniform(-extent_x / 2, extent_x / 2), np.random.uniform(-extent_z / 2, extent_z / 2)

        points.append([x, y, z])

    points = np.array(points)

    points = (obb.R @ points.T).T + obb.center


    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)

    return pcl

def spawn_vehicle_route(world, blueprint, traffic_manager, route, weather_type, spawn_point, type_name: str, speed):
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
    traffic_manager.random_left_lanechange_percentage(vehicle, 0)
    traffic_manager.random_right_lanechange_percentage(vehicle, 0)
    traffic_manager.auto_lane_change(vehicle, False)
    traffic_manager.set_desired_speed(vehicle, speed)
    traffic_manager.global_percentage_speed_difference(0)
    traffic_manager.ignore_vehicles_percentage(vehicle, 0)

    # traffic_manager.ignore_vehicles_percentage(vehicle, 100)
    for ind in route_indices:
        routes.append(spawn_points[ind].location)

    traffic_manager.set_path(vehicle, routes)

    if weather_type == "NightCloudy":
        traffic_manager.update_vehicle_lights(vehicle, True)

    return vehicle, spawn_points[route_indices[0]], spawn_points[route_indices[-1]]


def spawn_vehicle(world, blueprint, spawn_point: carla.Transform):
    # Get the blueprint for the vehicle - Tesla Model 3
    vehicle = world.try_spawn_actor(blueprint, spawn_point)
    if vehicle is None:
        raise Exception("Vehicle spawn failed: vehicle is None. Please check spawn point or blueprint.")

    vehicle.set_autopilot(True)

    return vehicle


def spawn_npc(world, blueprint, spawn_point: carla.Transform):
    vehicle = world.try_spawn_actor(blueprint, spawn_point)
    if vehicle is None:
        raise Exception("Vehicle spawn failed: vehicle is None. Please check spawn point or blueprint.")

    return vehicle


def filter_pcl_bounds_numpy(pcl, x_min=-10.0, x_max=10.0, y_min=-10.0, y_max=10.0,
                            z_min=-10.0, z_max=10.0):
    '''
    Restricts a point cloud to exclude coordinates outside a certain cube.
    :param pcl (N, D) numpy array: Point cloud with first 3 elements per row = (x, y, z).
    :return (N, D) numpy array: Filtered point cloud.
    '''
    mask_x = np.logical_and(x_min <= pcl[..., 0], pcl[..., 0] <= x_max)
    mask_y = np.logical_and(y_min <= pcl[..., 1], pcl[..., 1] <= y_max)
    mask_z = np.logical_and(z_min <= pcl[..., 2], pcl[..., 2] <= z_max)
    mask_xy = np.logical_and(mask_x, mask_y)
    mask_xyz = np.logical_and(mask_xy, mask_z)
    result = pcl[mask_xyz]
    return result


def get_ground_truth(queue_list, depth_camera_list, segmentation_list):
    """
    The function 'get_ground_truth' processes depth images from multiple cameras to generate a point cloud.

    :param segmentation_list: segmentation image queue
    :param queue_list: Dictionary containing queues for different types of depth images.
    :param depth_camera_list: Dictionary containing depth camera objects for front, right, left, and back cameras.

    :return: The function `ground_truth` returns three values:
    1. `points`: A numpy array containing the 3D points in the world space for all four cameras.
    2. `colors`: A numpy array containing the color information (RGB) corresponding to each 3D point.
    3. `front_extrinsic_matrix`: The extrinsic matrix corresponding to the front camera.
    """
    front_depth_image = queue_list['image_queue_depth_front'].get()
    segmentation_image = segmentation_list.get()
    camera2vehicle_matrix = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    # Get the intrinsic and extrinsic matrix of the 4 cameras
    front_intrinsic_matrix, front_extrinsic_matrix = ground_truth.get_intrinsic_extrinsic_matrix(
        depth_camera_list['front_depth_camera'], front_depth_image)

    # Get the points [[X...], [Y...], [Z...]] and the colors [[R...], [G...], [B...]]
    front_points_3D, front_color = ground_truth.point2D_to_point3D_(front_depth_image, front_intrinsic_matrix, segmentation_image)
    # To multiply by the extrinsic matrix (same shape as the extrinsic_matrix matrix)
    front_p3d = np.concatenate((front_points_3D, np.ones((1, front_points_3D.shape[1]))))

    transform_matrix = np.array(front_depth_image.transform.get_matrix())
    to_world_matrix = transform_matrix @ camera2vehicle_matrix
    # Get the 3D points in the world
    front_p3d_world = np.dot(to_world_matrix, front_p3d)[:3]

    # Reshape the array to (height * width, 3) -> X, Y and Z for each point
    front_p3d_world = np.transpose(front_p3d_world)

    points = front_p3d_world
    colors = front_color


    # Put the center of the point cloud in the origin
    if len(points) == 0:
        centroid = None
    else:
       centroid = np.mean(points, axis=0)
   # points = points - centroid

    return points, colors, front_intrinsic_matrix, centroid


def ensure_directory_exists(directory_path):
    os.makedirs(directory_path, exist_ok=True)


def filter_points_outside_ego_bbox(pcl):
    '''
    Filters out points that lie within the ego vehicle's bounding box.

    :param pcl: (N, D) numpy array. Point cloud where each row represents a point.
    :param bbox_min: (x_min, y_min, z_min) numpy array or list. Minimum bounds of the bounding box.
    :param bbox_max: (x_max, y_max, z_max) numpy array or list. Maximum bounds of the bounding box.
    :return: (N, D) numpy array. Points outside the bounding box.
    '''
    # Unpack bounding box limits
    x_min, y_min, z_min = -2, -2, -2
    x_max, y_max, z_max = 2, 2, 5

    # Create masks for points outside the bounding box
    mask_x = np.logical_or(pcl[:, 0] < x_min, pcl[:, 0] > x_max)
    mask_y = np.logical_or(pcl[:, 1] < y_min, pcl[:, 1] > y_max)
    mask_z = np.logical_or(pcl[:, 2] < z_min, pcl[:, 2] > z_max)

    # Combine masks: Points outside the box must be outside at least one dimension
    mask_outside = np.logical_or(np.logical_or(mask_x, mask_y), mask_z)

    # Filter the points outside the bounding box
    result = pcl[mask_outside]
    return result


def find_candidate(image_queue_lidar, lidar, tag, actor, relative_velocity: carla.Vector3D, id: None):
    """
    Transforms raw lidar data into a point cloud and projects it into the world coordinate system,
    and reflects the points across the lidar's origin x, y coordinates in world space.

    :param relative_velocity: the relative velocity of candidate
    :param loc: the center of candidate (x,y)
    :param image_queue_lidar: The queue that holds a byte array containing the raw lidar data.
    :param lidar: The CARLA lidar sensor object to retrieve its transform for coordinate conversion.
    :param tag: The tag of the interesting points.
    :param distance: The distance threshold for selecting points.
    :param ego: The ego vehicle object to get its location.

    :return:
    The ID of the interesting point closest to the ego within the specified distance, or None if no such point exists.
    """
    lidar_pcl_interesting = o3d.geometry.PointCloud()

    target_bbox = actor.bbox

    # Get the lidar transform matrix (lidar to world)
    lidar_to_world = np.array(lidar.get_transform().get_matrix())

    # Retrieve raw lidar data
    lidar_data = image_queue_lidar.get()
    raw_data = lidar_data.raw_data

    lidar_transform = lidar_data.transform
    lidar_rotation = lidar_transform.rotation
    lidar_yaw = lidar_rotation.yaw

    # Parse raw lidar data into structured arrays
    pcl_float = np.frombuffer(raw_data, dtype=np.dtype('f4')).reshape(-1, 6)[:, :4]  # (N, 4) (x, y, z, cosine_angle)
    pcl_int = np.frombuffer(raw_data, dtype=np.dtype('u4')).reshape(-1, 6)[:, 4:6].astype(np.float32)  # (N, 2)
    point_cloud_array = np.hstack([pcl_float, pcl_int])  # (N, 6)

    # Filter points by tag
    tagged_points = point_cloud_array[point_cloud_array[:, 5] == tag]

    # Remove points inside ego bounding box
    filtered_points = filter_points_outside_ego_bbox(tagged_points)
    if filtered_points.size == 0:
        return None, None

    # Process unique IDs in filtered points
    unique_ids = np.unique(filtered_points[:, 4])

    if id is None:
        for unique_id in unique_ids:
            points = filtered_points[filtered_points[:, 4] == unique_id][:, :3]
            if points.size == 0:
                continue
            # Transform points to world coordinates
            homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
            world_points = (lidar_to_world @ homogeneous_points.T).T[:, :3]
            '''
            time_delta = 0.1
            velocity_correction = np.array([relative_velocity.x, relative_velocity.y, relative_velocity.z]) * time_delta
            world_points -= velocity_correction
            '''

            # Check distance to ego for each transformed point
            for x, y, z in world_points:
                tmp_loc = carla.Location(x, y, z)
                if target_bbox.contains(tmp_loc, actor.get_transform()):
                    lidar_pcl_interesting.points = o3d.utility.Vector3dVector(world_points)
                    return unique_id, lidar_pcl_interesting

    else:
        points = filtered_points[filtered_points[:, 4] == id][:, :3]
        if points.size == 0:
            return None, None
        # Transform points to world coordinates
        homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
        world_points = (lidar_to_world @ homogeneous_points.T).T[:, :3]

        lidar_pcl_interesting.points = o3d.utility.Vector3dVector(world_points)
        return id, lidar_pcl_interesting

    return None, None


def lidar_transformation(image_queue_lidar, lidar, tag):
    """
    Transforms raw lidar data into a point cloud and projects it into the world coordinate system,
    and reflects the points across the lidar's origin x, y coordinates in world space.

    :param tag: the tag of the interesting points.
    :param image_queue_lidar: The queue that holds a byte array containing the raw lidar data.
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
    lidar_pcl_interesting = o3d.geometry.PointCloud()

    # Retrieve raw lidar data
    lidar_data = image_queue_lidar.get()
    raw_data = lidar_data.raw_data

    pcl_float = np.frombuffer(raw_data, dtype=np.dtype('f4'))
    pcl_float = np.reshape(pcl_float, (int(pcl_float.shape[0] / 6), 6))
    pcl_float = pcl_float[..., :4]  # (N, 4) with (x, y, z, cosine_angle).
    pcl_int = np.frombuffer(raw_data, dtype=np.dtype('u4'))
    pcl_int = np.reshape(pcl_int, (int(pcl_int.shape[0] / 6), 6))
    pcl_int = pcl_int[..., 4:6]  # (N, 2) with (object_index, semantic_tag).
    pcl_int = pcl_int.astype(np.float32)
    point_cloud_array = np.concatenate([pcl_float, pcl_int], axis=1)  # (N, 6).

    point_cloud_array_before_interesting = point_cloud_array[point_cloud_array[:, 5].astype(np.float32) == tag]
    point_cloud_array_interesting = filter_points_outside_ego_bbox(point_cloud_array_before_interesting)
    point_cloud_array_interesting_list = []
    column_4_values = point_cloud_array_interesting[:, 4].astype(np.float32)
    unique_values = np.unique(column_4_values)

    for id in unique_values:
        point_cloud_array_interesting_tmp = point_cloud_array_interesting[
            point_cloud_array_interesting[:, 4].astype(np.float32) == id]
        point_cloud_array_interesting_tmp = point_cloud_array_interesting_tmp[:, :3]
        transformed_points_tmp = np.zeros_like(point_cloud_array_interesting_tmp)
        transformed_points_tmp[:, 0] = point_cloud_array_interesting_tmp[:, 0]  # y → x
        transformed_points_tmp[:, 1] = point_cloud_array_interesting_tmp[:, 1]  # x → y
        transformed_points_tmp[:, 2] = point_cloud_array_interesting_tmp[:, 2]  # -z → z
        points_homogeneous_tmp = np.hstack(
            (transformed_points_tmp, np.ones((transformed_points_tmp.shape[0], 1))))
        points_world_tmp = np.dot(lidar_to_world, points_homogeneous_tmp.T).T[:,
                           :3]  # Transform and remove the homogeneous coordinate
        lidar_pcl_tmp = o3d.geometry.PointCloud()
        lidar_pcl_tmp.points = o3d.utility.Vector3dVector(points_world_tmp)
        point_cloud_array_interesting_list.append((id, lidar_pcl_tmp))

    # keep (x,y,z) of each point
    point_cloud_array_interesting = point_cloud_array_interesting[:, :3]
    point_cloud_array = point_cloud_array[:, :3]

    # Apply coordinate transformation (x, y, z) → (y, x, -z)
    transformed_points = np.zeros_like(point_cloud_array)
    transformed_points[:, 0] = point_cloud_array[:, 0]  # y → x
    transformed_points[:, 1] = point_cloud_array[:, 1]  # x → y
    transformed_points[:, 2] = point_cloud_array[:, 2]  # -z → z

    transformed_points_interesting = np.zeros_like(point_cloud_array_interesting)
    transformed_points_interesting[:, 0] = point_cloud_array_interesting[:, 0]  # y → x
    transformed_points_interesting[:, 1] = point_cloud_array_interesting[:, 1]  # x → y
    transformed_points_interesting[:, 2] = point_cloud_array_interesting[:, 2]  # -z → z

    # Transform lidar points to world coordinates
    # Add a column of 1s for homogeneous coordinates
    points_homogeneous = np.hstack((transformed_points, np.ones((transformed_points.shape[0], 1))))
    points_world = np.dot(lidar_to_world, points_homogeneous.T).T[:,
                   :3]  # Transform and remove the homogeneous coordinate

    points_homogeneous_interesting = np.hstack(
        (transformed_points_interesting, np.ones((transformed_points_interesting.shape[0], 1))))
    points_world_interesting = np.dot(lidar_to_world, points_homogeneous_interesting.T).T[:,
                               :3]  # Transform and remove the homogeneous coordinate

    # Get the lidar's origin coordinates in world space
    origin_world = np.array([lidar.get_location().x, lidar.get_location().y, lidar.get_location().z])

    # Reflect points_world across the origin's x, y in world space
    # points_world[:, 0] = 2 * origin_world[0] - points_world[:, 0]  # Reflect x
    # points_world[:, 1] = 2 * origin_world[1] - points_world[:, 1]  # Reflect y

    # Assign transformed points to the Open3D PointCloud
    lidar_pcl.points = o3d.utility.Vector3dVector(points_world)

    lidar_pcl_interesting.points = o3d.utility.Vector3dVector(points_world_interesting)

    # Get the lidar's origin in world coordinates (translation part of the transformation matrix)
    center_lidar = origin_world

    return lidar_pcl, center_lidar, lidar_pcl_interesting, point_cloud_array_interesting_list

def generate_points(vehicle: carla.Vehicle, density=10):
    """
    Generate a point cloud (Open3D format) within the bounding box of a vehicle in world coordinates.
    All y-coordinates are negated.

    :param vehicle: carla.Vehicle object
    :param density: Density of the points in the bounding box (points per cubic meter)
    :return: open3d.geometry.PointCloud object containing the points in world coordinates
    """
    if not isinstance(vehicle, carla.Vehicle):
        raise ValueError("Input is not a valid carla.Vehicle object.")

    # Get the bounding box of the vehicle
    bounding_box = vehicle.bounding_box
    bb_extent = bounding_box.extent  # Half-extent of the bounding box in local coordinates

    if bb_extent.x <= 0 or bb_extent.y <= 0 or bb_extent.z <= 0:
        raise ValueError(f"Invalid bounding box extent: {bb_extent}")

    # Calculate the number of points along each dimension
    x_points = max(1, int(2 * bb_extent.x * density))
    y_points = max(1, int(2 * bb_extent.y * density))
    z_points = max(1, int(2 * bb_extent.z * density))

    if x_points * y_points * z_points <= 0:
        raise ValueError(f"Invalid point grid dimensions: x={x_points}, y={y_points}, z={z_points}")

    # Generate points in local vehicle coordinates
    x_range = np.linspace(-bb_extent.x, bb_extent.x, x_points)
    y_range = np.linspace(-bb_extent.y, bb_extent.y, y_points)
    z_range = np.linspace(-bb_extent.z, bb_extent.z, z_points)

    # Create a grid of points
    x, y, z = np.meshgrid(x_range, y_range, z_range)
    local_points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T

    # Transform points to world coordinates
    world_points = []
    vehicle_transform = vehicle.get_transform()
    for point in local_points:
        try:
            # Convert to world coordinates
            local_location = carla.Location(point[0], point[1], point[2])
            world_location = vehicle_transform.transform(local_location)
            # Negate the y-coordinate
            world_points.append([world_location.x, world_location.y, world_location.z])
        except Exception as e:
            print(f"Error transforming point {point}: {e}")

    if not world_points:
        raise ValueError("No points were successfully transformed to world coordinates.")

    # Convert world points to a numpy array
    world_points_np = np.array(world_points)

    # Create an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(world_points_np)

    # Set the color of all points to red (RGB: [1.0, 0.0, 0.0])
    colors = np.array([[1.0, 0.0, 0.0]] * len(world_points_np))  # Red color for all points
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud

def generate_bbox(vehicle: carla.Vehicle, lidar_transform: carla.Transform):
    if not isinstance(vehicle, carla.Vehicle):
        raise ValueError("Input is not a valid carla.Vehicle object.")

    world_to_lidar = np.array(lidar_transform.get_inverse_matrix())

    bounding_box = vehicle.bounding_box

    bb_extent = np.array([bounding_box.extent.x, bounding_box.extent.y, bounding_box.extent.z]) * 2
    bb_center = bounding_box.location


    vehicle_transform = vehicle.get_transform()
    vehicle_to_world = np.array(vehicle_transform.get_matrix())


    label_in_lidar = np.matmul(world_to_lidar, vehicle_to_world)


    rotation_matrix_target = vehicle_to_world[:3, :3]
    translation_target = vehicle_to_world[:3, 3]

    obb = o3d.geometry.OrientedBoundingBox()
    obb.center = translation_target
    obb.R = rotation_matrix_target
    obb.extent = bb_extent
    obb.color = np.array([1.0, 0.0, 0.0])

    return obb

def project_3d_to_2d_image(pcl: o3d.geometry.PointCloud, cam_trans: carla.Transform, intrinsic_matrix: np.ndarray,
                           image_size=(1248, 384)):
    """
    Projects a 3D point cloud onto a 2D image plane using camera transformation and intrinsic parameters.

    :param pcl: Open3D PointCloud object containing 3D points.
    :param cam_trans: CARLA Transform object representing the camera transformation.
    :param intrinsic_matrix: 3x3 intrinsic matrix of the camera.
    :param image_size: Tuple (width, height) representing the image size.
    :return: RGB image with projected 3D points visualized.
    """
    width, height = image_size
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)


    world_points = np.asarray(pcl.points)  # (N, 3)


    num_points = world_points.shape[0]
    world_points_homo = np.hstack((world_points, np.ones((num_points, 1))))  # (N, 4)

    camera2vehicle_matrix = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    transform_matrix = np.array(cam_trans.get_matrix())
    to_world_matrix = transform_matrix @ camera2vehicle_matrix
    to_cam_matrix = np.linalg.inv(to_world_matrix)


    cam_points_homo = np.dot(to_cam_matrix, world_points_homo.T).T
    cam_points = cam_points_homo[:, :3]

    ''''''

    valid_depth_mask = cam_points[:, 2] > 0
    cam_points = cam_points[valid_depth_mask]


    pixel_coords_homo = intrinsic_matrix @ cam_points.T  # (3, N)

    u = pixel_coords_homo[0, :] / (pixel_coords_homo[2, :] )
    v = pixel_coords_homo[1, :] / (pixel_coords_homo[2, :] )
    depth = pixel_coords_homo[2, :]

    valid_mask = (0 <= u) & (u < width) & (0 <= v) & (v < height)


    u_valid, v_valid = u[valid_mask].astype(int), v[valid_mask].astype(int)

    for i in range(len(u_valid)):
        cv2.circle(rgb_image, (u_valid[i], v_valid[i]), radius=1, color=(0, 255, 0), thickness=-1)

    return rgb_image, len(u_valid)

def get_occlusion(candidate_img, npc_img):
    if candidate_img.shape != npc_img.shape:
        raise ValueError("Error calculating the occlusion mask.")

    pure_green = np.array([0, 255, 0])

    mask1 = np.all(candidate_img == pure_green, axis=-1)
    mask2 = np.all(npc_img == pure_green, axis=-1)

    green_overlap = np.logical_and(mask1, mask2).astype(np.uint8) * 255

    occlusion_count = np.sum(green_overlap > 0)

    return occlusion_count


def ray_tracing(candidate_pcl: o3d.geometry.PointCloud,
                candidate_bbox: o3d.geometry.OrientedBoundingBox,
                occluder_bboxs,
                sensor_loc: np.array):
    scene = o3d.t.geometry.RaycastingScene()


    def obb_to_trimesh(obb):
        mesh = o3d.geometry.TriangleMesh.create_box(width=obb.extent[0],
                                                    height=obb.extent[1],
                                                    depth=obb.extent[2])
        mesh.translate(-mesh.get_center())
        mesh.rotate(obb.R, center=np.array([0, 0, 0]))
        mesh.translate(obb.center)
        return mesh

    occluder_ids = []
    for occluder_bbox in occluder_bboxs:
        occluder_mesh = obb_to_trimesh(occluder_bbox)
        occluder_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(occluder_mesh))
        occluder_ids.append(occluder_id)

    candidate_mesh = obb_to_trimesh(candidate_bbox)
    candidate_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(candidate_mesh))

    points = np.asarray(candidate_pcl.points)

    directions = points - sensor_loc
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    rays = np.column_stack([np.tile(sensor_loc, (points.shape[0], 1)), directions])
    rays_o3d = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    result = scene.cast_rays(rays_o3d)

    hit_geometry_ids = result['geometry_ids'].numpy()
    hit_distances = result['t_hit'].numpy()
    is_hit = hit_distances < np.inf

    hit_points = sensor_loc + (hit_distances[:, None] * directions)
    distances_to_candidate = np.linalg.norm(hit_points - points, axis=1)

    occluder_mask = np.isin(hit_geometry_ids, occluder_ids) & is_hit
    self_bbox_mask = (hit_geometry_ids == candidate_id) & is_hit & (distances_to_candidate > 0.01)
    free_mask = ~occluder_mask & ~self_bbox_mask

    num_occluded = np.sum(occluder_mask)
    num_self_occluded = np.sum(self_bbox_mask)
    num_free = np.sum(free_mask)

    result_dict = {
        "num_occluded": num_occluded,
        "num_self_occluded": num_self_occluded,
        "num_free": num_free
    }

    return result_dict

'''
============ data collection ===========
'''

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
    minX = sensor_location.x - 120
    maxX = sensor_location.x + 120
    minY = sensor_location.y - 120
    maxY = sensor_location.y + 120

    if locationX >= minX and locationX<= maxX and locationY >= minY and locationY <= maxY:
        inRangeFlag = True
    else:
        inRangeFlag = False

    return inRangeFlag

def dis2sensor(actor_location):
    dis = mt.sqrt(np.square(actor_location.x - CFG.Lidar['LocationX']) + np.square(actor_location.y - CFG.Lidar['LocationY']))

    return round(dis, 2)


def project_to_2d(location, intrinsic_matrix, actor, sensor_transform):
    x, y, z = location.x, location.y, location.z
    extent_x, extent_y, extent_z = actor.bounding_box.extent.x, actor.bounding_box.extent.y, actor.bounding_box.extent.z

    transform_matrix = np.array(sensor_transform.get_matrix())
    rotation_matrix = transform_matrix[:3, :3]
    translation_vector = transform_matrix[:3, 3]

    corners_3d = [
        [x + extent_x, y, z + extent_z],
        [x - extent_x, y, z + extent_z],
        [x + extent_x, y, z - extent_z],
        [x - extent_x, y, z - extent_z],
    ]


    corners_3d_camera = [np.dot(rotation_matrix, np.array(corner) - translation_vector) for corner in corners_3d]

    corners_2d = []
    for corner in corners_3d_camera:
        point_2d = np.dot(intrinsic_matrix, np.array([corner[0], corner[1], corner[2]]))
        if point_2d[2] != 0:
            point_2d = point_2d[:2] / point_2d[2]
            corners_2d.append(point_2d)

    min_x = min(corner[0] for corner in corners_2d)
    max_x = max(corner[0] for corner in corners_2d)
    min_y = min(corner[1] for corner in corners_2d)
    max_y = max(corner[1] for corner in corners_2d)

    return [min_x, min_y, max_x, max_y]



def creat_kitti_datapoint(actor, sensor, actor_type, occlusion_level, intrinsic_matrix):
    if actor:
        datapoint = CarlaDataDescriptor()
        actor_location = actor.get_location()
        sensor_location = sensor.location

        _points = project_to_2d(actor_location, intrinsic_matrix, actor, sensor)
        bbox_2d = [_points[0], _points[1], _points[2], _points[3]]

        rotation_y = get_relative_rotation_y(actor.get_transform().rotation.yaw, sensor.rotation.yaw)
        # id = actor.id
        datapoint.set_bbox(bbox_2d)
        datapoint.set_3d_object_dimensions(actor.bounding_box.extent)
        datapoint.set_type(actor_type)
        datapoint.set_3d_object_location(actor, sensor, actor_type)
        datapoint.set_rotation_y(rotation_y)
        occlusion_level = float(occlusion_level)

        datapoint.set_occlusion(occlusion_level)
        datapoint.occluded = occlusion_level


        delta_x = actor_location.x - sensor_location.x
        delta_y = actor_location.y - sensor_location.y

        angle_to_object = math.atan2(delta_y, delta_x)
        alpha = angle_to_object - math.radians(sensor.rotation.yaw)

        if alpha > math.pi:
            alpha -= 2 * math.pi
        elif alpha < -math.pi:
            alpha += 2 * math.pi
        datapoint.set_alpha(alpha)

        '''
        truncation = float(1- cur_N/15) if cur_N < 15 else 0
        datapoint.set_truncated(truncation)
        '''
        # print(actor)
        # print(datapoint)
        return datapoint
    else:
        return None

def create_kitti_datapoint_tracking(actor, sensor, actor_type, occlusion_level, frame, uid):
    if actor:
        datapoint = CarlaDataDescriptorTracking()
        bbox_2d = [0, 0, 0, 0]  # no camera so far

        rotation_y = get_relative_rotation_y(actor.get_transform().rotation.yaw, sensor.rotation.yaw)
        # id = actor.id
        datapoint.set_bbox(bbox_2d)
        datapoint.set_3d_object_dimensions(actor.bounding_box.extent)
        datapoint.set_type(actor_type)
        datapoint.set_3d_object_location(actor, sensor, actor_type)
        datapoint.set_rotation_y(rotation_y)
        datapoint.set_frame(frame)
        datapoint.set_id(uid)
        occlusion_level = float(occlusion_level)

        datapoint.set_occlusion(occlusion_level)
        datapoint.occluded = occlusion_level

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

def save_kitti_label_data_tracking(filename, datapoints):
    with open(filename, 'a') as f:
        out_str = "\n".join([str(point) for point in datapoints if point])
        if out_str:
            f.write(out_str + "\n")
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
   # TR_velodyne = np.column_stack((R_velodyne, T_velodyne))
    TR_velodyne = np.array([[0, -1, 0],
                            [0, 0, -1],
                            [1, 0, 0]])
    # Add translation vector from velo to camera. This is 0 because the position of camera and lidar is equal in our configuration.
    TR_velodyne = np.column_stack((TR_velodyne, np.array([0, 0, 0])))


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

class Mutator():
    def __init__(self, seed_path):
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
        self.controller_list = []
        self.frame = 3000
        self.mutated_frame = 0
        self.mutated_npc = None
        self.candidate_location = None
        self.ego_location = None
        self.has_mutation = False
        self.N_max = 0
        self.interactive_point = None
        self.junction = None
        self.is_collision = None
        self.route = []
        self.visibility_list = []
        self._score = 0
        self._oracle = False
        self.record_flag = False
        self.has_mutated = False
        self.map = None
        self.npc_type = None
        self.controllers_list = []
        self.start_frame = None
        self.end_frame = None
        self.first_reach = False
        self.second_reach = True
        self.npc_speed = None
        self.triggering_time = None
        self.left_win = None
        self.right_win = None
        self.coliision_list = []
        self.seed_path = seed_path
        self.origin_settings = None
        self.hash_value = None
        self.cnt = 0
        self.ego_speed = 0
        self.candidate_speed = 0
        self.candidate_id = None
        self.instance_points_dict = dict()
        self.occ_dict = dict()
        self.npc_route = None
        self.emerge_frame = None
        self.disappearance_frame = None
        self.npcs = []
        self.mutation_loc = None
        self.mutation_rot = None
        self.actor_uid_map = {}
        self.uid_counter = 0
        self.tracking_dir = None
        self.initial_behavior = 0
        self.npc_spawn_point = None
        self.distance_list = []
        self.ego_start_loc = None
        self.acc_list = np.array([])
        self.Vx_list = np.array([])
        self.Vy_list = np.array([])
        self.SWA_list = np.array([])
        self.yaw_rate_list = np.array([])
        self.run = False


    def is_emerge(self, ego_loc: carla.Location, candidate_loc: carla.Location, offset: float = 3.0) -> bool:

        ego_x, ego_y = ego_loc.x, ego_loc.y
        candidate_x, candidate_y = candidate_loc.x, candidate_loc.y

        min_x = min(ego_x, candidate_x) - offset
        max_x = max(ego_x, candidate_x) + offset
        min_y = min(ego_y, candidate_y) - offset
        max_y = max(ego_y, candidate_y) + offset

        for npc in self.npcs:
            npc_loc = npc.get_location()
            npc_x, npc_y = npc_loc.x, npc_loc.y

            if min_x <= npc_x <= max_x and min_y <= npc_y <= max_y:
                return True

        return False

    def run_sim(self, seed, _client, desired_occlusion):
        try:
            begin_time = time.time()
            self._seed = seed
            # set client
            self.client = _client
            # set weather and map
            environment = seed["environment"]
            weather_type = environment["weather_type"]
            weather = environment["weather"]
            self.weather_type = weather_type
            map = environment["map"]
            print(map)
            world = self.client.load_world(map)
            self.world = world
            self.origin_settings = world.get_settings()
            server_settings(world)
            set_weather(world, weather)
            self.blueprint_library = world.get_blueprint_library()
            self.traffic_manager = self.client.get_trafficmanager()
            self.traffic_manager.set_hybrid_physics_mode(True)
            set_all_traffic_lights_to_green(world)
            map = world.get_map()
            self.map = map
            all_spawn_points = world.get_map().get_spawn_points()

            # get junction
            self.interactive_point = map.get_waypoint_xodr(road_id=seed["Interactive_point"]["road_id"],
                                                           lane_id=seed["Interactive_point"]["land_id"],
                                                           s=seed["Interactive_point"]["s"])
            junction = self.interactive_point.get_junction()

            self.junction = junction
            mutation_type = seed["mutation_type"]
            print(f"Mutation Type: {mutation_type}")
            lane_type = lane_type_pool.get(mutation_type, carla.LaneType.Driving)

            if lane_type is None:
                raise ValueError(f"Invalid mutation_type: {seed['mutation_type']}")
            try:
                pair_routes = junction.get_waypoints(lane_type)
            except AttributeError as e:
                raise AttributeError(f"error: {str(e)}，ensure junction is a instance of carla.client.Junction ")
            except Exception as e:
                raise Exception(f"Unknown error: {str(e)}")

            pair = random.choice(pair_routes)
            initial_waypoint = pair[1]
            prev_waypoints = initial_waypoint.previous_until_lane_start(1)
            prev_waypoints = prev_waypoints[:-1]
            prev_waypoints = prev_waypoints[::-1]
            # start_waypoint = random.choice(prev_waypoints[0].previous(5))
            distance_offset = random.randint(0, 30)
            start_waypoints = prev_waypoints[0].previous(distance_offset)
            start_waypoint = start_waypoints[0]
            for _point in prev_waypoints:
                print(_point.transform.location)


            # Get the center of the junction
            total_x, total_y, total_z = 0, 0, 0
            for pairs in pair_routes:
                start_point = pairs[0].transform
                end_point = pairs[1].transform
                total_x += start_point.location.x
                total_y += start_point.location.y
                total_z += start_point.location.z
                total_x += end_point.location.x
                total_y += end_point.location.y
                total_z += end_point.location.z

            num_points = len(pair_routes) * 2
            center_loc = carla.Location(
                x=total_x / num_points,
                y=total_y / num_points,
                z=total_z / num_points
            )

            # set ego
            ego = seed["ego"]
            ego_route = []
            for _t in ego["route"]:
                point = _t["point"]
                ego_route.append(point)

            ego_spawn_point = ego["route"][0]
            ego_spawn_idx = ego_spawn_point["point"]
            ego_spawn_transform = all_spawn_points[ego_spawn_idx]
            self.ego_start_loc = ego_spawn_transform.location

            ego_bp = self.blueprint_library.find(ego["type"])
            self.ego_speed = random.uniform(10, 35)
            self.ego, ego_start_point, ego_end_point = spawn_vehicle_route(world, ego_bp, self.traffic_manager,
                                                                           ego_route,
                                                                           weather_type, ego_spawn_transform, "ego",
                                                                           self.ego_speed)
            self.actor_list.append(self.ego)

            image_queue_depth = queue.Queue()
            image_queue_lidar = queue.Queue()
            image_queue_rgb = queue.Queue()
            image_queue_segmentation = queue.Queue()

            segmentation_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
            segmentation_bp.set_attribute('image_size_x', '1248')
            segmentation_bp.set_attribute('image_size_y', '384')
            segmentation_bp.set_attribute('fov', '90')
            segmentation_transform = carla.Transform(carla.Location(0, 0, 1.6), carla.Rotation(0, 0, 0))
            camera_segmentation = self.world.spawn_actor(segmentation_bp, segmentation_transform, attach_to=self.ego)
            camera_segmentation.listen(image_queue_segmentation.put)


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

            if mutation_type is 0:
                # set candidate
                candidate = seed["candidate"]
                candidate_route = []

                for _t in candidate["route"]:
                    point = _t["point"]
                    candidate_route.append(point)

                candidate_spawn_point = candidate["route"][0]
                candidate_spawn_idx = candidate_spawn_point["point"]
                candidate_spawn_transform = all_spawn_points[candidate_spawn_idx]

                candidate_spawn_point = carla.Transform(
                    carla.Location(x=candidate["spawn_point"]["x"], y=candidate["spawn_point"]["y"],
                                   z=candidate["spawn_point"]["z"]),
                    carla.Rotation(pitch=candidate["spawn_point"]["pitch"], yaw=candidate["spawn_point"]["yaw"],
                                   roll=candidate["spawn_point"]["roll"]))

                candidate_bp = self.blueprint_library.find(candidate["type"])

                self.candidate_speed = random.uniform(10, 35)
                self.candidate, candidate_start_point, candidate_end_point = spawn_vehicle_route(world, candidate_bp,
                                                                                                 self.traffic_manager,
                                                                                                 candidate_route,
                                                                                                 weather_type,
                                                                                                 candidate_spawn_transform,
                                                                                                 "candidate",
                                                                                                 self.candidate_speed)
                self.actor_list.append(self.candidate)


            elif mutation_type is 1:
                tag = 12
                candidate = seed["candidate"]
                candidate_spawn_point = carla.Transform(
                    carla.Location(x=candidate["spawn_point"]["x"], y=candidate["spawn_point"]["y"],
                                   z=candidate["spawn_point"]["z"]),
                    carla.Rotation(pitch=candidate["spawn_point"]["pitch"], yaw=candidate["spawn_point"]["yaw"],
                                   roll=candidate["spawn_point"]["roll"]))
                candidate_bp = self.blueprint_library.find(candidate["type"])
                self.candidate = world.try_spawn_actor(candidate_bp, candidate_spawn_point)
                pedestrain_control = carla.WalkerControl()
                pedestrain_control.speed = candidate["speed"]
                pedestrain_rotation = candidate_spawn_point.rotation
                pedestrain_control.direction = pedestrain_rotation.get_forward_vector()
                self.candidate.apply_control(pedestrain_control)

            # set npc
            npc_name_pool = []

            if len(seed["npcs"]) > 0:
                self.run = True
                npcs = seed["npcs"]
                for npc_data in npcs:
                    npc_name = next(iter(npc_data))
                    npc_name_pool.append(npc_name)
                    npc_route = []

                    self.initial_behavior = 1

                    npc_spawn_point = npc_data[npc_name]["spawn_point"]

                    npc_spawn_transform = carla.Transform(carla.Location(x=npc_spawn_point["x"], y=npc_spawn_point["y"], z=0.6),
                                                          carla.Rotation(pitch=npc_spawn_point["pitch"], yaw=npc_spawn_point["yaw"], roll=npc_spawn_point["roll"]))

                    npc_bp_str = npc_data[npc_name]["type"]
                    nav_type = npc_data[npc_name]["nav_type"]

                    npc_bp = self.blueprint_library.find(npc_bp_str)

                    self.npc_type = npc_bp
                    print(npc_bp)

                    npc = world.try_spawn_actor(npc_bp, npc_spawn_transform)


                    if npc is None:
                        print(f"Failed to spawn NPC vehicle at {npc_spawn_transform.transform}")
                        raise ValueError("NPC vehicle spawn failed.")

                    if nav_type == "IMMOBILE":
                        pass

                    elif nav_type == "LINEAR":
                        npc_speed = npc_data[npc_name]["speed"]
                        npc.set_autopilot(True)
                        self.traffic_manager.set_desired_speed(npc, npc_speed)

                    elif nav_type == "MANEUVER":
                        commands = []
                        for maneuver in npc_data[npc_name]["maneuvers"]:
                            direction, speed, trigger_frame = maneuver
                            command = {
                                "direction": direction,  # （-1: left，0: straight，1: right）
                                "speed": speed,  # speed
                                "trigger_frame": trigger_frame  # triggering frame
                            }
                            commands.append(command)

                        self.npc_list.append((npc, commands))


                    elif nav_type == "AUTOPILOT":
                        npc.set_autopilot(True)
                        self.traffic_manager.set_desired_speed(npc, npc_speed)
                        path = []
                        path.append(npc_spawn_transform.location)
                        destination = carla.Location(x=npc_data[npc_name]["destination"]["x"], y=npc_data[npc_name]["destination"]["y"], z=npc_data[npc_name]["destination"]["z"])
                        path.append(destination)

                        self.traffic_manager.set_path(npc, path)

                    self.actor_list.append(npc)
                    self.npcs.append(npc)


                self.mutated_npc = random.choice(npc_name_pool)



            ego_waypoint = self.map.get_waypoint(self.ego_start_loc)
            dis = random.uniform(5, 40)
            mutation_waypoints = ego_waypoint.next(dis)
            mutation_waypoint = random.choice(mutation_waypoints)

            if mutation_waypoint is None:
                print("Warning: Could not find a valid waypoint for mutation location.")
                return

            mutation_transform = mutation_waypoint.transform
            self.mutation_loc = mutation_transform.location
            self.mutation_rot = mutation_transform.rotation

            print(f"Insert initial NPC at location: {self.mutation_loc} with rotation: {self.mutation_rot}")


            # set puddle
            if len(seed["puddles"]) > 0:
                puddles = seed["puddles"]
                for puddle_data in puddles:
                    puddle_name = next(iter(puddle_data))
                    puddle_spawn_point = puddle_data[puddle_name]["spawn_point"]
                    puddle_size = puddle_data[puddle_name]["size"]
                    puddle_level = puddle_data[puddle_name]["level"]
                    puddle_spawn_transform = carla.Transform(carla.Location(x=puddle_spawn_point["x"], y=puddle_spawn_point["y"], z=puddle_spawn_point["z"]),
                                                          carla.Rotation(pitch=puddle_spawn_point["pitch"], yaw=puddle_spawn_point["yaw"], roll=puddle_spawn_point["roll"]))

                    puddle_bp = self.blueprint_library.find("static.trigger.friction")
                    puddle_bp.set_attribute('friction', str(puddle_level))
                    puddle_bp.set_attribute('extent_x', str(puddle_size[0]))
                    puddle_bp.set_attribute('extent_y', str(puddle_size[1]))
                    puddle_bp.set_attribute('extent_z', str(puddle_size[2]))

                    if puddle_bp:
                        puddle = world.try_spawn_actor(puddle_bp, puddle_spawn_transform)
                    else:
                        print("Puddle blueprint not found!")



            queue_list = {
                "image_queue_depth_front": image_queue_depth,
                "image_queue_rgb": image_queue_rgb
            }

            depth_camera_list = {"front_depth_camera": camera_depth}
            # set collision
            collision_bp = self.blueprint_library.find("sensor.other.collision")
            for _actor in self.actor_list:
                collision_sensor = world.try_spawn_actor(
                    collision_bp, _actor.get_transform(), attach_to=_actor)
                collision_sensor.listen(self.collision_callback)

            CarlaDataProvider.set_world(world)
            world.on_tick(on_tick_wrapper)
            frame = 0
            start_flag = True
            prob = True
            emerge = False
            disappear = False

            json_str = json.dumps(seed, sort_keys=True)
            hash_object = hashlib.sha256(json_str.encode())
            self.hash_value = hash_object.hexdigest()

            if self.run:
                while frame < self.frame:
                    world.tick()
                    frame += 1
                    if prob:
                        cur_acc = self.ego.get_acceleration()
                        cur_acc = np.array([cur_acc.x, cur_acc.y, cur_acc.z])
                        cur_acc_magnitude = np.linalg.norm(cur_acc)
                        self.acc_list = np.append(self.acc_list, cur_acc_magnitude)

                        cur_vel = self.ego.get_velocity()
                        self.Vx_list = np.append(self.Vx_list, cur_vel.x)
                        self.Vy_list = np.append(self.Vy_list, cur_vel.y)

                        cur_control = self.ego.get_control()
                        steer_input = cur_control.steer
                        max_steering_angle = 30
                        SWA = steer_input * max_steering_angle
                        self.SWA_list = np.append(self.SWA_list, SWA)

                        cur_angular_vel = self.ego.get_angular_velocity()
                        yaw_rate = cur_angular_vel.z
                        self.yaw_rate_list = np.append(self.yaw_rate_list, yaw_rate)

                        # set spectator
                        spectator = world.get_spectator()
                        ego_transform = self.ego.get_transform()
                        spectator_location = ego_transform.location + carla.Location(z=60)
                        spectator_rotation = carla.Rotation(-90, 90, 0)
                        spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))

                        ego_loc = self.ego.get_location()
                        candidate_loc = self.candidate.get_location()
                        npc_loc = npc.get_location()

                        cur_distance = npc_loc.distance(ego_loc)
                        self.distance_list.append(cur_distance)

                        candidate_points = generate_bbox(self.candidate, camera_lidar.get_transform())
                        npc_points = generate_bbox(npc, camera_lidar.get_transform())

                        # set npc control
                        if len(self.npc_list) > 0:
                            for npc_instance, commands in self.npc_list:
                                control = carla.VehicleControl()
                                control.throttle = 0.0
                                control.steer = 0.0
                                control.brake = 0.0

                                for command in commands:
                                    if command["trigger_frame"] == frame:
                                        control.throttle = command["speed"] / 10.0
                                        control.steer = command["direction"] * 0.3

                                        npc_instance.apply_control(control)

                        candidate_pcl = generate_surface_pointcloud_from_obb(candidate_points)
                        npc_pcl = generate_surface_pointcloud_from_obb(npc_points)
                        merged_pcl = o3d.geometry.PointCloud()
                        merged_pcl.points = o3d.utility.Vector3dVector(
                            list(candidate_pcl.points) + list(npc_pcl.points))

                        points, colors, intrinsic, centroid = get_ground_truth(queue_list, depth_camera_list,
                                                                               image_queue_segmentation)

                        candidate_gt_image, candidate_gt_vis = project_3d_to_2d_image(candidate_pcl,
                                                                                      camera_rgb.get_transform(),
                                                                                      intrinsic)
                        merge_gt_image, merge_gt_vis = project_3d_to_2d_image(merged_pcl, camera_rgb.get_transform(),
                                                                              intrinsic)

                        npc_gt_image, npc_gt_vis = project_3d_to_2d_image(npc_pcl, camera_rgb.get_transform(),
                                                                          intrinsic)
                        overlap_count = get_occlusion(candidate_gt_image, npc_gt_image)
                        '''
                        cur_occ = overlap_count / candidate_gt_vis if candidate_gt_vis not in (None, 0) else 0
                        cur_occ = np.clip(cur_occ, 0, 1)
                        '''
                        occluders = []
                        occluders.append(npc_points)
                        sensor_loc = camera_rgb.get_location()
                        sensor_loc = np.array([sensor_loc.x, sensor_loc.y, sensor_loc.z])

                        results_dict = ray_tracing(candidate_pcl, candidate_points, occluders, sensor_loc)

                        occluded_rays = results_dict['num_occluded']
                        free_rays = results_dict['num_free']
                        cur_occ = occluded_rays / (free_rays + occluded_rays)
                        cur_occ = np.clip(cur_occ, 0, 1)
                        print(f'frame:{frame}, results: {results_dict}, occ:{cur_occ}')

                        self.occ_dict[frame] = cur_occ

                        if self.is_emerge(ego_loc, candidate_loc) and not emerge:
                            self.emerge_frame = frame
                            emerge = True

                        if emerge and not self.is_emerge(ego_loc, candidate_loc) and not disappear:
                            self.disappearance_frame = frame
                            disappear = True

                        if not self.has_mutated and (
                                ego_loc.distance(center_loc) < random.randint(20, 50) or candidate_loc.distance(
                            center_loc) < random.randint(20, 50)
                                or ego_loc.distance(npc_loc) < random.randint(15, 30) or candidate_loc.distance(
                            npc_loc) < random.randint(15, 30)):
                            self.triggering_time = frame
                            self.has_mutated = True

                        if not self.first_reach and isInRange(self.candidate.get_location(), self.ego.get_location()):
                            self.start_frame = frame
                            self.end_frame = self.start_frame + 200
                            self.first_reach = True

                        # decide on the end frame
                        if self.first_reach and not isInRange(self.candidate.get_location(), self.ego.get_location()):
                            self.second_reach = False

                        # stop route
                        end_location = ego_end_point.location
                        distance_to_end = ego_loc.distance(end_location)

                        if self.end_frame is not None and frame > self.end_frame:
                            self.traffic_manager.set_desired_speed(self.ego, 0)
                            self.traffic_manager.set_desired_speed(self.candidate, 0)
                            prob = False
                            break




        finally:
            if self.end_frame is None and self.run:
                self.end_frame = frame
            print("Finish Initialization")
            self.origin_settings.synchronous_mode = False
            self.world.apply_settings(self.origin_settings)


    def collision_callback(self, collision_event):

        # print(f"collision position: {collision_event.normal_impulse}， collision object: {collision_event.other_actor.id}")
        self.is_collision = True

    def clean(self):
        for sensor in self.sensor_list:
            sensor.stop()
            sensor.destroy()

        for actor in self.actor_list:
            actor.destroy()

        for npc in self.npc_list:
            npc[1].destroy()

        print(f"All cleaned up!")

    def find_occlusion_win(self, desired_occlusion):
        if self.start_frame is None or self.emerge_frame is None:
            return 0, 200

        frame = max(self.emerge_frame, self.start_frame)
        left_flag = False
        right_flag = False
        left_frame = 0
        right_frame = 0
        while frame < min(self.disappearance_frame, self.end_frame) and not left_flag:
            if self.occ_dict[frame]> 0:
                left_flag = True
                left_frame = frame

            frame = frame + 1
        if not left_flag:
            return None, None

        while frame < min(self.disappearance_frame, self.end_frame) and not right_flag:
            if self.occ_dict[frame] <=0.05:
                right_flag = True
                right_frame = frame
            frame = frame + 1
        if not right_flag:
            return left_frame, self.disappearance_frame
        else:
            return left_frame, right_frame

    def check_hard_acc(self):
        return np.count_nonzero(self.acc_list >= 21.2 / 10)

    def check_hard_braking(self):
        return np.count_nonzero(self.acc_list <= -21.2 / 10)

    def check_hard_turn(self):
        hard_turn_min_sa = 20
        hard_turn_thres = 0.18
        hard_turn_indicator = np.divide(
            self.Vy_list,
            self.SWA_list,
            out=np.zeros_like(self.Vy_list),
            where=abs(self.SWA_list) > hard_turn_min_sa
        )
        return np.sum(abs(hard_turn_indicator) > hard_turn_thres)

    def butter_lowpass_filter(self, data, cutoff_freq, nyq_freq, order=4):
        b, a = signal.butter(order, cutoff_freq / nyq_freq, btype='lowpass')
        return signal.filtfilt(b, a, data)

    def get_oversteer_level(self, sa, la, yr):

        x_sa = np.arange(0, 55, 5)
        x_la = np.arange(0, 0.55, 0.05)
        x_yr = np.arange(0, 50, 5)  #
        x_os = np.arange(0, 11, 1)


        sa_low = skfuzzy.trimf(x_sa, [0, 0, 20])
        sa_med = skfuzzy.trimf(x_sa, [5, 25, 45])
        sa_high = skfuzzy.trimf(x_sa, [30, 50, 50])

        la_low = skfuzzy.trimf(x_la, [0, 0, 0.2])
        la_med = skfuzzy.trimf(x_la, [0.05, 0.25, 0.45])
        la_high = skfuzzy.trimf(x_la, [0.3, 0.5, 0.5])

        yr_low = skfuzzy.trimf(x_yr, [0, 0, 17.5])
        yr_med = skfuzzy.trimf(x_yr, [4, 22.5, 41])
        yr_high = skfuzzy.trimf(x_yr, [27.5, 45, 45])

        os_low = skfuzzy.trimf(x_os, [0, 0, 4])
        os_med = skfuzzy.trimf(x_os, [1, 5, 9])
        os_high = skfuzzy.trimf(x_os, [6, 10, 10])

        rule1 = np.fmin(sa_low, la_low)
        os_rule1 = np.fmin(rule1, os_low)
        rule2 = np.fmin(sa_med, la_med)
        os_rule2 = np.fmin(rule2, os_med)
        rule3 = np.fmin(sa_high, la_high)
        os_rule3 = np.fmin(rule3, os_high)

        os_rule4 = np.fmin(yr_low, os_low)
        os_rule5 = np.fmin(yr_med, os_med)
        os_rule6 = np.fmin(yr_high, os_high)

        os_aggregated = np.fmax(os_rule1, np.fmax(os_rule2, np.fmax(os_rule3,
                                                                    np.fmax(os_rule4, np.fmax(os_rule5, os_rule6)))))


        try:
            os_level = skfuzzy.defuzz(x_os, os_aggregated, "centroid")
        except AssertionError:
            os_level = 0

        return os_level

    def get_understeer_level(self, fd):
        x_fd = np.arange(0, 1.1, 0.1)  # 0 ~ 1
        x_us = np.arange(0, 11, 1)  # 0 ~ 10
        fd_high = skfuzzy.trimf(x_fd, [0, 1, 1])
        us_high = skfuzzy.trimf(x_us, [0, 10, 10])
        fd_level_high = skfuzzy.interp_membership(x_fd, fd_high, fd)
        us_rule = np.fmin(fd_level_high, us_high)

        try:
            us_level = skfuzzy.defuzz(x_us, us_rule, "centroid")
        except AssertionError:
            us_level = 0

        return us_level

    def get_vx_light(self):
        Vx_light = self.butter_lowpass_filter(
            self.Vx_list,
            3.5,
            5
        )

        return Vx_light

    def get_ay_list(self):
        Ay_list = np.diff(self.Vy_list)  # idx: frame, unit: km/h'frame (== km/s'1/20sec)
        kmphsec_to_g = 0.028325450360498
        Ay_list = 10 * Ay_list * kmphsec_to_g
        return Ay_list

    def get_ay_diff_list(self):
        # sample_rate = c.FRAME_RATE # there are FPS (e.g., 20) ticks in one sec
        Ay_light = self.butter_lowpass_filter(
            self.Ay_list,
            3.5,
            5
        )

        Ay_light_heavy = self.butter_lowpass_filter(
            Ay_light,
            0.5,
            5
        )

        # for OS detection
        Ay_diff_list = np.absolute(Ay_light - Ay_light_heavy)

        return Ay_diff_list

    def get_ay_heavy(self, Ay_list):
        Ay_ext = np.append(Ay_list, 0)  # sync the size with sa array

        Ay_heavy = np.absolute(self.butter_lowpass_filter(
            Ay_ext,
            0.5,
            5
        ))

        return Ay_heavy

    def get_swa_diff_list(self):
        SWA_light = self.butter_lowpass_filter(
            self.SWA_list,
            3.5,
            5
        )

        SWA_heavy = self.butter_lowpass_filter(
            SWA_light,
            0.5,
            5
        )

        # For OS
        SWA_diff_list = np.absolute(SWA_light - SWA_heavy)

        return SWA_diff_list

    def get_swa_heavy(self):
        SWA_heavy_list = np.absolute(self.butter_lowpass_filter(
            self.SWA_list,
            0.5,
            5
        ))

        return SWA_heavy_list

    def get_ay_gain(self, SWA_heavy_list, Ay_heavy):
        Ay_gain = np.absolute(
            np.divide(
                Ay_heavy,
                SWA_heavy_list,
                out=np.zeros_like(SWA_heavy_list),
                where=SWA_heavy_list > 0
            )
        )

        return Ay_gain

    def get_ay_peak(self, Ay_gain):
        Ay_peak = np.zeros_like(Ay_gain)
        d_Ay_gain = np.diff(Ay_gain)
        peak = 0
        for i, d in enumerate(d_Ay_gain):
            if d >= 0:
                peak = Ay_gain[i]
            Ay_peak[i] = peak

        return Ay_peak

    def get_frac_drop(self, Ay_gain, Ay_peak):
        frac_drop = np.divide(
            (Ay_gain - Ay_peak),
            Ay_peak,
            out=np.zeros_like(Ay_peak),
            where=Ay_peak > 0
        )

        return frac_drop

    def get_abs_yr(self, yaw_rate_list):
        yr = np.array(yaw_rate_list)
        yr = self.butter_lowpass_filter(
            yr,
            0.5,
            5
        )
        abs_yr = np.absolute(yr)

        # XXX: should decide whether we should apply the delay factor on
        # Ay and abs_yr.

        return abs_yr

    def calculate_driving_quality(self):

        deductions = 0

        ha = self.check_hard_acc()
        hb = self.check_hard_braking()
        ht = self.check_hard_turn()

        deductions += ha + hb + ht

        Vx_light = self.get_vx_light()
        Ay_list = self.get_ay_list()
        Ay_diff_list = self.get_ay_diff_list()
        SWA_diff_list = self.get_swa_diff_list()
        SWA_heavy_list = self.get_swa_heavy()
        abs_yr = self.get_abs_yr(self.yaw_rate_list)
        frac_drop = self.get_frac_drop(self.get_ay_gain(SWA_heavy_list, self.get_ay_heavy(Ay_list)),
                                       self.get_ay_peak(Ay_list))

        num_oversteer = 0
        num_understeer = 0

        for Vx, SWA_diff, Ay_diff, yr, SWA_heavy, fd in zip(Vx_light, SWA_diff_list, Ay_diff_list, abs_yr,
                                                            SWA_heavy_list, frac_drop):

            os_level = self.get_oversteer_level(SWA_diff, Ay_diff, yr)
            if os_level >= 4 and Vx > 5 and Ay_diff > 0.1:
                num_oversteer += 1

            us_level = self.get_understeer_level(fd)
            if us_level >= 4 and Vx > 5 and SWA_heavy > 10:
                num_understeer += 1

        deductions += num_oversteer + num_understeer

        return deductions

    def mutate_weather(self, desired_occlusion):
        mutated_seed = copy.deepcopy(self._seed)
        mutated_seed['N'] = self.N_max
        mutated_seed['start_frame'] = self.start_frame
        mutated_seed['end_frame'] = self.end_frame
        mutated_seed['cnt'] = 0
        mutated_seed['ego_speed'] = self.ego_speed
        mutated_seed['candidate_speed'] = self.candidate_speed
        mutated_seed['emerge_frame'] = self.emerge_frame
        mutated_seed['disappear_frame'] = self.disappearance_frame


        weather = mutated_seed['environment']['weather']
        weather["cloudiness"] = random.randint(0, 100)
        weather["precipitation"] = random.randint(0, 100)
        weather["precipitation_deposits"] = random.randint(0, 100)
        weather["wind_intensity"] = random.randint(0, 100)
        weather["fog_density"] = random.randint(0, 100)
        weather["wetness"] = random.randint(0, 100)
        weather["sun_azimuth_angle"] = random.randint(0, 360)
        weather["sun_altitude_angle"] = random.randint(-90, 90)

        self.left_win, self.right_win = self.find_occlusion_win(desired_occlusion)


        mutated_seed['left_window'] = self.left_win
        mutated_seed['right_window'] = self.right_win

        mean_score = 0


        for i in range(self.left_win, self.right_win):
            occ_ = self.occ_dict.get(i, 0 )

            mean_score += occ_

        mean_score /= self.right_win - self.left_win

        index = bisect.bisect_right(occlusion_levels, mean_score) - 1

        if index < 0:
            index = 0

        occlusion_level = occlusion_levels[index]

        fitness = self.calculate_driving_quality()

        return mutated_seed, mean_score, "Weather Mutation", fitness


    def add_player(self, location, rotation, desired_occlusion):
        mutated_seed = copy.deepcopy(self._seed)
        mutated_seed['N'] = self.N_max
        mutated_seed['start_frame'] = self.start_frame
        mutated_seed['end_frame'] = self.end_frame
        mutated_seed['cnt'] = 0
        mutated_seed['ego_speed'] = self.ego_speed
        mutated_seed['candidate_speed'] = self.candidate_speed
        mutated_seed['emerge_frame'] = self.emerge_frame
        mutated_seed['disappear_frame'] = self.disappearance_frame

        npc = None
        choice = random.randint(0, 3)
        if choice == 0:
            # add immobile npc
            if 5 < location.distance(self.ego_start_loc) < 40:
                npc_bp_str = random.choice(vehicle_pools)
                npc_bp = self.blueprint_library.find(npc_bp_str)
                vehicle_type = npc_bp.id
                npc = {
                    "spawn_point": {
                        "x": location.x,
                        "y": location.y,
                        "z": 1.5,
                        "pitch": rotation.pitch,
                        "yaw": rotation.yaw,
                        "roll": rotation.roll
                    },
                    "type": vehicle_type,
                    "nav_type": "IMMOBILE"
                }
                print("add player:IMMOBILE")
                npc_name = f"npc{len(mutated_seed['npcs']) + 1}"
                mutated_seed["npcs"].append({npc_name: npc})

        elif choice == 1:
            # add linear npc
            if 5 < location.distance(self.ego_start_loc) < 40:
                npc_speed = random.uniform(0, 35)
                npc_bp_str = random.choice(vehicle_pools)
                npc_bp = self.blueprint_library.find(npc_bp_str)
                vehicle_type = npc_bp.id
                npc = {
                    "spawn_point": {
                        "x": location.x,
                        "y": location.y,
                        "z": 1.5,
                        "pitch": rotation.pitch,
                        "yaw": rotation.yaw,
                        "roll": rotation.roll
                    },
                    "type": vehicle_type,
                    "nav_type": "LINEAR",
                    "speed": npc_speed
                }
                print("add player:LINEAR")
                npc_name = f"npc{len(mutated_seed['npcs']) + 1}"
                mutated_seed["npcs"].append({npc_name: npc})

        elif choice == 2:
            # add maneuver npc
            if 5 < location.distance(self.ego_start_loc) < 40:
                npc_speed = random.uniform(0, 35)
                npc_bp_str = random.choice(vehicle_pools)
                npc_bp = self.blueprint_library.find(npc_bp_str)
                vehicle_type = npc_bp.id

                maneuvers = [
                    [0, 0, 0],
                    [0, 8, 0],
                    [0, 8, 0],
                    [0, 8, 0],
                    [0, 8, 0],
                ]

                npc = {
                    "spawn_point": {
                        "x": location.x,
                        "y": location.y,
                        "z": 1.5,
                        "pitch": rotation.pitch,
                        "yaw": rotation.yaw,
                        "roll": rotation.roll
                    },
                    "type": vehicle_type,
                    "nav_type": "MANEUVER",
                    "maneuvers": maneuvers
                }
                print("add player:MANEUVER")
                npc_name = f"npc{len(mutated_seed['npcs']) + 1}"
                mutated_seed["npcs"].append({npc_name: npc})

        elif choice == 3:
            # add autopilot npc
            if 5 < location.distance(self.ego_start_loc) < 40:
                npc_speed = random.uniform(0, 35)
                npc_bp_str = random.choice(vehicle_pools)
                npc_bp = self.blueprint_library.find(npc_bp_str)
                vehicle_type = npc_bp.id
                spawn_points = self.map.get_spawn_point()
                destination = random.choice(spawn_points)
                dest_loc = destination.location

                npc = {
                    "spawn_point": {
                        "x": location.x,
                        "y": location.y,
                        "z": 1.5,
                        "pitch": rotation.pitch,
                        "yaw": rotation.yaw,
                        "roll": rotation.roll
                    },
                    "type": vehicle_type,
                    "nav_type": "AUTOPILOT",
                    "destination": {
                        "x": dest_loc.x,
                        "y": dest_loc.y,
                        "z": dest_loc.z
                    }
                }
                print("add player:AUTOPILOT")
                npc_name = f"npc{len(mutated_seed['npcs']) + 1}"
                mutated_seed["npcs"].append({npc_name: npc})

        if len(self.npcs) > 0:
            self.left_win, self.right_win = self.find_occlusion_win(desired_occlusion)

        else:

            return mutated_seed, None, None, None

        mutated_seed['left_window'] = self.left_win
        mutated_seed['right_window'] = self.right_win

        mean_score = 0
        fitness = 0
        if len(self.npcs) > 0:
            for i in range(self.left_win, self.right_win):
                occ_ = self.occ_dict.get(i, 0)

                mean_score += occ_

            mean_score /= self.right_win - self.left_win

            index = bisect.bisect_right(occlusion_levels, mean_score) - 1

            if index < 0:
                index = 0

            occlusion_level = occlusion_levels[index]

            fitness = self.calculate_driving_quality()


        return mutated_seed, mean_score, "Add player Mutation", fitness


    def mutate_maneuvers(self, desired_occlusion):
        mutated_seed = copy.deepcopy(self._seed)
        mutated_seed['N'] = self.N_max
        mutated_seed['start_frame'] = self.start_frame
        mutated_seed['end_frame'] = self.end_frame
        mutated_seed['cnt'] = 0
        mutated_seed['ego_speed'] = self.ego_speed
        mutated_seed['candidate_speed'] = self.candidate_speed
        mutated_seed['emerge_frame'] = self.emerge_frame
        mutated_seed['disappear_frame'] = self.disappearance_frame


        mutated_npc = None
        for npc in mutated_seed['npcs']:
            if self.mutated_npc in npc:
                mutated_npc = npc[self.mutated_npc]
                break

        if mutated_npc is not None and mutated_npc["nav_type"] == "MANEUVER":
            i = random.randint(0, 4)
            speed = random.randint(0, 10)
            direction = random.randint(-1, 1)
            maneuvers = mutated_npc["maneuvers"]
            maneuvers[i] = [direction, speed, 0]
            mutated_npc["maneuvers"] = maneuvers

        self.left_win, self.right_win = self.find_occlusion_win(desired_occlusion)


        mutated_seed['left_window'] = self.left_win
        mutated_seed['right_window'] = self.right_win

        mean_score = 0


        for i in range(self.left_win, self.right_win):
            occ_ = self.occ_dict.get(i, 0 )

            mean_score += occ_

        mean_score /= self.right_win - self.left_win

        index = bisect.bisect_right(occlusion_levels, mean_score) - 1

        if index < 0:
            index = 0

        occlusion_level = occlusion_levels[index]

        fitness = self.calculate_driving_quality()

        return mutated_seed, mean_score, "Change Maneuver Mutation", fitness



    def add_puddle(self, location,  desired_occlusion):

        mutated_seed = copy.deepcopy(self._seed)
        mutated_seed['N'] = self.N_max
        mutated_seed['start_frame'] = self.start_frame
        mutated_seed['end_frame'] = self.end_frame
        mutated_seed['cnt'] = 0
        mutated_seed['ego_speed'] = self.ego_speed
        mutated_seed['candidate_speed'] = self.candidate_speed
        mutated_seed['emerge_frame'] = self.emerge_frame
        mutated_seed['disappear_frame'] = self.disappearance_frame


        if 5 < location.distance(self.ego_start_loc) < 40:
            level = random.uniform(0.0,1.0)
            size_x = random.uniform(0.0,1.0)
            size_y = random.uniform(0.0,1.0)
            size_z = random.uniform(0.0,1.0)

            puddle = {
                "level": level,
                "size": {
                    "x": size_x,
                    "y": size_y,
                    "z": size_z
                },
                "spawn_point": {
                    "x": location.x,
                    "y": location.y,
                    "z": location.z,
                    "pitch": 0,
                    "yaw": 0,
                    "roll": 0
                }
            }

            puddle_name = f"puddle{len(mutated_seed['puddles']) + 1}"
            mutated_seed["puddles"].append({puddle_name: puddle})

        self.left_win, self.right_win = self.find_occlusion_win(desired_occlusion)


        mutated_seed['left_window'] = self.left_win
        mutated_seed['right_window'] = self.right_win

        mean_score = 0


        for i in range(self.left_win, self.right_win):
            occ_ = self.occ_dict.get(i, 0 )

            mean_score += occ_

        mean_score /= self.right_win - self.left_win

        index = bisect.bisect_right(occlusion_levels, mean_score) - 1

        if index < 0:
            index = 0

        occlusion_level = occlusion_levels[index]

        fitness = self.calculate_driving_quality()

        return mutated_seed, mean_score, "Add puddle Mutation", fitness


    def excuate_drivefuzz_muation(self, desired_occlusion):
        if len(self.npcs) <=0 :
            return self.add_player(self.mutation_loc, self.mutation_rot, desired_occlusion)

        for i in range(4):
            if i == 0:
                return self.mutate_weather(desired_occlusion)
            elif i == 1:
                return self.add_puddle(self.mutation_loc, desired_occlusion)
            elif i == 2:
                return self.add_player(self.mutation_loc, self.mutation_rot, desired_occlusion)
            elif i == 3:
                return self.mutate_maneuvers(desired_occlusion)



    def update_scenario(self, desired_occlusion):
        self._seed['N'] = self.N_max
        self._seed['start_frame'] = self.start_frame
        self._seed['end_frame'] = self.end_frame
        self._seed['cnt'] = 0
        self._seed['ego_speed'] = self.ego_speed
        self._seed['candidate_speed'] = self.candidate_speed
        self._seed['emerge_frame'] = self.emerge_frame
        self._seed['disappear_frame'] = self.disappearance_frame

        self.left_win, self.right_win = self.find_occlusion_win(desired_occlusion)

        self._seed['left_window'] = self.left_win
        self._seed['right_window'] = self.right_win