Lidar = {
    "LocationX": -64.0,
    "LocationY": 7.0,
    "LocationZ": 3.74,
    "Pitch": 0,
    "Yaw": 0,
    "Roll": 0
}

Lidar_Town03 = {
    "LocationX": -95,
    "LocationY": 120,
    "LocationZ": 2.74,
    "Pitch": 0,
    "Yaw": 0,
    "Roll": 0
}

Lidar_KITTI = {
    "LocationX": -93,
    "LocationY": 124,
    "LocationZ": 1.73,
    "Pitch": 0,
    "Yaw": 0,
    "Roll": 0
}

Camera = {
    "LocationX": Lidar["LocationX"],
    "LocationY": Lidar["LocationY"] + 10,
    "LocationZ": Lidar["LocationZ"] + 3,
    "Pitch": -20,
    "Yaw": 0,
    "Roll": 0
}

Camera_Town03 = {
    "LocationX": Lidar["LocationX"] +18,
    "LocationY": Lidar["LocationY"] ,
    "LocationZ": 30,
    "Pitch": -90,
    "Yaw": 0,
    "Roll": 0
}

Vehicle = {
    "LocationX": -52.5,
    "LocationY": -15.0,
    "LocationZ": 0.5,
    "Pitch": 0,
    "Yaw": 90,
    "Roll": 0
}

# Lidar can be saved in bin to comply to kitti, or the standard .ply format
LIDAR_DATA_FORMAT = "bin"
assert LIDAR_DATA_FORMAT in [
    "bin", "ply"], "Lidar data format must be either bin or ply"

""" CARLA SETTINGS """
LIDAR_HEIGHT_POS = Lidar["LocationZ"]

Lidar_label_range = {
    "minX": Lidar["LocationX"],
    "maxX": Lidar["LocationX"] + 50,
    "minY": Lidar["LocationY"] - 25,
    "maxY": Lidar["LocationY"] + 25
}

Lidar_label_range_100x100 = {
    "minX": Lidar["LocationX"] - 50,
    "maxX": Lidar["LocationX"] + 50,
    "minY": Lidar["LocationY"] - 50,
    "maxY": Lidar["LocationY"] + 50
}
