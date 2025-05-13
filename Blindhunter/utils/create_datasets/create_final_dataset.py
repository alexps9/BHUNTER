import os
import random
from os import listdir
from shutil import move
from os import mkdir

sensor_folders = ["rgb", "depth", "lidar", "ground_truth"]
metre = "NightCloudy"

out_dir = "../../DataSets_final/" + metre
in_dir = "../../valid_DataSets/" + metre
aux_dir = "../../valid_DataSets/" + metre + "/aux"

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

train_path = f"{out_dir}/train"
validation_path = f"{out_dir}/validation"
test_path = f"{out_dir}/test"


def create_folders():
    # Create folders if they don't exist

    mkdir(train_path) if not os.path.exists(train_path) else None
    mkdir(validation_path) if not os.path.exists(validation_path) else None
    mkdir(test_path) if not os.path.exists(test_path) else None
    for sensor_folder in sensor_folders:
        mkdir(f"{train_path}/{sensor_folder}") if not os.path.exists(f"{train_path}/{sensor_folder}") else None
        mkdir(f"{validation_path}/{sensor_folder}") if not os.path.exists(
            f"{validation_path}/{sensor_folder}") else None
        mkdir(f"{test_path}/{sensor_folder}") if not os.path.exists(f"{test_path}/{sensor_folder}") else None


def agrupate_data(folder):
    rgb = [f for f in sorted(listdir(in_dir + "/" + folder + "/rgb"))]
    depth = [f for f in sorted(listdir(in_dir + "/" + folder + "/depth"))]
    lidar = [f for f in sorted(listdir(in_dir + "/" + folder + "/lidar"))]
    ground_truth = [f for f in sorted(listdir(in_dir + "/" + folder + "/ground_truth"))]

    print(f"{len(rgb)}, {len(depth)}, {len(lidar)}, {len(ground_truth)}")

    for i in range(len(rgb)):
        move(f"{in_dir}/{folder}/rgb/{rgb[i]}", f"{aux_dir}/rgb/{rgb[i]}")
        move(f"{in_dir}/{folder}/depth/{depth[i]}", f"{aux_dir}/depth/{depth[i]}")
        move(f"{in_dir}/{folder}/lidar/{lidar[i]}", f"{aux_dir}/lidar/{lidar[i]}")
        move(f"{in_dir}/{folder}/ground_truth/{ground_truth[i]}", f"{aux_dir}/ground_truth/{ground_truth[i]}")

    # count the number of files in the folder
    rgb_num = len([f for f in listdir(aux_dir + "/rgb")])
    depth_num = len([f for f in listdir(aux_dir + "/depth")])
    lidar_num = len([f for f in listdir(aux_dir + "/lidar")])
    ground_truth_num = len([f for f in listdir(aux_dir + "/ground_truth")])

    print(f"Total Aux folder: {rgb_num} RGB | {depth_num} Depth | {lidar_num} Lidar | {ground_truth_num} Ground Truth")


def main(num_samples):
    rgb = [f for f in sorted(listdir(aux_dir + "/rgb"))]
    depth = [f for f in sorted(listdir(aux_dir + "/depth"))]
    lidar = [f for f in sorted(listdir(aux_dir + "/lidar"))]
    ground_truth = [f for f in sorted(listdir(aux_dir + "/ground_truth"))]

    # Shuffle all the data
    indices = list(range(len(rgb)))
    # Shuffle the indices
    random.shuffle(indices)

    # Apply the shuffled indices to reorder each list
    rgb_data = [rgb[i] for i in indices]
    depth_data = [depth[i] for i in indices]
    lidar_data = [lidar[i] for i in indices]
    ground_truth_data = [ground_truth[i] for i in indices]

    # Save all the data
    # Number of samples for train, validation and test
    num_samples_train = int(num_samples * train_ratio)
    num_samples_val = int(num_samples * val_ratio)
    num_samples_test = int(num_samples * test_ratio)
    print(f"num_samples_train: {num_samples_train}")
    print(f"num_samples_validation: {num_samples_val}")
    print(f"num_samples_test: {num_samples_test}")

    # Train + Validation
    num_samples_train_val = num_samples_train + num_samples_val
    depth_data_train_val = depth_data[:num_samples_train_val]
    rgb_data_train_val = rgb_data[:num_samples_train_val]
    lidar_data_train_val = lidar_data[:num_samples_train_val]
    ground_truth_train_val = ground_truth_data[:num_samples_train_val]

    depth_train_data = depth_data_train_val[:num_samples_train]
    rgb_train_data = rgb_data_train_val[:num_samples_train]
    lidar_train_data = lidar_data_train_val[:num_samples_train]
    ground_truth_train_data = ground_truth_train_val[:num_samples_train]
    print(f"\n-----------TRAIN-----------")
    print(f"depth_train_data: {len(depth_train_data)}")
    print(f"rgb_train_data: {len(rgb_train_data)}")
    print(f"lidar_train_data: {len(lidar_train_data)}")
    print(f"ground_truth_train_data: {len(ground_truth_train_data)}")

    depth_val_data = depth_data_train_val[num_samples_train:]
    rgb_val_data = rgb_data_train_val[num_samples_train:]
    lidar_val_data = lidar_data_train_val[num_samples_train:]
    ground_truth_val_data = ground_truth_train_val[num_samples_train:]
    print(f"\n-----------VALIDATION-----------")
    print(f"depth_val_data: {len(depth_val_data)}")
    print(f"rgb_val_data: {len(rgb_val_data)}")
    print(f"lidar_val_data: {len(lidar_val_data)}")
    print(f"ground_truth_val_data: {len(ground_truth_val_data)}")

    depth_test_data = depth_data[num_samples_train_val:]
    rgb_test_data = rgb_data[num_samples_train_val:]
    lidar_test_data = lidar_data[num_samples_train_val:]
    ground_truth_test_data = ground_truth_data[num_samples_train_val:]
    print(f"\n-----------TEST-----------")
    print(f"depth_test_data: {len(depth_test_data)}")
    print(f"rgb_test_data: {len(rgb_test_data)}")
    print(f"lidar_test_data: {len(lidar_test_data)}")
    print(f"ground_truth_test_data: {len(ground_truth_test_data)}")

    for image in depth_train_data:
        move(f"{aux_dir}/depth/{image}", f"{train_path}/depth")
    for image in rgb_train_data:
        move(f"{aux_dir}/rgb/{image}", f"{train_path}/rgb")
    for image in lidar_train_data:
        move(f"{aux_dir}/lidar/{image}", f"{train_path}/lidar")
    for image in ground_truth_train_data:
        move(f"{aux_dir}/ground_truth/{image}", f"{train_path}/ground_truth")

    for image in depth_val_data:
        move(f"{aux_dir}/depth/{image}", f"{validation_path}/depth")
    for image in rgb_val_data:
        move(f"{aux_dir}/rgb/{image}", f"{validation_path}/rgb")
    for image in lidar_val_data:
        move(f"{aux_dir}/lidar/{image}", f"{validation_path}/lidar")
    for image in ground_truth_val_data:
        move(f"{aux_dir}/ground_truth/{image}", f"{validation_path}/ground_truth")

    for image in depth_test_data:
        move(f"{aux_dir}/depth/{image}", f"{test_path}/depth")
    for image in rgb_test_data:
        move(f"{aux_dir}/rgb/{image}", f"{test_path}/rgb")
    for image in lidar_test_data:
        move(f"{aux_dir}/lidar/{image}", f"{test_path}/lidar")
    for image in ground_truth_test_data:
        move(f"{aux_dir}/ground_truth/{image}", f"{test_path}/ground_truth")

    print(
        f"\nTrain: {len(depth_train_data)} depth, {len(rgb_train_data)} rgb, {len(lidar_train_data)} lidar, {len(ground_truth_train_data)} ground_truth")
    print(
        f"Validation: {len(depth_val_data)} depth, {len(rgb_val_data)} rgb, {len(lidar_val_data)} lidar, {len(ground_truth_val_data)} ground_truth")
    print(
        f"Test: {len(depth_test_data)} depth, {len(rgb_test_data)} rgb, {len(lidar_test_data)} lidar, {len(ground_truth_test_data)} ground_truth")


if __name__ == '__main__':
    agrupate_data("T1R1_750_" + metre)
    agrupate_data("T1R2_750_" + metre)
    agrupate_data("T1R3_750_" + metre)
    agrupate_data("T1R4_750_" + metre)
    agrupate_data("T2R1_750_" + metre)
    agrupate_data("T2R2_750_" + metre)
    agrupate_data("T2R3_750_" + metre)
    agrupate_data("T2R4_750_" + metre)

    main(6000)