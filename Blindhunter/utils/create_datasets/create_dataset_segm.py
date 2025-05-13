import os
from os import mkdir, listdir
from shutil import move
import os
import random

data_dir = "../data"
out_dir = "../_out/lidarSegm"

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

sensor_folders = ["lidarSegm"]

train_path = f"{data_dir}/train"
validation_path = f"{data_dir}/validation"
test_path = f"{data_dir}/test"


def create_folders():
    # Create train, validation and test folders if they don't exist
    mkdir(train_path) if not os.path.exists(train_path) else None
    mkdir(validation_path) if not os.path.exists(validation_path) else None
    mkdir(test_path) if not os.path.exists(test_path) else None


def create_dataset(num_samples):
    # Save the lidar data
    lidar_data = [f for f in sorted(listdir(f"{out_dir}"))]
    random.seed(42)
    random.shuffle(lidar_data)

    # Number of samples for train, validation and test
    num_samples_train = int(num_samples * train_ratio)
    num_samples_val = int(num_samples * val_ratio)
    num_samples_test = int(num_samples * test_ratio)
    print(f"num_samples_train: {num_samples_train}")
    print(f"num_samples_validation: {num_samples_val}")
    print(f"num_samples_test: {num_samples_test}")

    # Train + Validation
    num_samples_train_val = num_samples_train + num_samples_val
    lidar_data_train_val = lidar_data[:num_samples_train_val]

    lidar_train_data = lidar_data_train_val[:num_samples_train]
    random.shuffle(lidar_train_data)
    print(f"\n-----------TRAIN-----------")
    print(f"lidar_train_data: {len(lidar_train_data)}")
    for image in lidar_train_data:
        move(f"{out_dir}/{image}", f"{train_path}")

    lidar_val_data = lidar_data_train_val[num_samples_train:]
    random.shuffle(lidar_val_data)
    print(f"\n-----------VALIDATION-----------")
    print(f"lidar_val_data: {len(lidar_val_data)}")
    for image in lidar_val_data:
        move(f"{out_dir}/{image}", f"{validation_path}")

    lidar_test_data = lidar_data[num_samples_train_val:]
    random.shuffle(lidar_test_data)
    print(f"\n-----------TEST-----------")
    print(f"lidar_test_data: {len(lidar_test_data)}")
    for image in lidar_test_data:
        move(f"{out_dir}/{image}", f"{test_path}")


if __name__ == "__main__":
    create_folders()

    num_samples = len(listdir(out_dir))
    print(f"Number of samples: {num_samples}")

    create_dataset(num_samples)
    print(f"Data as added to {train_path}, {validation_path} and {test_path} folders!")

    num_samples_train = len(listdir(f"{train_path}"))
    print(f"\n\nNumber of samples in train folder: {num_samples_train}")
    num_samples_val = len(listdir(f"{validation_path}"))
    print(f"Number of samples in validation folder: {num_samples_val}")
    num_samples_test = len(listdir(f"{test_path}"))
    print(f"Number of samples in test folder: {num_samples_test}")