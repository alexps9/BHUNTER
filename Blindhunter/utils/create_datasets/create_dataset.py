import os
from os import mkdir, listdir
from shutil import move
import os
from PIL import Image

data_dir = "../data"
out_dir = "../_out"

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

sensor_folders = ["rgb", "depth", "lidar"]

train_path = f"{data_dir}/train"
validation_path = f"{data_dir}/validation"
test_path = f"{data_dir}/test"


def create_folders():
    # Create train, validation and test folders if they don't exist
    mkdir(train_path) if not os.path.exists(train_path) else None
    mkdir(validation_path) if not os.path.exists(validation_path) else None
    mkdir(test_path) if not os.path.exists(test_path) else None
    for sensor_folder in sensor_folders:
        mkdir(f"{validation_path}/{sensor_folder}") if not os.path.exists(
            f"{validation_path}/{sensor_folder}") else None
        mkdir(f"{train_path}/{sensor_folder}") if not os.path.exists(f"{train_path}/{sensor_folder}") else None
        mkdir(f"{test_path}/{sensor_folder}") if not os.path.exists(f"{test_path}/{sensor_folder}") else None


def create_dataset(num_samples):
    # Save the depth, rgb and lidar images
    imgs_depth = [f for f in sorted(listdir(f"{out_dir}/depth"))]
    imgs_rgb = [f for f in sorted(listdir(f"{out_dir}/rgb"))]
    imgs_lidar = [f for f in sorted(listdir(f"{out_dir}/lidar"))]

    # Number of samples for train, validation and test
    num_samples_train = int(num_samples * train_ratio)
    num_samples_val = int(num_samples * val_ratio)
    num_samples_test = int(num_samples * test_ratio)
    print(f"num_samples_train: {num_samples_train}")
    print(f"num_samples_validation: {num_samples_val}")
    print(f"num_samples_test: {num_samples_test}")

    # Train + Validation
    num_samples_train_val = num_samples_train + num_samples_val
    imgs_depth_train_val = imgs_depth[:num_samples_train_val]
    imgs_rgb_train_val = imgs_rgb[:num_samples_train_val]
    imgs_lidar_train_val = imgs_lidar[:num_samples_train_val]

    imgs_train_depth = imgs_depth_train_val[:num_samples_train]
    imgs_train_rgb = imgs_rgb_train_val[:num_samples_train]
    imgs_train_lidar = imgs_lidar_train_val[:num_samples_train]
    print(f"\n-----------TRAIN-----------")
    print(f"imgs_train_depth: {len(imgs_train_depth)}")
    print(f"imgs_train_rgb: {len(imgs_train_rgb)}")
    print(f"imgs_train_lidar: {len(imgs_train_lidar)}")

    imgs_val_depth = imgs_depth_train_val[num_samples_train:]
    imgs_val_rgb = imgs_rgb_train_val[num_samples_train:]
    imgs_val_lidar = imgs_lidar_train_val[num_samples_train:]
    print(f"\n-----------VALIDATION-----------")
    print(f"imgs_test_depth: {len(imgs_val_depth)}")
    print(f"imgs_test_rgb: {len(imgs_val_rgb)}")
    print(f"imgs_test_lidar: {len(imgs_val_lidar)}")

    imgs_test_depth = imgs_depth[num_samples_train_val:]
    imgs_test_rgb = imgs_rgb[num_samples_train_val:]
    imgs_test_lidar = imgs_lidar[num_samples_train_val:]
    print(f"\n-----------TEST-----------")
    print(f"imgs_test_depth: {len(imgs_test_depth)}")
    print(f"imgs_test_rgb: {len(imgs_test_rgb)}")
    print(f"imgs_test_lidar: {len(imgs_test_lidar)}")

    for image in imgs_train_depth:
        move(f"{out_dir}/depth/{image}", f"{train_path}/depth")
    for image in imgs_train_rgb:
        move(f"{out_dir}/rgb/{image}", f"{train_path}/rgb")
    for image in imgs_train_lidar:
        move(f"{out_dir}/lidar/{image}", f"{train_path}/lidar")

    for image in imgs_val_depth:
        move(f"{out_dir}/depth/{image}", f"{validation_path}/depth")
    for image in imgs_val_rgb:
        move(f"{out_dir}/rgb/{image}", f"{validation_path}/rgb")
    for image in imgs_val_lidar:
        move(f"{out_dir}/lidar/{image}", f"{validation_path}/lidar")

    for image in imgs_test_depth:
        move(f"{out_dir}/depth/{image}", f"{test_path}/depth")
    for image in imgs_test_rgb:
        move(f"{out_dir}/rgb/{image}", f"{test_path}/rgb")
    for image in imgs_test_lidar:
        move(f"{out_dir}/lidar/{image}", f"{test_path}/lidar")

    print(f"\nTrain: {len(imgs_train_depth)} depth, {len(imgs_train_rgb)} rgb, {len(imgs_train_lidar)} lidar")
    print(f"Validation: {len(imgs_val_depth)} depth, {len(imgs_val_rgb)} rgb, {len(imgs_val_lidar)} lidar")
    print(f"Test: {len(imgs_test_depth)} depth, {len(imgs_test_rgb)} rgb, {len(imgs_test_lidar)} lidar")


def set_same_num_samples():
    imgs_depth_after = [f for f in sorted(listdir(f"../_out/depth"))]
    imgs_rgb_after = [f for f in sorted(listdir(f"../_out/rgb"))]
    imgs_lidar_after = [f for f in sorted(listdir(f"../_out/lidar"))]

    num_samples = min(len(imgs_depth_after), len(imgs_rgb_after), len(imgs_lidar_after))

    # Remove the extra images
    for i in range(num_samples, len(imgs_depth_after)):
        os.remove(f"../_out/depth/{imgs_depth_after[i]}")
    for i in range(num_samples, len(imgs_rgb_after)):
        os.remove(f"../_out/rgb/{imgs_rgb_after[i]}")
    for i in range(num_samples, len(imgs_lidar_after)):
        os.remove(f"../_out/lidar/{imgs_lidar_after[i]}")

    imgs_depth = [f for f in sorted(listdir(f"../_out/depth"))]
    imgs_rgb = [f for f in sorted(listdir(f"../_out/rgb"))]
    imgs_lidar = [f for f in sorted(listdir(f"../_out/lidar"))]

    return len(imgs_depth), len(imgs_rgb), len(imgs_lidar)


def remove_corrupt_images():
    depth_dir = "../_out/depth"
    rgb_dir = "../_out/rgb"

    for file in os.listdir(depth_dir):
        file_path = os.path.join(depth_dir, file)
        try:
            img = Image.open(file_path)  # open the image file
            img.verify()  # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
            print('Bad file:', file_path)  # print out the names of corrupt files
            os.remove(file_path)

    for file in os.listdir(rgb_dir):
        file_path = os.path.join(rgb_dir, file)
        try:
            img = Image.open(file_path)  # open the image file
            img.verify()  # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
            print('Bad file:', file_path)  # print out the names of corrupt files
            os.remove(file_path)

    print("All corrupt images removed!")


if __name__ == "__main__":
    remove_corrupt_images()
    print("All cleaned up!")

    num_depth, num_rgb, num_lidar = set_same_num_samples()
    print(f"Depth: {num_depth} | RGB: {num_rgb} | Lidar: {num_lidar}")

    create_folders()

    create_dataset(num_samples=num_depth)
    print(f"Data as added to {train_path}, {validation_path} and {test_path} folders!")