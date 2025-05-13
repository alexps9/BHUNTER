import os


def count_imgs_data():
    train_path = "../data/train"
    validation_path = "../data/validation"
    test_path = "../data/test"

    validation_lidar = len(os.listdir(validation_path + "/lidar"))
    validation_rgb = len(os.listdir(validation_path + "/rgb"))
    validation_depth = len(os.listdir(validation_path + "/depth"))

    train_lidar = len(os.listdir(train_path + "/lidar"))
    train_rgb = len(os.listdir(train_path + "/rgb"))
    train_depth = len(os.listdir(train_path + "/depth"))

    test_lidar = len(os.listdir(test_path + "/lidar"))
    test_rgb = len(os.listdir(test_path + "/rgb"))
    test_depth = len(os.listdir(test_path + "/depth"))

    print(f"Train Lidar: {train_lidar}")
    print(f"Train RGB: {train_rgb}")
    print(f"Train Depth: {train_depth}")
    print(f"------------")
    print(f"Validation Lidar: {validation_lidar}")
    print(f"Validation RGB: {validation_rgb}")
    print(f"Validation Depth: {validation_depth}")
    print(f"------------")
    print(f"Test Lidar: {test_lidar}")
    print(f"Test RGB: {test_rgb}")
    print(f"Test Depth: {test_depth}")
    print(f"------------")
    print(f"TOTAL PER SENSOR: {test_lidar + train_lidar + validation_lidar}")


if __name__ == "__main__":
    count_imgs_data()
