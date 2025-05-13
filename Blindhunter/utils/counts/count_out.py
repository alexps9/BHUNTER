import os

path_rgb = '../../_out/rgb/'
path_depth = '../../_out/depth/'
path_lidar = '../../_out/lidar/'
path_ground_truth = '../../_out/ground_truth/'

def count_imgs():
    print(f"Number of images in {path_rgb}: {len([name for name in os.listdir(path_rgb) if os.path.isfile(os.path.join(path_rgb, name))])}")
    print(f"Number of images in {path_depth}: {len([name for name in os.listdir(path_depth) if os.path.isfile(os.path.join(path_depth, name))])}")
    print(f"Number of images in {path_lidar}: {len([name for name in os.listdir(path_lidar) if os.path.isfile(os.path.join(path_lidar, name))])}")
    print(f"Number of images in {path_ground_truth}: {len([name for name in os.listdir(path_ground_truth) if os.path.isfile(os.path.join(path_ground_truth, name))])}")

if __name__ == '__main__':
    count_imgs()