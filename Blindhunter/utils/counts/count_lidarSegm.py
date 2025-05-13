import os

path_lidarSegm = '../../_out/lidarSegm/'

def count_files():
    print(f"Number of images in /lidarSegm: {len([name for name in os.listdir(path_lidarSegm) if os.path.isfile(os.path.join(path_lidarSegm, name))])}")


if __name__ == '__main__':
    count_files()