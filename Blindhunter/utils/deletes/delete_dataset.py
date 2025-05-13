import os


def delete_dataset():
    data_paths = {
        "train": "../data/train",
        "validation": "../data/validation",
        "test": "../data/test",
    }

    for dataset_type, path in data_paths.items():
        for subdir in ["lidar", "rgb", "depth"]:
            subdir_path = os.path.join(path, subdir)
            try:
                os.removedirs(subdir_path)
            except OSError:
                for f in os.listdir(subdir_path):
                    os.remove(os.path.join(subdir_path, f))


if __name__ == "__main__":
    delete_dataset()
    print("All cleaned up!")
