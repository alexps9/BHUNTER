import shutil
import os

target_base_folder = "../../merged_dataset"
datasets_path = "../../../../../DataSets_200x200x16"

# What type of weather to merge
main_folders = ['DayClear', 'DayRain', 'DayCloudy', 'NightCloudy']

# Define subfolders and categories
subfolders = ['train', 'test', 'validation']
categories = ['rgb', 'depth', 'lidar', 'ground_truth']


def create_target():
    # Create the target base folder if it doesn't exist
    if not os.path.exists(target_base_folder):
        print("Creating the target base folder...")
        os.makedirs(target_base_folder)

    # Create the subfolder structure within the target base folder
    for subfolder in subfolders:
        for category in categories:
            print(f"Creating {subfolder}/{category} folder...")
            os.makedirs(os.path.join(target_base_folder, subfolder, category), exist_ok=True)


# Function to copy files from source to destination
def copy_files(source, destination):
    print(f"Copying files from {source} to {destination}...")

    for file_name in os.listdir(source):
        full_file_name = os.path.join(source, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, destination)


if __name__ == "__main__":
    create_target()

    # Iterate over each main folder and copy its contents to the target base folder
    for main_folder in main_folders:
        for subfolder in subfolders:
            for category in categories:
                source_folder = os.path.join(datasets_path, main_folder, subfolder, category)
                target_folder = os.path.join(target_base_folder, subfolder, category)
                if os.path.exists(source_folder):
                    copy_files(source_folder, target_folder)
