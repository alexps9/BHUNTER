import open3d as o3d
import glob
import argparse
import os
from open3d import visualization


def read_point_clouds_from_path(path_pattern):
    files = glob.glob(path_pattern)
    if not files:
        print(f"[Warning] No files found at: {path_pattern}")
        return []
    return [o3d.io.read_point_cloud(f) for f in files]


def visualize_point_clouds(clouds, title="Point Clouds"):
    if not clouds:
        print(f"[Info] No point clouds to visualize for {title}")
        return
    print(f"[Info] Visualizing {len(clouds)} point clouds for {title}")
    visualization.draw_geometries(clouds)


def main():
    parser = argparse.ArgumentParser(description="Visualize point clouds from specified paths.")
    parser.add_argument('--lidar', '-L', help='Path to LIDAR point cloud directory or file pattern (e.g., path/*.ply)')
    parser.add_argument('--segmentation', '-S', help='Path to segmentation point cloud directory or file pattern')
    parser.add_argument('--ground_truth', '-G', help='Path to ground truth point cloud directory or file pattern')
    args = parser.parse_args()

    if args.ground_truth:
        gt_clouds = read_point_clouds_from_path(args.ground_truth)
        visualize_point_clouds(gt_clouds, title="Ground Truth")

    if args.lidar:
        lidar_clouds = read_point_clouds_from_path(args.lidar)
        visualize_point_clouds(lidar_clouds, title="LIDAR")

    if args.segmentation:
        segm_clouds = read_point_clouds_from_path(args.segmentation)
        visualize_point_clouds(segm_clouds, title="Segmentation")


if __name__ == "__main__":
    main()
