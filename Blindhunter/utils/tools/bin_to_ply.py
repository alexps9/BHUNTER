import numpy as np
from plyfile import PlyData, PlyElement
import argparse
import glob
import os

def bin_to_ply(bin_path, ply_path):
    try:
        point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    except Exception as e:
        print(f"[Error] Reading .bin file {bin_path}: {e}")
        return

    try:
        vertices = np.array(
            [tuple(point) for point in point_cloud],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4')]
        )
        ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=True)
        os.makedirs(os.path.dirname(ply_path), exist_ok=True)
        ply.write(ply_path)
        print(f"[Success] Converted: {bin_path} -> {ply_path}")
    except Exception as e:
        print(f"[Error] Writing .ply file {ply_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert .bin point cloud files to .ply format.")
    parser.add_argument('--input', '-i', required=True, help='Path pattern to .bin files (e.g., path/*.bin)')
    parser.add_argument('--output_dir', '-o', required=True, help='Directory to save .ply files')

    args = parser.parse_args()
    bin_files = glob.glob(args.input)

    if not bin_files:
        print(f"[Error] No .bin files found at: {args.input}")
        return

    for bin_file in bin_files:
        filename = os.path.splitext(os.path.basename(bin_file))[0] + ".ply"
        ply_path = os.path.join(args.output_dir, filename)
        bin_to_ply(bin_file, ply_path)

if __name__ == "__main__":
    main()
