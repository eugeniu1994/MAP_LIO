
#pip install open3d


import open3d as o3d
import os
from pathlib import Path
import argparse
import re
import numpy as np

def natural_sort_key(path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', path.stem)]

#this will save all the data - not only x y z 
#pip install pypcd
# from pypcd import pypcd
# def read_and_merge_pcds_full(pcd_paths):
#     merged_data = []

#     for path in pcd_paths:
#         pc = pypcd.PointCloud.from_path(str(path))
#         merged_data.append(pc.pc_data)

#     all_data = np.concatenate(merged_data, axis=0)
#     #merged_pc = pypcd.make_xyz_point_cloud(all_data)  # make_xyzrgb_point_cloud for color

#     return pypcd.PointCloud(pc_data=all_data, fields=pc.fields, size=pc.size,
#                             count=pc.count, width=len(all_data), height=1,
#                             viewpoint=pc.viewpoint, dtype=pc.dtype, version=pc.version)

def apply_transformation(pcd, T):
    T = np.array(T, dtype=np.float64)
    
    points = np.asarray(pcd.points, dtype=np.float64)
    transformed_points = points.dot(T[:3, :3].T) + T[:3, 3]

    pcd.points = o3d.utility.Vector3dVector(transformed_points)

    return pcd

def read_and_merge_pcds(pcd_paths):
    merged_pcd = o3d.geometry.PointCloud()
    for path in pcd_paths:
        #print('str(path):',str(path))
        pcd = o3d.io.read_point_cloud(str(path))
        merged_pcd += pcd

    print(f"\nmerged_pcd: {len(merged_pcd.points)} points")

    return merged_pcd

def read_se3_and_inverse_from_txt(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    matrices = {}
    current_key = None
    current_matrix = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith("T_"):
            if current_key and current_matrix:
                matrices[current_key] = np.array(current_matrix, dtype=float)
                current_matrix = []
            current_key = line
        else:
            current_matrix.append([float(x) for x in line.split()])
    
    if current_key and current_matrix:
        matrices[current_key] = np.array(current_matrix, dtype=float)

    return matrices["T_als2mls"], matrices["T_mls2als"]

def main(input_dir, output_dir, local_global_T, group_size = 100, visualize=False):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    #all_pcds = sorted(input_path.glob("*.pcd"))
    all_pcds = sorted(input_path.glob("*.pcd"), key=natural_sort_key)

    total_files = len(all_pcds)
    print(f"Found {total_files} .pcd files.")

    T_als2mls, T_mls2als = read_se3_and_inverse_from_txt(local_global_T)

    print("ALS to MLS:\n", T_als2mls)
    print("MLS to ALS:\n", T_mls2als)
    
    transform_mls_to_als_frame = False # True

    for i in range(0, total_files, group_size):
        group_files = all_pcds[i:i + group_size]
        if not group_files:
            continue

        print(f"Merging files {i+1} to {i+len(group_files)}...")
        merged_pcd = read_and_merge_pcds(group_files)  # now they are in the mls frame

        if transform_mls_to_als_frame:
            if T_mls2als is not None:
                merged_pcd = apply_transformation(merged_pcd, T_mls2als)
                print("Transformation from MLS to ALS applied to the merged point cloud.")

        output_file = output_path / f"merged_{i//group_size:03d}.pcd"
        o3d.io.write_point_cloud(str(output_file), merged_pcd)
        print(f"Saved: {output_file}")

        if visualize:
            print("Visualizing...")
            try:
                o3d.visualization.draw_geometries([merged_pcd])
            except KeyboardInterrupt:
                print("\nVisualization interrupted. Closing window...")


if __name__ == "__main__":
   

    parser = argparse.ArgumentParser(description="Merge .pcd files into chunks of 100.")
    parser.add_argument("--input_dir", default="/home/eugeniu/vux-georeferenced/hesai1/", help="Folder containing .pcd files")
    parser.add_argument("--output_dir", default="/home/eugeniu/vux-georeferenced/merged/", help="Folder to save merged .pcd files")
    parser.add_argument("--local_global_T", default="/home/eugeniu/vux-georeferenced/als2mls_dense.txt", help="File with mls to als transform")
    parser.add_argument("--visualize", action="store_false", help="Visualize merged point clouds")
    parser.set_defaults(visualize=False)
    args = parser.parse_args()


    main(args.input_dir, args.output_dir, args.local_global_T, 300, args.visualize)
