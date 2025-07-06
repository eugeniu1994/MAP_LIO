import os
import numpy as np
import open3d as o3d

def get_first_points_from_folder(folder_path, max_files=100):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pcd')])[:max_files]
    #files = [f for f in os.listdir(folder_path) if f.endswith('.pcd')][:max_files]

    first_points = []

    for file in files:
        print(', ',file)
        pcd = o3d.io.read_point_cloud(os.path.join(folder_path, file))
        points = np.asarray(pcd.points)
        if points.shape[0] > 0:
            first_points.append(points[0])
        else:
            print(f"Warning: {file} has no points.")
            first_points.append(np.array([np.nan, np.nan, np.nan]))

    print('\nfirst_points:', np.shape(first_points))

    return np.array(first_points)

def compare_distances(folder1_points, folder2_points):
    distances = np.linalg.norm(folder1_points - folder2_points, axis=1)
    for i, d in enumerate(distances):
        print(f"File {i}: Distance = {d:.6f}")
    print("\nStats:")
    print(f"  Max distance: {np.nanmax(distances):.6f}")
    print(f"  Mean distance: {np.nanmean(distances):.6f}")
    print(f"  Std dev: {np.nanstd(distances):.6f}")
    return distances


folder1 = "/home/eugeniu/vux-georeferenced/No_refinement/gnss-imu0"
folder2 = "/home/eugeniu/vux_save_clouds"

points1 = get_first_points_from_folder(folder1, 10)
points2 = get_first_points_from_folder(folder2, 10)

# Compare distances
distances = compare_distances(points1, points2)
