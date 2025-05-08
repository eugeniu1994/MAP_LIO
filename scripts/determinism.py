import numpy as np
import matplotlib.pyplot as plt

def compare_numpy_files(file1, file2, tolerance=1e-9):
    # Load the files into numpy arrays
    data1 = np.loadtxt(file1)
    data2 = np.loadtxt(file2)

    print('data1:', np.shape(data1), 'data2:', np.shape(data2))
    if data1.shape != data2.shape:
        print(f"Shape mismatch: {data1.shape} vs {data2.shape}")
        return

    # Compute the absolute difference
    diff = np.abs(data1 - data2)

    # Find where the difference is greater than the tolerance
    mismatch_indices = np.argwhere(diff > tolerance)

    if mismatch_indices.size == 0:
        print("Files match within tolerance.")
    else:
        print(f"{len(mismatch_indices)} differences found (tolerance = {tolerance}):")
        for idx in mismatch_indices:
            row, col = idx
            print(f"Line {row+1}, Column {col+1}: File1={data1[row, col]}, File2={data2[row, col]}, Δ={diff[row, col]}")


file1_path = "/home/eugeniu/x_vux-georeferenced-final/test_gnss-imu2_1/_debug_.txt"
file2_path = "/home/eugeniu/x_vux-georeferenced-final/test_gnss-imu2_2/_debug_.txt"


file1_path = "/home/eugeniu/x_vux-georeferenced-final/test_gnss-imu1_1/_debug_.txt"
file2_path = "/home/eugeniu/x_vux-georeferenced-final/test_gnss-imu1_2/_debug_.txt"


compare_numpy_files(file1_path, file2_path, tolerance=1e-5)