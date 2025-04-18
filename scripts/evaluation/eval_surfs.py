import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Add as many methods as needed here
paths = {
    'GNSS-IMU0': '/home/eugeniu/vux-georeferenced/No_refinement/gnss-imu0/surface-eval',
    'Hesai0': '/home/eugeniu/vux-georeferenced/No_refinement/hesai0/surface-eval',

    'GNSS-IMU1-2BA': "/home/eugeniu/vux-georeferenced/BA-2_iterations/gnss-imu1/surface-eval",
    'GNSS-IMU1-3BA': "/home/eugeniu/vux-georeferenced/BA-3_iterations/gnss-imu1/surface-eval"
}


data = {}
for label, folder in paths.items():
    all_data = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith('.txt'):
            file_path = os.path.join(folder, fname)
            scan_data = np.loadtxt(file_path)
            # Filter out invalid rows (p2plane_error or num_neighbors == -1)
            valid_rows = (scan_data[:, 1] >= 0) & (scan_data[:, 5] >= 0)
            all_data.append(scan_data[valid_rows])
    data[label] = np.vstack(all_data)

# Metrics to evaluate
metrics = {
    'Point-to-surface error': 1,
    'Number of neighbours in a 1 m radius ball': 5
}

def compute_stats(arr):
    return {
        'Mean': np.mean(arr),
        'Median': np.median(arr),
        'RMSE': np.sqrt(np.mean(np.square(arr))),
        'Std Dev': np.std(arr),
    }

# Plotting stats
for metric_name, col_idx in metrics.items():
    plt.figure(figsize=(10, 6))

    stats_summary = {label: compute_stats(d[:, col_idx]) for label, d in data.items()}
    stat_keys = list(next(iter(stats_summary.values())).keys())
    x = np.arange(len(stat_keys))
    width = 0.8 / len(data)

    for i, (label, stats) in enumerate(stats_summary.items()):
        values = [stats[k] for k in stat_keys]
        plt.bar(x + i * width - width * len(data)/2, values, width, label=label)

    plt.ylabel(metric_name)
    plt.title(f'Statistics for {metric_name}')
    plt.xticks(x, stat_keys)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.draw()
    #plt.show()

# KDE plots for distribution
for metric_name, col_idx in metrics.items():
    plt.figure(figsize=(10, 5))
    for label in data:
        sns.kdeplot(data[label][:, col_idx], label=label, fill=True, bw_adjust=0.5)
    plt.title(f'Distribution of {metric_name}')
    plt.xlabel(metric_name)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.show()

plt.show()
