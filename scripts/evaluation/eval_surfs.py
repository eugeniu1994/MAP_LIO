import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

sns.set(style="whitegrid")

# p2plane error, furtherst_d, closest_d, curvature, neighbours in a radius ball 
methods = {
    'GNSS-IMU0': '/home/eugeniu/vux-georeferenced/No_refinement/gnss-imu0/surface-eval',
    #'Hesai0': '/home/eugeniu/vux-georeferenced/No_refinement/hesai0/surface-eval',

    #'GNSS-IMU1-2BA': "/home/eugeniu/vux-georeferenced/BA-2_iterations/gnss-imu1/surface-eval",
    #'GNSS-IMU1-3BA': "/home/eugeniu/vux-georeferenced/BA-3_iterations/gnss-imu1/surface-eval"
}

data = {}
for label, folder in methods.items():
    all_data = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith('.txt'):
            file_path = os.path.join(folder, fname)
            scan_data = np.loadtxt(file_path)
            valid_rows = (scan_data[:, 0] >= 0) # Filter out invalid rows (p2plane_error == -1)
            all_data.append(scan_data[valid_rows])
    data[label] = np.vstack(all_data)

# Metrics to evaluate
metrics = {
    'Point-to-surface error': 0,
    'Number of neighbours in a 1 m radius ball': 4
}

def compute_stats(arr):
    return {
        'Mean': np.mean(arr),
        'Median': np.median(arr),
        'RMSE': np.sqrt(np.mean(np.square(arr))),
        'Std Dev': np.std(arr),
    }

def show_stats():
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
        plt.draw()

def show_correlation():
    plt.figure(figsize=(10, 6))
    #Pearson correlation coefficient matrix between x and y. 1 (perfect positive correlation), 0.1 – 0.3	Weak positive correlation
    #WE WANT TO SHOW THAT THERE IS NO CORRELATION - BETWEEN THE CURVATURE AND THE POINT TO PLANE ERROR 
    for label, d in data.items():
        x = d[:, 3]  # Curvature
        y = d[:, 0]  # Point-to-plane error

        plt.scatter(x, y, s=1, alpha=0.5, label=f'{label} (r={np.corrcoef(x, y)[0,1]:.2f})')

        # Plot horizontal line: mean error
        mean_error = np.mean(y)
        plt.axhline(mean_error, color='red', linestyle='--',  linewidth=5, alpha=1)
        plt.text(np.max(x)*0.95, mean_error, f"mean={mean_error:.3f}", va='bottom', ha='right', fontsize=14, color='black')

        # Optional: Fit a line (linear regression)
        # slope, intercept, r_value, _, _ = linregress(x, y)
        # x_fit = np.linspace(np.min(x), np.max(x), 500)
        # y_fit = slope * x_fit + intercept
        # plt.plot(x_fit, y_fit, linestyle='-', c = 'g', linewidth=5, label=f'{label} fit (slope={slope:.2e})')

        plt.title('Point-to-Plane Error vs Curvature')
        plt.xlabel('Curvature')
        plt.ylabel('Point-to-Plane Error')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.draw()

show_stats()

plt.show()

show_correlation()

plt.show()