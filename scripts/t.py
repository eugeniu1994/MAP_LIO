import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D


import matplotlib.pyplot as plt
import numpy as np

# Datasets
datasets = ["KITTI-360", "Fog", "Rain", "Snow"]

# Relative error (translation [%]) values
rel_trans = {
    "KISS-ICP": [1.46, 1.56, 1.75, 1.59],
    "FPR":      [1.39, 1.44, 1.41, 1.42]
}

# Plot relative translation error with zoomed y-axis for better emphasis
x = np.arange(len(datasets))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))
rects1 = ax.bar(x - width/2, rel_trans["KISS-ICP"], width, label="KISS-ICP", color="tab:blue")
rects2 = ax.bar(x + width/2, rel_trans["FPR"], width, label="FPR", color="tab:green")

ax.set_ylabel("Error (%)")
ax.set_title("Relative Translation Error")
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()

# Zoom y-axis to highlight differences
ax.set_ylim(1.3, 1.9)

# Add grid
ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.7)
ax.set_axisbelow(True)

# Annotate values on top of bars
for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # offset text
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.show()



x = -1.34
y = 2.98
z = -21.56

magnitude = math.sqrt(x**2 + y**2 + z**2)
print(magnitude)  # Output: ≈ 21.78



plt.style.use('default')

# --- Generate smooth base trajectory ---
num_points = 20  # Fewer control points for smoothness
t = np.linspace(0, 15, num_points)

# Control points for smooth s-curve with elevation
x = np.linspace(0, 80, num_points)
y = 8 * np.sin(x/25)  # Base s-curve
z = 0.1 * x + 1 * np.sin(x/30)  # Elevation profile

# Create cubic spline interpolators
cs_x = CubicSpline(t, x)
cs_y = CubicSpline(t, y)
cs_z = CubicSpline(t, z)

# Sample at higher resolution for smoothness
t_dense = np.linspace(0, 20, 20)
x_smooth = cs_x(t_dense)
y_smooth = cs_y(t_dense)
z_smooth = cs_z(t_dense)

# --- Sensor transformation ---
# Rotation (pitch 5°, yaw 20°, roll 3°)
Rx = np.array([[1, 0, 0],
               [0, np.cos(np.radians(0)), -np.sin(np.radians(0))],
               [0, np.sin(np.radians(0)), np.cos(np.radians(0))]])

Ry = np.array([[np.cos(np.radians(.01)), 0, np.sin(np.radians(.01))],
               [0, 1, 0],
               [-np.sin(np.radians(.01)), 0, np.cos(np.radians(.01))]])

Rz = np.array([[np.cos(np.radians(.01)), -np.sin(np.radians(.01)), 0],
               [np.sin(np.radians(.01)), np.cos(np.radians(.01)), 0],
               [0, 0, 1]])

extrinsic_rotation = Rz @ Ry @ Rx
extrinsic_translation = np.array([.2, -0.7, 0.6])  # Sensor offset

# Apply transformation
traj1 = np.vstack((x_smooth, y_smooth, z_smooth)).T
traj2 = (extrinsic_rotation @ traj1.T).T + extrinsic_translation

# --- Visualization ---
fig = plt.figure(figsize=(10, 8), facecolor='none')
ax = fig.add_subplot(111, projection='3d')

ax.plot(traj2[:,0], traj2[:,1], traj2[:,2], 
        c='#ff7f0e', marker='s', markersize=14,
        label='Lidar (Sensor 2)', 
        linewidth=1.5, alpha=0.9)
# Plot smooth trajectories
ax.plot(traj1[:,0], traj1[:,1], traj1[:,2], 
        c='#1f77b4', marker='o', markersize=14, 
        label='GNSS/IMU (Sensor 1)', 
        linewidth=1.5, alpha=0.9)



# Style adjustments
ax.set_xlabel('East (m)', fontsize=10)
ax.set_ylabel('North (m)', fontsize=10)
ax.set_zlabel('Up (m)', fontsize=10)
ax.legend()

# Equal aspect ratio
ax.set_box_aspect([1, 1, 1])

# Transparent background
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    axis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.grid(True)

# Set viewing angle for best visualization
ax.view_init(elev=25, azim=-45)

#plt.tight_layout()
#plt.savefig('smooth_trajectories.png', transparent=True, dpi=300)
plt.show()