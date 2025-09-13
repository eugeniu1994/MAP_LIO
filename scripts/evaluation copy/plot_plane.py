import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# 1. Generate 3D points near a plane
# -----------------------------
np.random.seed(1)
n_points = 9

# True plane: 0.4x + 0.3y + z = 0  (normal = [0.4,0.3,1])
normal_true = np.array([0.4, 0.3, 1.0])
normal_true /= np.linalg.norm(normal_true)

# Generate random (x,y), solve for z, add noise
X = np.random.uniform(-1, 1, (n_points, 2))
Z = (-normal_true[0]*X[:,0] - normal_true[1]*X[:,1]) / normal_true[2]
Z += 0.05*np.random.randn(n_points)  # small deviation
points = np.column_stack((X, Z))

# -----------------------------
# 2. PCA on the points
# -----------------------------
mean = np.mean(points, axis=0)
centered = points - mean

cov = np.cov(centered.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# Sort in descending order
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

# -----------------------------
# 3. Plot points, PCA directions, normal, and plane
# -----------------------------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Plot points
ax.scatter(points[:,0], points[:,1], points[:,2], c='b', s=50, label="Points")

# Plot PCA directions from mean, scaled by eigenvalues
colors = ['r','g','b']
for i in range(3):
    vec = eigenvectors[:, i]
    ax.quiver(mean[0], mean[1], mean[2],
              vec[0], vec[1], vec[2],
              length=eigenvalues[i]*5,  # scale for visibility
              color=colors[i], linewidth=2, label=f"PC{i+1}")

# Normal vector = smallest eigenvector (normalized), start at first point
normal_vec = eigenvectors[:, -1] / np.linalg.norm(eigenvectors[:, -1])
p0 = points[0]
ax.quiver(p0[0], p0[1], p0[2],
          normal_vec[0], normal_vec[1], normal_vec[2],
          length=0.6, color='k', linewidth=2, linestyle='dashed', label="Plane normal")

# Plot fitted plane (through mean with normal = smallest eigenvector)
plane_size = 1.2
xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
zz = (-normal_vec[0]*(xx-mean[0]) - normal_vec[1]*(yy-mean[1]))/normal_vec[2] + mean[2]
ax.plot_surface(xx, yy, zz, alpha=0.3, color='cyan')

# -----------------------------
# 4. Make background white and remove axes
# -----------------------------
#ax.set_axis_off()
#ax.grid(False)
#ax.xaxis.pane.fill = False
#ax.yaxis.pane.fill = False
#ax.zaxis.pane.fill = False
#fig.patch.set_facecolor('white')
#ax.set_facecolor('white')

# Legend
ax.legend(loc="upper left")

plt.tight_layout()
plt.show()
