
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def suGRAD_1D():
    
    
    def f(x):
        return np.sin(3 * x) + 0.3 * x**2

    # --------------------------------------------------
    # Global gradient definition (paper)
    # --------------------------------------------------
    def F(x, y):
        return (f(y) - f(x)) / (y - x)


    # --------------------------------------------------
    # Parameters
    # --------------------------------------------------
    a, b = -2.0, 2.0         # search interval
    alpha = 0.05             # step size
    eta = 0.01 # 1e-3               # tolerance
    max_iter = 200
    delay = 0.4             # seconds between iterations


    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------
    x1 = a
    x2 = b


    # --------------------------------------------------
    # Prepare static function plot
    # --------------------------------------------------
    x_plot = np.linspace(a, b, 1000)
    y_plot = f(x_plot)

    plt.ion()  # interactive mode
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_plot, y_plot, linewidth=2)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Super Gradient Descent (SuGD) – Interactive 1D Optimization")

    # dynamic elements
    p1, = ax.plot([], [], 'ro', markersize=8)
    p2, = ax.plot([], [], 'bo', markersize=8)
    line, = ax.plot([], [], 'k--', linewidth=1)

    # --------------------------------------------------
    # Main SuGD loop
    # --------------------------------------------------
    for n in range(max_iter):

        G = F(x2, x1)

        # Update plot
        p1.set_data([x1], [f(x1)])
        p2.set_data([x2], [f(x2)])
        line.set_data([x1, x2], [f(x1), f(x2)])

        ax.set_title(
            f"Iteration {n} | "
            f"x1 = {x1:.3f}, x2 = {x2:.3f}, |x2-x1|(1+|F|) = {abs(x2-x1)*(1+abs(G)):.3e}"
        )

        plt.draw()
        plt.pause(delay)

        # stopping condition (paper)
        if abs(x2 - x1) * (1 + abs(G)) <= eta:
            break

        # SuGD update rule
        if f(x2) - f(x1) < 0:                       # f(x1) is bigger -> update x1
            x1 = x1 - alpha * (x1 - x2) * (1 - G)
        else:                                       #f(x2) is bigger  -> update x2 
            x2 = x2 - alpha * (x2 - x1) * (1 + G)


    plt.ioff()

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------
    x_star = 0.5 * (x1 + x2)
    print("Converged to x ≈", x_star)
    print("f(x) ≈", f(x_star))

    plt.show()

# suGRAD_1D()


# Set global parameters
plt.rcParams.update({
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes3d.grid': False,  # Remove grid
    'axes3d.xaxis.panecolor': (1.0, 1.0, 1.0, 1.0),  # White
    'axes3d.yaxis.panecolor': (1.0, 1.0, 1.0, 1.0),  # White
    'axes3d.zaxis.panecolor': (1.0, 1.0, 1.0, 1.0),  # White
})

# Define the 2D function to optimize
# Example: a simple 2D quadratic with minimum inside the square
def f(x):
    # x = [x, y]
    return (x[0]-2)**2 + (x[1]-1)**2 + np.sin(3*x[0]) * np.cos(2*x[1])

def f(xy):
    # x = [x, y]
    x, y = xy[0],xy[1]
    return (
            0.1 * (x**2 + y**2)
            + 0.8 * np.sin(3 * x) * np.sin(3 * y)
            + 0.3 * np.cos(5 * x)
            + 0.3 * np.cos(5 * y)
        )

def f(xy):
    # x = [x, y]
    x, y = xy[0], xy[1]
    # Rastrigin function (highly multimodal)
    return 20 + (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y))
    return (x-2)**2 + (y-1)**2 + np.sin(3*x) * np.cos(2*y)

def f_(x, y):
    # Rastrigin function (highly multimodal)
    return 20 + (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y))
    return (x-2)**2 + (y-1)**2 + np.sin(3*x) * np.cos(2*y)

# -------------------------------
# Initialize corners of the square search space
x_corners = np.array([
    [-4.,   4.],   # x1
    [4.0,   4.0],  # x2
    [4.0,  -4.0],  # x3
    [-4.0, -4.0]   # x4
], dtype=float)

x_corners = np.array([
    [-5.,   5.],   # x1
    [5.0,   5.0],  # x2
    [5.0,  -5.0],  # x3
    [-5.0, -5.0]   # x4
], dtype=float)

alpha = 0.005      # step size
iterations = 500  # number of iterations
tolerance = .2  # distance threshold for convergence

# Define the boundaries of the square
x_min_bound = -5.0
x_max_bound = 5.0
y_min_bound = -5.0
y_max_bound = 5.0


# x_min_bound = -10.0
# x_max_bound = 15.0
# y_min_bound = -15.0
# y_max_bound = 15.0

d = 2
# N = pow(2,d) # 4 
N = 4  # number of points

# x_corners = np.column_stack((
#     np.random.uniform(x_min_bound, x_max_bound, N),
#     np.random.uniform(y_min_bound, y_max_bound, N)
# ))






# Setup the dynamic plot
plt.ion()
# fig, ax = plt.subplots(figsize=(8, 8))
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax = fig.add_subplot(122)

# Plot the function as a contour for visualization
X = np.linspace(x_min_bound, x_max_bound, 100)
Y = np.linspace(y_min_bound, y_max_bound, 100)
X_mesh, Y_mesh = np.meshgrid(X, Y)
Z = f([X_mesh, Y_mesh])

x_plot = np.linspace(x_min_bound, x_max_bound, 100)
y_plot = np.linspace(y_min_bound, y_max_bound, 100)
X, Y = np.meshgrid(x_plot, y_plot)
Z = f_(X, Y)
# Z = f([X_mesh, Y_mesh])




contour = ax.contourf(X_mesh, Y_mesh, Z, levels=50, cmap=cm.coolwarm,) # cmap='viridis'
fig.colorbar(contour, label='Cost')


# 3D surface plot
surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, 
                          alpha=0.4, linewidth=0, antialiased=True)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')
ax1.set_title('Optimization - 3D Surface View')
ax1.set_facecolor('white')
    

points_3d = [
    ax1.plot([], [], 'ro', markersize=10)[0],
    ax1.plot([], [], 'bo', markersize=10)[0],
    ax1.plot([], [], color='orange', marker='o', markersize=10)[0],
    ax1.plot([], [], color='magenta', marker='o', markersize=10)[0]
]

lines_3d = [
    ax1.plot([], [], 'k--', linewidth=2)[0],
    ax1.plot([], [], 'k--', linewidth=2)[0],
    ax1.plot([], [], 'k--', linewidth=2)[0]
]


for i, pt in enumerate(points_3d):
    pt.set_data([x_corners[i,0]], [x_corners[i,1]])
    pt.set_3d_properties([f(x_corners[i])])


# Dynamic points
points = [
    ax.plot([], [], 'ro', markersize=10)[0],
    ax.plot([], [], 'bo', markersize=10)[0],
    ax.plot([], [], color='orange', marker='o', markersize=10)[0],
    ax.plot([], [], color='magenta', marker='o', markersize=10)[0]
]

lines = [
    ax.plot([], [], 'k--', linewidth=1)[0],
    ax.plot([], [], 'k--', linewidth=1)[0],
    ax.plot([], [], 'k--', linewidth=1)[0]
]

# History paths for each point
history = [ [x_corners[i].copy()] for i in range(N) ]  # list of lists
path_lines = [
    ax.plot([], [], '-', color='r', linewidth=1)[0],
    ax.plot([], [], '-', color='b', linewidth=1)[0],
    ax.plot([], [], '-', color='orange', linewidth=1)[0],
    ax.plot([], [], '-', color='magenta', linewidth=1)[0]
]


ax.set_xlim(x_min_bound, x_max_bound)
ax.set_ylim(y_min_bound, y_max_bound)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("2D view")

for i, pt in enumerate(points):
    pt.set_data(x_corners[i,0], x_corners[i,1])



# -------------------------------
# Optimization loop
for it in range(iterations):
    input("\nPress Enter to perform the next iteration...")  # wait for Enter

    # Compute function values
    g = np.array([f(xi) for xi in x_corners])
    
    # Find the index of the minimum cost
    min_idx = np.argmin(g)
    x_min = x_corners[min_idx]
    g_min = g[min_idx]
    
    # Update all points except the minimum
    line_idx = 0
    for i in range(N):
        if i != min_idx:

            # Global gradient toward minimum
            F = (g[i] - g_min) / np.linalg.norm(x_corners[i] - x_min)  # note g[i] - g_min
            # Update toward minimum
            x_corners[i] = x_corners[i] - alpha * (x_corners[i] - x_min) * (1 + F)

            # Clip to stay inside the square
            x_corners[i,0] = np.clip(x_corners[i,0], x_min_bound, x_max_bound)
            x_corners[i,1] = np.clip(x_corners[i,1], y_min_bound, y_max_bound)

            # Update the line connecting this point to the minimum
            lines[line_idx].set_data([x_corners[i,0], x_min[0]],
                                     [x_corners[i,1], x_min[1]])
            
            lines_3d[line_idx].set_data([x_corners[i,0], x_min[0]], [x_corners[i,1], x_min[1]])
            lines_3d[line_idx].set_3d_properties([f(x_corners[i]), f(x_min)])


            line_idx += 1

    # Update dynamic points and history paths
    for i, pt in enumerate(points):
        pt.set_data(x_corners[i,0], x_corners[i,1])
        history[i].append(x_corners[i].copy())
        path = np.array(history[i])
        path_lines[i].set_data(path[:,0], path[:,1])
    
    for i, pt in enumerate(points_3d):
        pt.set_data([x_corners[i,0]], [x_corners[i,1]])
        pt.set_3d_properties([f(x_corners[i])])


    plt.pause(.1)  # pause for animation
    # Convergence check
    distances = np.linalg.norm(x_corners - x_min, axis=1)
    distances = np.delete(distances, min_idx)  # ignore minimum itself

    print("distances:",distances)
    if np.all(distances < tolerance):
        print(f"Converged at iteration {it+1}")
        plt.ioff()
        plt.show()
        break

plt.ioff()
plt.show()
