
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
    # return (x-2)**2 + (y-1)**2 + np.sin(3*x) * np.cos(2*y)

def f_(x, y):
    # Rastrigin function (highly multimodal)
    return 20 + (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y))
    # return (x-2)**2 + (y-1)**2 + np.sin(3*x) * np.cos(2*y)


def numerical_gradient(f, x, eps=1e-6):
    grad = np.zeros_like(x, dtype=float)
    f0 = f(x)

    for i in range(len(x)):
        x_eps = x.copy()
        x_eps[i] += eps
        grad[i] = (f(x_eps) - f0) / eps

    return grad

def gradient_descent_step(x):
    grad = numerical_gradient(f, x)
    return grad

def residual(x):
    return np.array([f(x)])

def numerical_jacobian_residual(x, eps=1e-6):
    r0 = residual(x)
    J = np.zeros((len(r0), len(x)))

    for i in range(len(x)):
        x_eps = x.copy()
        x_eps[i] += eps
        J[:, i] = (residual(x_eps) - r0) / eps

    return J

def gauss_newton_step(x, lam=1e-6):
    r = residual(x)
    J = numerical_jacobian_residual(x)

    H = J.T @ J + lam * np.eye(len(x))
    g = J.T @ r

    dx = -np.linalg.solve(H, g)
    return dx


def local_step(x, method="gd"):
    if method == "gd":
        return -gradient_descent_step(x)

    elif method == "gn":
        return gauss_newton_step(x)

    else:
        raise ValueError(f"Unknown method: {method}")




def test_prev():

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
                #One step on their own            
                # dx = -0.1*alpha * local_step(x_corners[i], method="gd") # "gd" or "gn"
                dx = alpha * local_step(x_corners[i], method="gn") # "gd" or "gn"
                x_corners[i] += dx


                alpha = 0.01

                # Global gradient toward minimum - directional_derivative
                eps = 1e-9 
                F = (g[i] - g_min) / (np.linalg.norm(x_corners[i] - x_min) + eps)
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

        print("distances:",distances, ', g_min:',g_min)
        if np.all(distances < tolerance):
            print(f"Converged at iteration {it+1}")
            plt.ioff()
            plt.show()
            break

    plt.ioff()
    plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class OptimizerVisualizer:
    def __init__(
        self,
        f,
        f_surface,
        local_step,
        N=4,
        bounds=(-5.0, 5.0, -5.0, 5.0),
        init_mode="corners",   # "corners" or "uniform"
        alpha=0.005,
        iterations=500,
        tolerance=0.2,
        pause_time=0.001,
        wait_for_enter=True,
        seed=None
    ):
        """
        Parameters
        ----------
        f : callable
            Cost function f(x) where x is shape (2,)
        f_surface : callable
            Cost function f(X, Y) for surface plotting
        local_step : callable
            Local optimization step (GD / GN)
        N : int
            Number of points
        bounds : tuple
            (x_min, x_max, y_min, y_max)
        init_mode : str
            "corners" or "uniform"
        """
        self.f = f
        self.f_surface = f_surface
        self.local_step = local_step
        self.N = N
        self.alpha = alpha
        self.iterations = iterations
        self.tolerance = tolerance
        self.pause_time = pause_time
        self.wait_for_enter = wait_for_enter

        self.x_min, self.x_max, self.y_min, self.y_max = bounds

        if seed is not None:
            np.random.seed(seed)

        self._init_points(init_mode)
        self._init_plot()


    def _init_points(self, init_mode):
        if init_mode == "corners":
            if self.N != 4:
                raise ValueError("Corner initialization requires N=4")
            self.x = np.array([
                [self.x_min, self.y_max],
                [self.x_max, self.y_max],
                [self.x_max, self.y_min],
                [self.x_min, self.y_min]
            ], dtype=float)

        elif init_mode == "uniform":
            self.x = np.column_stack((
                np.random.uniform(self.x_min, self.x_max, self.N),
                np.random.uniform(self.y_min, self.y_max, self.N)
            ))

        elif init_mode == "uniform_bounds":
            self.x = np.zeros((self.N, 2))
            sides = np.random.randint(0, 4, self.N)

            for i, s in enumerate(sides):
                if s == 0:      # left
                    self.x[i] = [self.x_min, np.random.uniform(self.y_min, self.y_max)]
                elif s == 1:    # right
                    self.x[i] = [self.x_max, np.random.uniform(self.y_min, self.y_max)]
                elif s == 2:    # bottom
                    self.x[i] = [np.random.uniform(self.x_min, self.x_max), self.y_min]
                else:           # top
                    self.x[i] = [np.random.uniform(self.x_min, self.x_max), self.y_max]

        else:
            raise ValueError(
                "init_mode must be 'corners', 'uniform', or 'uniform_bounds'"
            )

        self.history = [[self.x[i].copy()] for i in range(self.N)]

    def _init_plot(self):
        plt.ion()
        self.fig = plt.figure(figsize=(14, 6))
        self.ax3d = self.fig.add_subplot(121, projection="3d")
        self.ax2d = self.fig.add_subplot(122)

        # Surface / contour
        grid = np.linspace(self.x_min, self.x_max, 100)
        X, Y = np.meshgrid(grid, grid)
        Z = self.f_surface(X, Y)

        contour = self.ax2d.contourf(X, Y, Z, levels=50, cmap=cm.coolwarm)
        self.fig.colorbar(contour, ax=self.ax2d, label="Cost")

        self.ax3d.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.4)

        self.ax3d.set_title("3D Surface View")
        self.ax3d.set_xlabel("x")
        self.ax3d.set_ylabel("y")
        self.ax3d.set_zlabel("f(x,y)")

        self.ax2d.set_title("2D Optimization View")
        self.ax2d.set_xlim(self.x_min, self.x_max)
        self.ax2d.set_ylim(self.y_min, self.y_max)
        self.ax2d.set_xlabel("x")
        self.ax2d.set_ylabel("y")

        colors = ["r", "b", "orange", "magenta", "g", "c", "y"]

        self.points_2d = []
        self.points_3d = []
        self.paths_2d = []

        for i in range(self.N):
            color = colors[i % len(colors)]
            self.points_2d.append(
                self.ax2d.plot([], [], marker="o", color=color, markersize=8)[0]
            )
            self.points_3d.append(
                self.ax3d.plot([], [], [], marker="o", color=color, markersize=8)[0]
            )
            self.paths_2d.append(
                self.ax2d.plot([], [], "-", color=color, linewidth=1)[0]
            )

        self.lines_2d = [self.ax2d.plot([], [], "k--", lw=1)[0] for _ in range(self.N-1)]
        self.lines_3d = [self.ax3d.plot([], [], [], "k--", lw=2)[0] for _ in range(self.N-1)]

    def step(self):
        g = np.array([self.f(xi) for xi in self.x])
        min_idx = np.argmin(g)
        x_min = self.x[min_idx]
        self.g_min = g[min_idx]

        line_idx = 0
        for i in range(self.N):
            if i == min_idx:
                continue

            # Local step
            # dx = .5 * self.local_step(self.x[i], method="gn")
            # self.x[i] += dx

            eps = 1e-9
            F = (g[i] - self.g_min) / (np.linalg.norm(self.x[i] - x_min) + eps)
            self.x[i] -= self.alpha * (self.x[i] - x_min) * (1 + F)

            self.x[i, 0] = np.clip(self.x[i, 0], self.x_min, self.x_max)
            self.x[i, 1] = np.clip(self.x[i, 1], self.y_min, self.y_max)

            self.lines_2d[line_idx].set_data(
                [self.x[i, 0], x_min[0]],
                [self.x[i, 1], x_min[1]]
            )
            self.lines_3d[line_idx].set_data(
                [self.x[i, 0], x_min[0]],
                [self.x[i, 1], x_min[1]]
            )
            self.lines_3d[line_idx].set_3d_properties(
                [self.f(self.x[i]), self.f(x_min)]
            )
            line_idx += 1

        return min_idx

    def _update_plots(self):
        for i in range(self.N):
            self.points_2d[i].set_data(self.x[i, 0], self.x[i, 1])
            self.points_3d[i].set_data([self.x[i, 0]], [self.x[i, 1]])
            self.points_3d[i].set_3d_properties([self.f(self.x[i])])

            self.history[i].append(self.x[i].copy())
            path = np.array(self.history[i])
            self.paths_2d[i].set_data(path[:, 0], path[:, 1])

        plt.pause(self.pause_time)


    def run(self):
        self._update_plots()
        for it in range(self.iterations):
            if self.wait_for_enter:
                input("\nPress Enter to perform the next iteration...")

            min_idx = self.step()
            self._update_plots()

            distances = np.linalg.norm(self.x - self.x[min_idx], axis=1)
            distances = np.delete(distances, min_idx)

            print(f"Iteration {it+1}, distances: {distances}")

            if np.all(distances < self.tolerance):
                print(f"Converged at iteration {it+1}, with g:{self.g_min}")
                break

        plt.ioff()
        plt.show()


optimizer = OptimizerVisualizer(
    f=f,
    f_surface=f_,
    local_step=local_step,
    N=10,
    init_mode="uniform_bounds",   # uniform_bounds  corners  uniform
    bounds=(-5, 5, -5, 5),
    alpha=0.01,
    wait_for_enter=True
)

optimizer.run()

