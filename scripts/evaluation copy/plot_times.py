

path = "/home/eugeniu/z_tighly_coupled/_0aaa/"
# loosely_times = read_times(path+"mls_alone_0*")
# tightly_times = read_times(path+"mls_alone_1*")

import numpy as np
import matplotlib.pyplot as plt
import glob

fontsize = 14

def read_all_runs(pattern):
    """Read all txt files matching pattern and return as 2D array (runs x timesteps)."""
    files = sorted(glob.glob(pattern))
    runs = []
    for f in files:
        data = np.loadtxt(f)

        data -= 25 #minus 10ms

        runs.append(data)
    return np.vstack(runs)  # shape: (n_runs, n_timesteps)

# Read data 3 and 0 
mls_runs = read_all_runs(path+"mls_alone_*.txt")
loosely_runs = read_all_runs(path+"l_*.txt")
tightly_runs = read_all_runs(path+"t_*.txt")

# Compute mean and std across runs (axis=0 = across runs, per timestep)
mls_mean = np.mean(mls_runs, axis=0)
mls_std  = np.std(mls_runs, axis=0)

loosely_mean = np.mean(loosely_runs, axis=0)
loosely_std  = np.std(loosely_runs, axis=0)

tightly_mean = np.mean(tightly_runs, axis=0)
tightly_std  = np.std(tightly_runs, axis=0)

# X axis = timesteps
timesteps = np.arange(1, loosely_runs.shape[1] + 1)
timesteps_mls = np.arange(1, mls_runs.shape[1] + 1)
timesteps_t = np.arange(1, tightly_runs.shape[1] + 1)

plt.figure(figsize=(10,6))

# Loosely coupled
# for run in mls_runs:
#     plt.plot(timesteps_mls, run, color="green", alpha=0.3)
# plt.plot(timesteps_mls, mls_mean, color="green", linewidth=2, label="MLS mean")
# plt.fill_between(timesteps_mls,
#                  mls_mean - mls_std,
#                  mls_mean + mls_std,
#                  color="green", alpha=0.2)

# Loosely coupled
for run in loosely_runs:
    plt.plot(timesteps, run, color="skyblue", alpha=0.3)
plt.plot(timesteps, loosely_mean, color="tab:blue", linewidth=2, label="Loosely coupled time")
plt.fill_between(timesteps,
                 loosely_mean - loosely_std,
                 loosely_mean + loosely_std,
                 color="tab:blue", alpha=0.2)

# Tightly coupled
for run in tightly_runs:
    plt.plot(timesteps_t, run, color="lightcoral", alpha=0.3)
plt.plot(timesteps_t, tightly_mean, color="tab:red", linewidth=2, label="Tightly coupled time")
plt.fill_between(timesteps_t,
                 tightly_mean - tightly_std,
                 tightly_mean + tightly_std,
                 color="tab:red", alpha=0.2)

plt.xlabel("Scans",fontsize=fontsize)
plt.ylabel("Runtime (ms)", fontsize=fontsize)
#plt.title("Runtime Comparison: Loosely vs Tightly Coupled (10 runs)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()



plt.tick_params(axis='both', which='major', labelsize=fontsize)
plt.legend(fontsize=fontsize)

plt.show()

