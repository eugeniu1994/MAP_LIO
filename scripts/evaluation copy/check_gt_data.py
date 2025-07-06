
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Read and parse the uploaded file
file_path = "/media/eugeniu/T7/roamer/evo_20240725_reCalculated2_poqmodouEugent.txt"

df = pd.read_csv(file_path, delim_whitespace=True, skiprows=31)

# Extract required fields
fields_of_interest = ['GPSTime',
    'Easting', 'Northing', 'H-Ell',
    'SDNorth', 'SDEast', 'SDHeight',
    'E-Sep', 'N-Sep', 'H-Sep'
]
#df = df.iloc[1:]
df_selected = df[fields_of_interest]

# Convert all fields to numeric, coercing errors to NaN
for field in fields_of_interest:
    df_selected[field] = pd.to_numeric(df_selected[field], errors='coerce')

# Drop rows with any NaNs in those fields
df_selected.dropna(subset=fields_of_interest, inplace=True)

df_selected = df_selected[(df_selected['GPSTime'] > 380946) & (df_selected['GPSTime'] < 381512.9)]


# Preview the first 5 rows
print("Preview the first 5 rows df:\n",df.head())
print("Preview the first 5 rows df_selected:\n",df_selected.head())


# Plot 3D scatter for trajectory
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')

x = df_selected['Easting']
y = df_selected['Northing']
z = df_selected['H-Ell']

# Plot the points
ax_3d.scatter(x, y, z, c='blue', marker='o')

# Set equal scaling
max_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min()) / 2.0
mid_x = (x.max() + x.min()) * 0.5
mid_y = (y.max() + y.min()) * 0.5
mid_z = (z.max() + z.min()) * 0.5

ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)

ax_3d.set_xlabel('Easting')
ax_3d.set_ylabel('Northing')
ax_3d.set_zlabel('H-Ell')
plt.draw()


# Standard deviations per axis
sd_x = df_selected['SDEast'].values
sd_y = df_selected['SDNorth'].values
sd_z = df_selected['SDHeight'].values

# Separations per axis
sep_x = abs(df_selected['E-Sep'].values)
sep_y = abs(df_selected['N-Sep'].values)
sep_z = abs(df_selected['H-Sep'].values)

# Compute combined std dev per point as Euclidean norm of SDs
combined_sd = np.sqrt(sd_x**2 + sd_y**2 + sd_z**2)      #norm
combined_sep = np.sqrt(sep_x**2 + sep_y**2 + sep_z**2)  #norm

# === Plot 1: Color by combined standard deviation ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=combined_sd, cmap='viridis', s=5)
ax.set_title('Trajectory colored by combined standard deviation')
ax.set_xlabel('Easting')
ax.set_ylabel('Northing')
ax.set_zlabel('H-Ell')
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
plt.colorbar(sc, label='Combined Std Dev')
plt.draw()

# === Plot 2: Color by each separation axis ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=combined_sep, cmap='viridis', s=5)
ax.set_title('Trajectory colored by combined separation')
ax.set_xlabel('Easting')
ax.set_ylabel('Northing')
ax.set_zlabel('H-Ell')
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
plt.colorbar(sc, label='Combined separation')
plt.draw()

# Plot SDs for each axis
fig_sd, axs_sd = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
axs_sd[0].plot(df_selected['SDNorth'], label='SDNorth', color='red')
axs_sd[0].set_ylabel('SDNorth (m)')
axs_sd[0].legend()

axs_sd[1].plot(df_selected['SDEast'], label='SDEast', color='green')
axs_sd[1].set_ylabel('SDEast (m)')
axs_sd[1].legend()

axs_sd[2].plot(df_selected['SDHeight'], label='SDHeight', color='blue')
axs_sd[2].set_ylabel('SDHeight (m)')
axs_sd[2].set_xlabel('Sample Index')
axs_sd[2].legend()
fig_sd.suptitle('Standard Deviations (SD) for Each Axis')

# Plot separations
fig_sep, axs_sep = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
axs_sep[0].plot(df_selected['E-Sep'], label='E-Sep', color='orange')
axs_sep[0].set_ylabel('E-Sep (m)')
axs_sep[0].legend()

axs_sep[1].plot(df_selected['N-Sep'], label='N-Sep', color='purple')
axs_sep[1].set_ylabel('N-Sep (m)')
axs_sep[1].legend()

axs_sep[2].plot(df_selected['H-Sep'], label='H-Sep', color='brown')
axs_sep[2].set_ylabel('H-Sep (m)')
axs_sep[2].set_xlabel('Sample Index')
axs_sep[2].legend()
fig_sep.suptitle('East, North, and Height Separations')

# Plot separations
fig_sep, axs_sep = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
axs_sep[0].plot(abs(df_selected['E-Sep']), label='E-Sep', color='orange')
axs_sep[0].set_ylabel('E-Sep (m)')
axs_sep[0].legend()

axs_sep[1].plot(abs(df_selected['N-Sep']), label='N-Sep', color='purple')
axs_sep[1].set_ylabel('N-Sep (m)')
axs_sep[1].legend()

axs_sep[2].plot(abs(df_selected['H-Sep']), label='H-Sep', color='brown')
axs_sep[2].set_ylabel('H-Sep (m)')
axs_sep[2].set_xlabel('Sample Index')
axs_sep[2].legend()
fig_sep.suptitle('East, North, and Height Separations absolute')

plt.tight_layout()
plt.show()
