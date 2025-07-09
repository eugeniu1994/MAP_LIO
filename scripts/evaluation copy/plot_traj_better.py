import geopandas as gpd
import contextily
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point

# Example NumPy array of [Easting, Northing] points (EPSG:3067 - Finnish CRS)
coords = np.array([
    [385000, 6670000],
    [385100, 6670100],
    [385200, 6670200],
    [385300, 6670300],
])

# Convert NumPy array to GeoDataFrame
geometry = [Point(xy) for xy in coords]
trajectory = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:3067")

# Plot
fig, ax = plt.subplots(figsize=(15, 15))
trajectory.plot(ax=ax, marker='o', color='red')

# Set labels and title
plt.xlabel("Easting")
plt.ylabel("Northing")
plt.title("Example NumPy Trajectory")
plt.ticklabel_format(style='plain')

# Get plot limits and adjust to make square
xstart, xend = ax.get_xlim()
ystart, yend = ax.get_ylim()
xlenght = xend - xstart
ylenght = yend - ystart

if xlenght > ylenght:
    step = xlenght / 10
    ymid = ystart + ylenght / 2
    ystart = ymid - xlenght / 2
    yend = ymid + xlenght / 2
else:
    step = ylenght / 10
    xmid = xstart + xlenght / 2
    xstart = xmid - ylenght / 2
    xend = xmid + ylenght / 2

ax.set_xlim(xstart, xend)
ax.set_ylim(ystart, yend)
ax.set_aspect('equal')
ax.xaxis.set_ticks(np.arange(xstart, xend, step))
ax.yaxis.set_ticks(np.arange(ystart, yend, step))

# Style
plt.grid()
plt.xticks(rotation=90)

# Add basemap
contextily.add_basemap(ax, crs='EPSG:3067', source=contextily.providers.OpenStreetMap.Mapnik)

# Show or save
plt.show()
# plt.savefig("trajectory_plot.jpg")
