# MAP LIO estimator
Robust Tightly Coupled MLS-ALS Fusion with 2D and 3D LiDAR Using Data-Driven Covariances for Accurate 3D Mapping

## Work in Progress

This code was developed as part of the work for the paper:

**Robust Tightly Coupled MLS-ALS Fusion with 2D/3D LiDAR Using Data-Driven Covariances for Accurate 3D Mapping**, submitted to the *ISPRS Journal of Photogrammetry and Remote Sensing*.

---

## Requirements

To build and run this project, the following libraries are required:

- [Eigen]([http://eigen.tuxfamily.org](https://libeigen.gitlab.io/))
- [PCL (Point Cloud Library)](http://pointclouds.org)
- [Sophus]([https://github.com/strasdat/Sophus](https://github.com/strasdat/Sophus))
For prior map usage with ALS data (`.las` files), the [LASTools](https://lastools.github.io/) library is required.
---

## Building

### Lidar-Inertial Navigation only (no prior map / ALS)
```bash
catkin_make -DCATKIN_WHITELIST_PACKAGES="map_lio" -DUSE_ALS=OFF


### If you want to use prior ALS map data:
```bash 
catkin_make -DCATKIN_WHITELIST_PACKAGES="map_lio" -DUSE_ALS=ON

##  Running
Example of running the system with ROS:
```bash 
roslaunch map_lio hesai.launch bag_file:=bag_file_path
