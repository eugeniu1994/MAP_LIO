# MAP LIO estimator

## Work in Progress
This code was developed as part of the work for the paper:

**Robust Tightly Coupled MLS-ALS Fusion with 2D/3D LiDAR Using Data-Driven Covariances for Accurate 3D Mapping**, submitted to the *ISPRS Journal of Photogrammetry and Remote Sensing*.
<p align="center">
  <img src="https://github.com/eugeniu1994/MAP_LIO/blob/master/demo.png?raw=true" alt="MAP LIO Estimator Demo" width="600"/>
  <br>
  <em>Example output</em>
</p>

##  ✨ Features

- **MAP-LIO** a single-state maximum a posteriori (MAP) state estimator   
- **Robust data-driven covariances**   
- **Tightly fusion with prior map**:  Airborn laser data was used in the paper
- **Absolute and relative SE(3) integrations**:  provided from GNSS-INS 

---

## Requirements

To build and run this project, the following libraries are required:

- [Eigen]([http://eigen.tuxfamily.org](https://libeigen.gitlab.io/))
- [PCL (Point Cloud Library)](http://pointclouds.org)
- [Sophus]([https://github.com/strasdat/Sophus](https://github.com/strasdat/Sophus))
- For prior map usage with ALS data (`.las` files), the [LASTools](https://lastools.github.io/) library is required.
---

## 🚀 Build Instructions

```sh
cd ~/catkin_ws/src/ #change this according to your system
git clone https://github.com/eugeniu1994/MAP_LIO.git
cd ..
```

### Lidar-Inertial Navigation only (no prior map / ALS)
```bash
catkin_make -DCATKIN_WHITELIST_PACKAGES="map_lio" -DUSE_ALS=OFF
```
## If you want to use prior ALS map data: 
```bash
catkin_make -DCATKIN_WHITELIST_PACKAGES="map_lio" -DUSE_ALS=ON
```

## 📦 Dataset  
Download the toy example ROS bag dataset for quick testing:  
🔗 [Toy Example ROS Bag]( https://drive.google.com/file/d/1uCoBnLeaZuqW00wt41YLrFWCleYauWaX/view?usp=sharing )


##  ▶️ Usage

Example of running the system with ROS:
```bash
roslaunch map_lio hesai.launch bag_file:=bag_file_path/example.bag
```


🛠️ TODO

-Add support for different LiDARs (currently only Hesai is supported)

Note: 
-The experiments in this paper used the 2D Riegl VUX LiDAR. Accessing the raw data requires the proprietary Riegl software API. Due to this dependency, the VUX LiDAR data interface is not included in the public release.

-The ALS dataset cannot be redistributed because its license does not permit resharing. However, the data can be obtained directly from the National Land Survey of Finland:https://www.maanmittauslaitos.fi/en/maps-and-spatial-data/datasets-and-interfaces/product-descriptions/laser-scanning-data-5-p

## ⚖️ License
Academic Research Use Only
This software is provided for academic research purposes only. It is not licensed for commercial use.
By downloading or using this software, you agree to use it solely for non-commercial, research-oriented purposes.
For commercial licensing inquiries, please contact the authors.


📧 Maintainer
Eugeniu Vezeteu
📩 vezeteu.eugeniu@yahoo.com
 











