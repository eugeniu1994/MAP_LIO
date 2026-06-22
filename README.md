# MAP LIO estimator

## Work in Progress
This code was developed as part of the work for the paper:

**Robust Tightly Coupled MLS-ALS Fusion with 2D/3D LiDAR Using Local Geometric Uncertainty for Accurate 3D Mapping**
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

---

## Requirements

To build and run this project, the following libraries are required:

- [Eigen]([http://eigen.tuxfamily.org](https://libeigen.gitlab.io/))
- [PCL (Point Cloud Library)](http://pointclouds.org)
- [Sophus]
- For prior map usage with ALS data (`.las` files), the [LASTools](https://lastools.github.io/) library is required.
---

## 🚀 Build Instructions

```sh
cd ~/catkin_ws/src/ #change this according to your system
git clone --recurse-submodules https://github.com/eugeniu1994/MAP_LIO.git
cd ..
```

### Dependencies

```bash
# Detect ROS version automatically
ROS_DISTRO=$(. /opt/ros/*/setup.bash 2>/dev/null && echo $ROS_DISTRO)

if [ -z "$ROS_DISTRO" ]; then
    echo "ROS not found. Please install ROS first."
    exit 1
fi

echo "Detected ROS distro: $ROS_DISTRO"

sudo apt-get update && sudo apt-get install -y \
    ros-$ROS_DISTRO-pcl-ros \
    ros-$ROS_DISTRO-gps-common \
    ros-$ROS_DISTRO-geodesy \
    ros-$ROS_DISTRO-robot-state-publisher \
    ros-$ROS_DISTRO-joint-state-publisher \
    ros-$ROS_DISTRO-hector-trajectory-server \
    ros-$ROS_DISTRO-hector-map-server \
    libyaml-cpp-dev libeigen3-dev libpcl-dev \
    libgeographic-dev geographiclib-tools \
    libgoogle-glog-dev libgflags-dev libsuitesparse-dev

echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc
```
```bash
# ===== Build Sophus (from third_party) =====
cd third_party/Sophus
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
make -j$(nproc)
sudo make install
sudo ldconfig
cd ../../..
```
```bash
#YOU CAN SKIP THESE IF NOT PRIOR ALS MAP IS USED
# ===== Build LASzip (from third_party) =====
cd third_party/LASzip
rm -rf build
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig
cd ../../..

# ===== Build libLAS (from third_party) =====
cd third_party/libLAS
rm -rf build
mkdir build && cd build
cmake .. \
    -DWITH_GDAL=OFF \
    -DWITH_GEOTIFF=ON \
    -DBUILD_TESTS=OFF
make -j$(nproc)
sudo make install
sudo ldconfig
cd ../../..
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


## Docker Build and Run 

```bash
./Build_Run.sh  #from /docker_map
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash

roslaunch map_lio hesai.launch bag_file:=bag file here.bag
```

## :pencil: Citation

If you use this work in your research or software, see the following papers:
```
@article{vezeteu2026robust,
  title={Robust tightly coupled MLS--ALS fusion with 2D and 3D LiDAR using local geometric uncertainty for accurate 3D mapping},
  author={Vezeteu, Eugeniu and El Issaoui, Aimad and Hyyti, Heikki and Muhojoki, Jesse and Hyypp{\"a}, Eric and Manninen, Petri and Hakala, Teemu and Kukko, Antero and Kaartinen, Harri and Kyrki, Ville and others},
  journal={Science of Remote Sensing},
  volume={13},
  pages={100449},
  year={2026},
  publisher={Elsevier}
}
```
```
@article{vezeteu2025direct,
  title={Direct 3D mapping with a 2D LiDAR using sparse reference maps},
  author={Vezeteu, Eugeniu and El Issaoui, Aimad and Hyyti, Heikki and Muhojoki, Jesse and Manninen, Petri and Hakala, Teemu and Hyypp{\"a}, Eric and Kukko, Antero and Kaartinen, Harri and Kyrki, Ville and others},
  journal={ISPRS Open Journal of Photogrammetry and Remote Sensing},
  pages={100109},
  year={2025},
  publisher={Elsevier}
}
```
```
@article{vezeteu2025direct,
  title={Direct integration of ALS and MLS for real-time localization and mapping},
  author={Vezeteu, Eugeniu and El Issaoui, Aimad and Hyyti, Heikki and Hakala, Teemu and Muhojoki, Jesse and Hyypp{\"a}, Eric and Kukko, Antero and Kaartinen, Harri and Kyrki, Ville and Hyypp{\"a}, Juha},
  journal={ISPRS Open Journal of Photogrammetry and Remote Sensing},
  volume={16},
  pages={100088},
  year={2025},
  publisher={Elsevier}
}
```



🛠️# TODO

-Add support for different LiDARs (currently only Hesai is supported)

Note: 
-The experiments in this paper used the 2D Riegl VUX LiDAR. Accessing the raw data requires the proprietary Riegl software API. Due to this dependency, the VUX LiDAR data interface is not included in the public release.

-The ALS dataset cannot be redistributed because its license does not permit resharing. However, the data can be obtained directly from the National Land Survey of Finland:https://www.maanmittauslaitos.fi/en/maps-and-spatial-data/datasets-and-interfaces/product-descriptions/laser-scanning-data-5-p
The released code assumes that the prior map is supplied as 50 × 50 m tiles in LAS format.

## ⚖️ License
Academic Research Use Only
This software is provided for academic research purposes only. It is not licensed for commercial use.
By downloading or using this software, you agree to use it solely for non-commercial, research-oriented purposes.
For commercial licensing inquiries, please contact the authors.


📧 Maintainer
Eugeniu Vezeteu
📩 vezeteu.eugeniu@yahoo.com
 











