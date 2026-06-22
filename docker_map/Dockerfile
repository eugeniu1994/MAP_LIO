# Should automatically select the correct architecture based on which machine it used to build
# FROM ros:noetic-ros-core
FROM osrf/ros:noetic-desktop-full

# Not to stop in the questions
ENV DEBIAN_FRONTEND=noninteractive


SHELL ["/bin/bash", "-c"]

# Tools needed for building ROS packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential                      \
    cmake \
    git \
    nano \
    vim \
    python3-catkin-tools                 \
    python3-osrf-pycommon                \
    ros-noetic-eigen-conversions         \
    ros-noetic-tf                        \
    ros-noetic-tf2-ros                   \
    ros-noetic-pcl-ros                   \
    ros-noetic-gps-common                \
    ros-noetic-geodesy                   \
    ros-noetic-rviz \
    ros-noetic-robot-state-publisher \
    ros-noetic-joint-state-publisher \
    libyaml-cpp-dev                      \
    ros-noetic-hector-trajectory-server \
    ros-noetic-hector-map-server \
    libeigen3-dev \
    libpcl-dev \
    libgeographic-dev \
    geographiclib-tools \
    libgoogle-glog-dev \
    libgflags-dev \
    libsuitesparse-dev \
    && rm -rf /var/lib/apt/lists/*

# Build and install Sophus from source (find_package(Sophus REQUIRED))
COPY MAP_LIO/third_party/Sophus /tmp/Sophus

RUN cd /tmp/Sophus && \
    rm -rf build && \
    mkdir build && cd build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && \
    make install && \
    ldconfig


# Copy libLAS from your local machine
COPY MAP_LIO/third_party/LASzip /tmp/LASzip

RUN cd /tmp/LASzip && \
    rm -rf build && \
    mkdir -p build && cd build && \
    cmake .. && \
    make -j$(nproc) install && \
    ldconfig && \
    rm -rf /tmp/LASzip

COPY MAP_LIO/third_party/libLAS /tmp/liblas
RUN cd /tmp/liblas && \
    rm -rf build && \
    mkdir -p build && cd build && \
    cmake .. \
        -DWITH_GDAL=OFF \
        -DWITH_GEOTIFF=ON \
        -DBUILD_TESTS=OFF && \
    make -j$(nproc) install && \
    ldconfig && \
    rm -rf /tmp/liblas

RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# Create catkin workspace
RUN mkdir -p /root/catkin_ws/src

WORKDIR /root/catkin_ws


COPY MAP_LIO src/MAP_LIO

RUN source /opt/ros/noetic/setup.bash && \
    source ~/.bashrc && \
    catkin_make \
      -DUSE_ALS=ON

# source ROS
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]
