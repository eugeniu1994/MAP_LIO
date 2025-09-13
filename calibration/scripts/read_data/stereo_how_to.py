#!/usr/bin/env python3

import rosbag
import cv2
import numpy as np
import glob
import os
import pandas as pd
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

global_path = "/media/eugeniu/T7/calibration/saved_data_raw/" #//change this to your system

#Basler RGB
K = np.array(   [[1396.42642353,    0. ,         986.87440169],
                [   0.  ,       1398.23028572  ,607.60275135],
                [   0.       ,     0. ,           1.        ]])

D = np.array( [[-0.12242753 , 0.05002637 , 0.00010842 , 0.00189244 , 0.03743694]])

#Center camera 0.1 px reprojection error
save_dir = "/media/eugeniu/T7/calibration/extrinsic_saved_data_raw/center/"
intr_thermal_ = global_path+"intrinsic_center_thermal_ST.npz"
save_result = save_dir+"extrinsic_center_thermal_to_baseler.npz"

# #right - 0.17 px reprojection error
# save_dir = "/media/eugeniu/T7/calibration/extrinsic_saved_data_raw/right/"
# intr_thermal_ = global_path+"intrinsic_right_thermal_ST.npz"
# save_result = save_dir+"extrinsic_right_thermal_to_baseler.npz"

# # left - - 0.12 px reprojection error 
# save_dir = "/media/eugeniu/T7/calibration/extrinsic_saved_data_raw/left/"
# intr_thermal_ = global_path+"intrinsic_left_thermal_ST.npz"
# save_result = save_dir+"extrinsic_left_thermal_to_baseler.npz"


thermal_intrinsic = np.load(intr_thermal_, allow_pickle=True)
K_thermal = thermal_intrinsic["K"]
D_thermal = thermal_intrinsic["dist"]

# np.savez(
#             save_result,
#             objpoints = np.array(objpoints, dtype=object),
#             imgpoints_l = np.array(imgpoints_l, dtype=object),
#             imgpoints_r = np.array(imgpoints_r, dtype=object),
#             K_thermal = K_thermal,
#             D_thermal = D_thermal,
#             K_baseler=K,
#             D_baseler=D,
#             R = R,
#             T = T,
#             E = E,
#             F = F
#     )

loaded = np.load(save_result, allow_pickle=True)

K_thermal = loaded["K_thermal"]  #intrinsic matrix of thermal cam
D_thermal = loaded["D_thermal"]  #distortion of thermal cam
K_baseler = loaded["K_baseler"]  
D_baseler = loaded["D_baseler"]
R = loaded["R"]                  #extrinsic rotation from thermal cam to basler camera
T = loaded["T"]                  #extrinsic translation from thermal cam to basler camera
E = loaded["E"]
F = loaded["F"] 
