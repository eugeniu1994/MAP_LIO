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

#0.1 px reprojection error
save_dir = "/media/eugeniu/T7/calibration/extrinsic_saved_data_raw/center/"
intr_thermal_ = global_path+"intrinsic_center_thermal_ST.npz"
save_result = save_dir+"extrinsic_center_thermal_to_baseler.npz"

#right - 0.17 px reprojection error
save_dir = "/media/eugeniu/T7/calibration/extrinsic_saved_data_raw/right/"
intr_thermal_ = global_path+"intrinsic_right_thermal_ST.npz"
save_result = save_dir+"extrinsic_right_thermal_to_baseler.npz"

# left - - 0.12 px reprojection error 
save_dir = "/media/eugeniu/T7/calibration/extrinsic_saved_data_raw/left/"
intr_thermal_ = global_path+"intrinsic_left_thermal_ST.npz"
save_result = save_dir+"extrinsic_left_thermal_to_baseler.npz"


thermal_intrinsic = np.load(intr_thermal_, allow_pickle=True)
K_thermal = thermal_intrinsic["K"]
D_thermal = thermal_intrinsic["dist"]

b_path, t_path =  save_dir + "basler/", save_dir + "thermal/"

def load_images(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
    return [np.load(os.path.join(folder, f)) for f in files]

images_16bit = load_images(t_path)
basler = load_images(b_path)

objpoints = []  # 3d point in real world space
imgpoints_l = []  # 2d points in image plane. - thermal camera
imgpoints_r = []  # 2d points in image plane. - rgb camera

square = 0.1  # m (the size of each chessboard square is 10cm)
objp = np.zeros((10 * 7, 3), np.float32) #chessboard is 7x10
objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2) * square
term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 1000, 0.0001)

def stereo_calibrate():
    for thermal_img, basler_img in zip(images_16bit, basler):

        if True:
            valid_pixels = thermal_img[thermal_img > 0]
            min_val = np.percentile(valid_pixels, 1)   #1% of the pixels are darker (colder) than this.
            max_val = np.percentile(valid_pixels, 99)  #99% of the pixels are darker than this
            img_clipped = np.clip(thermal_img, min_val, max_val)
            img8 = ((img_clipped - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8) 
            #invert_img_b = np.array(255.0 - img8, dtype='uint8')
            invert_img_b = img8
                
        ret_t, corners_t = cv2.findChessboardCorners(invert_img_b, (10, 7),
                                                        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret_t:
            cv2.drawChessboardCorners(invert_img_b, (10, 7), corners_t, ret_t)


        ret_b, corners_b = cv2.findChessboardCorners(basler_img, (10, 7),
                                                        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret_b:
            cv2.drawChessboardCorners(basler_img, (10, 7), corners_b, ret_b)


        cv2.imshow("basler_img", cv2.resize(basler_img, None, fx=.4, fy=.4))
        cv2.imshow("img_t", invert_img_b)

        if ret_t and ret_b:
            objpoints.append(objp)
            imgpoints_l.append(corners_t)
            imgpoints_r.append(corners_b)

        key = cv2.waitKey(20)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()

    flags = cv2.CALIB_FIX_INTRINSIC
    print("\n Start extrinsic calibration with {} images".format(len(imgpoints_r)))
    rms_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                objpoints, imgpoints_l, imgpoints_r, K_thermal, D_thermal, K, D, imageSize=None, criteria=term_criteria, flags=flags)

    print('Stereo calibraion Thermal done')
    print('rms_stereo:{}'.format(rms_stereo))
    print('Rotation R')
    print(R)
    print('Translation T')
    print(T)

    np.savez(
            save_result,
            objpoints = np.array(objpoints, dtype=object),
            imgpoints_l = np.array(imgpoints_l, dtype=object),
            imgpoints_r = np.array(imgpoints_r, dtype=object),
            K_thermal = K_thermal,
            D_thermal = D_thermal,
            K_baseler=K,
            D_baseler=D,
            R = R,
            T = T,
            E = E,
            F = F
    )
    print("Saved ",save_result)
    # ===== Read it back =====
    loaded = np.load(save_result, allow_pickle=True)
    K_loaded = loaded["K_thermal"]
    dist_loaded = loaded["D_thermal"]

    print("\n=== Loaded Calibration Data ===")
    print("Loaded Camera Matrix (K):\n", K_loaded)
    print("Loaded Distortion Coefficients:\n", dist_loaded.ravel())

    # Check if saving was correct
    print("\nK match:", np.allclose(K_thermal, K_loaded))
    print("dist match:", np.allclose(D_thermal, dist_loaded))


#stereo_calibrate()

def test_extrinsics():
    loaded = np.load(save_result, allow_pickle=True)
    K_thermal = loaded["K_thermal"]
    D_thermal = loaded["D_thermal"]   
    R = loaded["R"]
    T = loaded["T"]

    #LEFT FRAME IS THE THERMAL CAMERA ALWAYS HERE 
    for thermal_img, basler_img in zip(images_16bit, basler):

        if True:
            valid_pixels = thermal_img[thermal_img > 0]
            min_val = np.percentile(valid_pixels, 1)   #1% of the pixels are darker (colder) than this.
            max_val = np.percentile(valid_pixels, 99)  #99% of the pixels are darker than this
            img_clipped = np.clip(thermal_img, min_val, max_val)
            img8 = ((img_clipped - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8) 
            #invert_img_b = np.array(255.0 - img8, dtype='uint8')
            invert_img_b = img8
                
        ret_t, corners_t = cv2.findChessboardCorners(invert_img_b, (10, 7),
                                                        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret_t:
           cv2.drawChessboardCorners(invert_img_b, (10, 7), corners_t, ret_t)


        ret_b, cornersR = cv2.findChessboardCorners(basler_img, (10, 7),
                                                        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        #if ret_b:
        #    cv2.drawChessboardCorners(basler_img, (10, 7), cornersR, ret_b)

        if ret_t and ret_b:
            success, rvecL, tvecL = cv2.solvePnP(objp, corners_t, K_thermal, D_thermal)

            # Convert rvecL to rotation matrix
            R_chess_L, _ = cv2.Rodrigues(rvecL)

            # Homogeneous transform of chessboard in left frame
            T_chess_L = np.eye(4)
            T_chess_L[:3,:3] = R_chess_L
            T_chess_L[:3, 3] = tvecL.ravel()

            # Stereo extrinsics: transform from left -> right
            T_RL = np.eye(4)
            T_RL[:3,:3] = R  # rotation from left to right
            T_RL[:3, 3]  = T.ravel()  # translation from left to right

            # Transform chessboard pose to right camera frame
            T_chess_R = T_RL @ T_chess_L
            R_chess_R = T_chess_R[:3,:3]
            t_chess_R = T_chess_R[:3, 3]

            # Project 3D object points into right image
            proj_points, _ = cv2.projectPoints(objp, cv2.Rodrigues(R_chess_R)[0], 
                                   t_chess_R, K, D)
            
            if ret_b:
                # cornersR = cv2.cornerSubPix(basler_img, cornersR, (11,11), (-1,-1),
                #                             (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))
                
                error = cv2.norm(cornersR, proj_points, cv2.NORM_L2) / len(proj_points)
                print("Reprojection error between predicted and detected corners:", error)
                cv2.drawChessboardCorners(basler_img, (10, 7), proj_points, ret_b)


        cv2.imshow("basler_img", cv2.resize(basler_img, None, fx=.4, fy=.4))
        cv2.imshow("img_t", invert_img_b)
        
        key = cv2.waitKey(0)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()

test_extrinsics()