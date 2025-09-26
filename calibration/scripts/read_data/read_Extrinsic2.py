#!/usr/bin/env python3

import rosbag
import cv2
import numpy as np
import glob
import os
import pandas as pd
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# ----------------------------
# CONFIG
# ----------------------------
bags_folder = "/media/eugeniu/T7/calibration/rosbags/"
csv_file = "/home/eugeniu/Downloads/combined_sync.csv"

# ----------------------------
# CSV READER
# ----------------------------
df = pd.read_csv(csv_file)


basler_pattern = "extrinsic_cameras_camera_basler_front_*.bag"
basler_topic = "/camera_basler_front_24219235/image_raw"
cam_ts = df["Cam_ts"].to_numpy()

#center
save_dir = "/media/eugeniu/T7/calibration/extrinsic_saved_data_raw/center/"
thermal_pattern = "extrinsic_cameras_thermal_camera_Flir_Center_*.bag"
thermal_topic = "/thermal_camera_Flir_Center_0161886/image_raw"
thermal_ts = df["Thermal_center_ts"].to_numpy()
baseler_step = 0 

#right
save_dir = "/media/eugeniu/T7/calibration/extrinsic_saved_data_raw/right/"
thermal_pattern = "extrinsic_cameras_thermal_camera_Flir_Right_*.bag"
thermal_topic = "/thermal_camera_Flir_Right_0161599/image_raw"
thermal_ts = df["Thermal_right_ts"].to_numpy()
baseler_step = 420 #right


#left
save_dir = "/media/eugeniu/T7/calibration/extrinsic_saved_data_raw/left/"
thermal_pattern = "extrinsic_cameras_thermal_camera_Flir_Left_*.bag"
thermal_topic = "/thermal_camera_Flir_Left_0161605/image_raw"
thermal_ts = df["Thermal_left_ts"].to_numpy()
baseler_step = 890 #right


save_dir = "/media/eugeniu/T7/calibration/extrinsic_saved_data_raw/test/"


pattern_size = (10, 7)  
square_size = 0.1 #m  
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)






print("CSV loaded. Example rows:")
print(df.head())

# ----------------------------
# ROSBAG FILES
# ----------------------------

basler_files = glob.glob(bags_folder+basler_pattern)
thermal_files = glob.glob(bags_folder+thermal_pattern)

if not basler_files or not thermal_files:
    print("No rosbag files found. Check patterns.")
    exit(1)

print(f"Found {len(basler_files)} basler bags and {len(thermal_files)} thermal bags.")

# ----------------------------
# HELPERS
# ----------------------------
bridge = CvBridge()

def read_image(img16, is_thermal=True):
    """Convert thermal image to 8-bit displayable."""
    if is_thermal:
        valid_pixels = img16[img16 > 0]
        if len(valid_pixels) == 0:
            return np.zeros_like(img16, dtype=np.uint8)
        min_val = np.percentile(valid_pixels, 1)
        max_val = np.percentile(valid_pixels, 99)
        img_clipped = np.clip(img16, min_val, max_val)
        img8 = ((img_clipped - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)
        return 255 - img8
    else:
        return cv2.cvtColor(img16, cv2.COLOR_BGR2GRAY)

# Function to read messages from a bag file for a specific topic
def read_bag_messages(bag_file, topic):
    images = []
    with rosbag.Bag(bag_file, "r") as bag:
        for topic_name, msg, t in bag.read_messages(topics=[topic]):
            images.append((t, msg))  # store timestamp and message
    return images


# Read all Thermal bags
thermal_images = []
for bag_file in thermal_files:
    thermal_images.extend(read_bag_messages(bag_file, thermal_topic))

print(f"Total Thermal images: {len(thermal_images)}")


b = 0

image_counter = 0
try:
    j = 0
    for i in range(len(basler_files)):
        print("read basler bag:",i)
        with rosbag.Bag(basler_files[i], "r") as bag_basler:
            basler_iter = bag_basler.read_messages(topics=[basler_topic])

            while True:
                try:
                    topic_b, msg_b, t_b = next(basler_iter)
                except StopIteration:
                    print("End of bags reached - error")
                    break

                # --- Basler image ---
                ts_b = t_b.to_nsec() #msg_b.header.stamp.to_sec()
                img_b = bridge.imgmsg_to_cv2(msg_b, desired_encoding="bgr8")
                idx_b = np.argmin(np.abs(cam_ts - ts_b)) 
                e_t_t = thermal_ts[idx_b]

                b+=1
                print("B:",b)
                if b < baseler_step:
                    continue 
            
                #print("idx_b:",idx_b,", e_t_t:",e_t_t)
                found_b, corners = cv2.findChessboardCorners(img_b, pattern_size)
                copy_b = img_b.copy()
                if found_b:
                     cv2.drawChessboardCorners(copy_b, pattern_size, corners, found_b)

                found_b = True 
                cv2.imshow("img_b", cv2.resize(copy_b, None, fx=.4, fy=.4))
                cv2.waitKey(1)

                #continue 
                while True:
                    t_t, msg_t = thermal_images[j]
                    j+=1

                    ts_t = t_t.to_nsec()
                    if ts_t == e_t_t:
                        #print("use picture")
                        img_t = bridge.imgmsg_to_cv2(msg_t, desired_encoding="passthrough")
                        img_t = read_image(img_t, is_thermal=True)
                        

                        found_t, corners = cv2.findChessboardCorners(img_t, pattern_size)
                        copy_t = img_t.copy()
                        if found_t:
                            cv2.drawChessboardCorners(copy_t, pattern_size, corners, found_b)

                        
                        cv2.imshow("img_t", copy_t)
                        key = cv2.waitKey(0) 

                        found_t = True 

                        if key == ord('s') or key == ord('S'):
                            if found_b and found_t:
                                filename_t = os.path.join(save_dir, f"thermal/image_{image_counter:04d}.npy")
                                np.save(filename_t, img_t)
                                print(f"\nSaved: {filename_t}")

                                filename_b = os.path.join(save_dir, f"basler/image_{image_counter:04d}.npy")
                                np.save(filename_b, img_b)
                                print(f"Saved: {filename_b}")

                                image_counter += 1
                        
                        if key == ord('q') or key == 27: 
                            break

                    elif ts_t > e_t_t:
                        break 
                      
finally:
    print('Finally')

cv2.destroyAllWindows()
print("Done.")
