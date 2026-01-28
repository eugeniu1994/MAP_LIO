#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from evo.core.trajectory import PoseTrajectory3D
import numpy as np
from evo.core import metrics
from evo.tools import log
import pprint
# from evo.tools import plot
import matplotlib.pyplot as plt
# temporarily override some package settings
from evo.tools.settings import SETTINGS
from evo.core import lie_algebra as lie

SETTINGS.plot_usetex = False
from evo.core import sync
from evo.tools import file_interface
import copy
from matplotlib import pyplot
from termcolor import colored
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams['lines.linewidth'] = 2  

from scipy.stats import norm
from scipy import stats
from scipy.stats import exponpow
from matplotlib.dates import date2num, DateFormatter
import datetime as dt
from matplotlib.lines import Line2D
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from evo.core import lie_algebra as lie
from evo.core.transformations import quaternion_from_matrix, quaternion_matrix
import numpy as np
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
import networkx as nx

max_diff = .01 # 0.000001  #s

max_diff = 0.1

import warnings

warnings.filterwarnings("ignore")

import matplotlib
from scipy.stats import linregress

#-----------------
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point, LineString
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from scipy.ndimage import rotate
# Define custom tile provider
from matplotlib.ticker import FuncFormatter
import random
from xyzservices import TileProvider
from PIL import Image

sns.set()



font_legend = 14
font = 14


font_legend = 16-2
font = 18-2


#PUT THESE BACK 
matplotlib.rc('xtick', labelsize=font)
matplotlib.rc('ytick', labelsize=font)
plt.rcParams.update({'font.size': font})
plt.rc('axes', titlesize=font)     # Set the font size of the title
plt.rc('axes', labelsize=font)     # Set the font size of the x and y labels
plt.rc('xtick', labelsize=font)    # Set the font size of the x tick labels
plt.rc('ytick', labelsize=font)    # Set the font size of the y tick labels
plt.rc('legend', fontsize=font_legend)    # Set the font size of the legend
plt.rc('font', size=font)          # Set the general font size'''

plt.rcParams['hatch.linewidth'] = 2.0  # default is ~1.0

markersize = 15
# plt.rcParams['lines.markersize'] = 10  # radius
sns.set_style("whitegrid")
#----------------------------------------------------------------


translation_ENU_to_origin = np.array([4525805.18109165318310260773, 5070965.88124799355864524841,  114072.22082747340027708560])
rotation_ENU_to_origin = np.array([[-0.78386212744029792887, -0.62091317757072628236,  0.00519529438949398702],
                                    [0.62058202788821892337, -0.78367126767620609584, -0.02715310076057271885],
                                    [0.02093112101431114647, -0.01806018100107483967,  0.99961778597386541367]])

R_origin_to_ENU = rotation_ENU_to_origin.T
t_origin_to_ENU = -R_origin_to_ENU @ translation_ENU_to_origin


bbox_to_anchor=(0.5, -0.12) #bottom
bbox_to_anchor=(0.5, 1.2)   #top
# bbox_to_anchor=(0.5, 1)   #top

ncol = 3
# ncol = 4

# ncol = 2


#used so far
search_radius = 3.0
min_time_diff = 100
distance_upper_bound = 1.0 #distance_upper_bound xy max distance between the points 


class TrajectoryReader(object):
    def __init__(self, path_gt, path_model, model_name = '', path_time='', path_iterations='', align = True):
        self.path_gt = path_gt
        self.path_model = path_model
        self.model_name = model_name
        self.time = [0]
        self.iterations = [0]
        #the prev values
        # traj_gt = file_interface.read_custom_trajectory_file(self.path_gt)
        # traj_model = file_interface.read_custom_trajectory_file(self.path_model)

        traj_gt = file_interface.read_custom_trajectory_file2(self.path_gt)
        traj_model = file_interface.read_custom_trajectory_file2(self.path_model)

        traj_gt = self.clip(traj_gt)

        # self.traj_gt, self.traj_model = traj_gt, traj_model
        self.traj_gt, self.traj_model = sync.associate_trajectories(traj_gt, traj_model, max_diff)

        # if align:
        self.traj_model.align(self.traj_gt, correct_scale=False, correct_only_scale=False, n=-1)     
        # self.traj_model.align_origin(traj_ref=self.traj_gt)
        # self.traj_model.align(self.traj_gt, correct_scale=False, correct_only_scale=False, n=150)     

        
        self.T_origin_to_ENU = np.eye(4)
        self.T_origin_to_ENU[:3, :3] = R_origin_to_ENU
        self.T_origin_to_ENU[:3, 3] = t_origin_to_ENU

        self.traj_gt = self.transform_trajectory_to_ENU(self.traj_gt, self.T_origin_to_ENU)
        self.traj_model = self.transform_trajectory_to_ENU(self.traj_model, self.T_origin_to_ENU)

        print("Reference trajectory:", self.traj_gt.positions_xyz.shape)
        print("Model:", self.traj_model.positions_xyz.shape)

        if path_time != '':
            self.time = np.loadtxt(path_time)[:,2]

        if path_iterations != '':
            self.iterations = np.loadtxt(path_iterations)[:,2]

    def transform_trajectory_to_ENU(self, traj: PoseTrajectory3D, T_origin_to_ENU: np.ndarray) -> PoseTrajectory3D:
        new_poses_se3 = []
        for pose in traj.poses_se3:
            T_local = pose
            T_enu = T_origin_to_ENU @ T_local
            new_poses_se3.append(T_enu)

        test_no_fail = 9999999
        #this is used to show when it fails the ablation study 
        
        # if self.model_name in ['LI', 'LI-VUX', '*-LI', '*-LI-VUX', 'test', 'test_now']:
        #     test_no_fail = 4650

        return PoseTrajectory3D(
            poses_se3=new_poses_se3[:test_no_fail],
            timestamps=traj.timestamps[:test_no_fail])

    def clip(self, traj: PoseTrajectory3D) -> PoseTrajectory3D:
        new_poses_se3 = []
        for pose in traj.poses_se3:
            new_poses_se3.append(pose)

        test_no_fail = 4700 # 5400  # 9999999
        
        # test_no_fail = 1300
        test_no_fail = 196

        # values = [data[label][100:]

        print("Original poses:", np.shape(new_poses_se3))

        test_no_fail = 7780 # 7716
        start = 0 # 100
        ret = PoseTrajectory3D(
            poses_se3=new_poses_se3[start:test_no_fail],
            timestamps=traj.timestamps[start:test_no_fail])
    
        print("Return model poses:", np.shape(ret.poses_se3))

        return ret

    

    def traveled_distance(self, est_xyz: np.ndarray) -> float:
        # Compute differences between consecutive positions
        diffs = np.diff(est_xyz, axis=0)
        
        # Compute Euclidean distances between consecutive points
        segment_lengths = np.linalg.norm(diffs, axis=1)
        
        return np.sum(segment_lengths)

    def APE_translation(self):
        print('\nAPE_translation')
        # Calculate absolute pose error
        ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
        # ape_metric = metrics.APE(metrics.PoseRelation.full_transformation)
        # ape_metric = metrics.APE(metrics.PoseRelation.point_distance)

        ape_metric.process_data((self.traj_gt, self.traj_model))
        self.ape_statistics_t = ape_metric.get_all_statistics()
        print("Model ape_statistics_t:", self.ape_statistics_t)

        self.ape_t_error_vectors = ape_metric.error  # shape (N, 3)
        print("ATE shape:", np.shape(self.ape_t_error_vectors))
    
    def APE_rotation(self):
        print('\nAPE_rotation')
        # Calculate absolute pose error
        ape_metric = metrics.APE(metrics.PoseRelation.rotation_angle_deg)

        ape_metric.process_data((self.traj_gt, self.traj_model))
        self.ape_statistics_r = ape_metric.get_all_statistics()
        print("Model ape_statistics_r:", self.ape_statistics_r)

        self.ape_r_error_vectors = ape_metric.error  # shape (N, 3)
    
    def RPE_translation(self):
        print('\nRPE_translation')
        
        # Calculate relative pose error
        all_pairs = True
        delta_unit = metrics.Unit.meters
        delta = 100

        

        # delta = 1000
        rpe_metric = metrics.RPE(metrics.PoseRelation.translation_part, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)
        # rpe_metric = metrics.RPE(metrics.PoseRelation.point_distance, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)
        # rpe_metric = metrics.RPE(metrics.PoseRelation.full_transformation, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)
        # rpe_metric = metrics.RPE(metrics.PoseRelation.point_distance_error_ratio, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)

        segment_lengths = [100] # [25, 50, 100, 200, 400, 800, 1000]
        
        for i, delta in enumerate(segment_lengths):
            # rpe_metric = metrics.RPE(metrics.PoseRelation.point_distance_error_ratio, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)
            # rpe_metric = metrics.RPE(metrics.PoseRelation.point_distance, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)

            rpe_metric.process_data((self.traj_gt, self.traj_model))
            self.rpe_statistics_t = rpe_metric.get_all_statistics()
            print("Model rpe_statistics_t at {} length:".format(delta), self.rpe_statistics_t)
            print("RPE shape:", np.shape(rpe_metric.error))
            if(i == 0):
                self.rpe_t_error_vectors = rpe_metric.error  # shape (N, 3)
            else:
                self.rpe_t_error_vectors = np.hstack((self.rpe_t_error_vectors, rpe_metric.error)) 

    def RPE_rotation(self):
        print('\nRPE_rotation')
        
        # Calculate relative pose error
        all_pairs = True
        delta_unit = metrics.Unit.meters
        delta = 100
                
        rpe_metric = metrics.RPE(metrics.PoseRelation.rotation_angle_deg, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)

        rpe_metric.process_data((self.traj_gt, self.traj_model))
        self.rpe_statistics_r = rpe_metric.get_all_statistics()
        print("Model rpe_statistics_r:", self.rpe_statistics_r)

        self.rpe_r_error_vectors = rpe_metric.error


path_gt =  "/home/eugeniu/zz_zx_final/ref/MLS_prev.txt"
# path_gt =  "/home/eugeniu/zz_zx_final/ref/MLS_gnss.txt"
path_gt =  "/home/eugeniu/zz_zx_final/ref/MLS.txt"


# todo compare the prev results with the current one se which ones are better 

methods = {
    # 'rko'             : '/home/eugeniu/zz_zx_final/rko',
    # 'point-lio'             : '/home/eugeniu/zz_zx_final/point-lio',

    'fast_lio2'             : '/home/eugeniu/zz_zx_final/fast_lio2', 
    'dlo'             : '/home/eugeniu/zz_zx_final/dlo', 
    # 'dlo-5'             : '/home/eugeniu/zz_zx_final/dlo.5', 

    'a0'             : '/home/eugeniu/zz_zx_final/a0',    #our implementation with no robust no cov, with fixed Fx and Fw 
    'a1'             : '/home/eugeniu/zz_zx_final/a1',    #prev a0 with robust kernel only 
    # 'a2'             : '/home/eugeniu/zz_zx_final/a2',    #our best with fixed prediction and robust & adaptive 
    # 'a2_bar'             : '/home/eugeniu/zz_zx_final/a2_bar',    #a2 with update using original iekf code 
    # 'a3'             : '/home/eugeniu/zz_zx_final/a3',    #a2 with gravity
    'a4'             : '/home/eugeniu/zz_zx_final/a4',    #similar to a2 but with final best weighting 
    
    # 'test'             : '/home/eugeniu/zz_zx_final/test',

    'b0'             : '/home/eugeniu/zz_zx_final/b0',    #a0 with vux
    'b1'             : '/home/eugeniu/zz_zx_final/b1',    #a1 with vux 
    # 'b2'             : '/home/eugeniu/zz_zx_final/b2',    #a2 with added vux
    'b4'             : '/home/eugeniu/zz_zx_final/b4',    #similar to b2 but with final best weighting 

    # 'HeliALS'             : '/home/eugeniu/zz_zx_final/HeliALS',  # a4 LI + HeliALS
    # 'Sparse ALS'             : '/home/eugeniu/zz_zx_final/s_ALS', # a4 LI + s-ALS
    # 'Sparse ALS VUX'             : '/home/eugeniu/zz_zx_final/s_ALS_VUX', # b4 LI VUX + s-ALS

    # 'LI-VUX + S-ALS (l-coupled)' : '/home/eugeniu/z_tighly_coupled/7',
    # 'LI-VUX + S-ALS (t-coupled)' : '/home/eugeniu/z_tighly_coupled/8',


    # 'GNSS_'             : '/home/eugeniu/zz_zx_final/GNSS_INS',   # a4 + GNSS-INS   9 sigma of gnss-ins
    # 'GNSS_VUX'             : '/home/eugeniu/zz_zx_final/GNSS_INS_VUX', #same as GNSS_ + VUX
    # 'GNSS_INS-alone'             : '/home/eugeniu/zz_zx_final/GNSS_INS-alone',
    # 'GNSS_INS-alone-back'             : '/home/eugeniu/z_tighly_coupled/0',

    ## 'Reference trajectory' : '/home/eugeniu/zz_zx_final/ref',
    # '0_LI'                     : '/home/eugeniu/zz_zx_final/0_LI',
    # '1_LI_robust_adaptive' : '/home/eugeniu/zz_zx_final/1_LI_robust_adaptive',
    # '3_LI_robust_adaptive_p2p_p2pl' : '/home/eugeniu/zz_zx_final/3_LI_robust_adaptive_p2p_p2pl',
    # 'madgwick_p2p_p2pl'             : '/home/eugeniu/zz_zx_final/madgwick',
    # 'g_p2p_p2pl'             : '/home/eugeniu/zz_zx_final/g_p2p_p2pl',
    # 'g_p2p_p2pl_backwardPass'             : '/home/eugeniu/zz_zx_final/g_p2p_p2pl_backwardPass',

    # # 'LC_fixed'             : '/home/eugeniu/zz_zx_final/1.0',
    # 'GNSS_'             : '/home/eugeniu/zz_zx_final/GNSS_',

    # 'GNSS_LC'             : '/home/eugeniu/zz_zx_final/GNSS_LC',
    # 'HeliALS'             : '/home/eugeniu/zz_zx_final/3.0',

    # 'prev_t_coupled_s'             : '/home/eugeniu/zz_zx_final/prev_t_coupled_s',

    # 'Sparse ALS'             : '/home/eugeniu/zz_zx_final/4.0',
    # 'Sparse ALS_LC'             : '/home/eugeniu/zz_zx_final/4.1',
    # 'Sparse ALS_LC_GNSS'             : '/home/eugeniu/zz_zx_final/4.2',

    
    

    # 'GNSS_s-ALS'             : '/home/eugeniu/zz_zx_final/GNSS_s-ALS',     #  test gnss + li + sALS
    # 'GNSS_s-ALS_rel'             : '/home/eugeniu/zz_zx_final/GNSS_s-ALS_rel',            #  a4 + rel SE3 +sALS

    # 'test-prev'             : '/home/eugeniu/zz_zx_final/test-prev',  #  a4 + rel SE3
}

methods_ = {
    'GNSS_INS-alone'        : '/home/eugeniu/zz_zx_final/GNSS_INS-alone',
    
    'GNSS-INS'              : '/home/eugeniu/z_tighly_coupled/0',  # '/home/eugeniu/zz_zx_final/GNSS_INS-alone',
    # 'LI'                    : '/home/eugeniu/zz_zx_final/a4',
    # 'LI-VUX'                : '/home/eugeniu/zz_zx_final/b4',

    'LI-VUX + SE3'          : '/home/eugeniu/zz_zx_final/GNSS_INS',    #a4 + GNSS-INS   3 sigma of gnss-ins
    
    # 'test-prev'             : '/home/eugeniu/zz_zx_final/test-prev',
        'test'             : '/home/eugeniu/zz_zx_final/test',  #li-vux + SE3


    'LI-VUX + S-ALS'        : '/home/eugeniu/zz_zx_final/s_ALS',
    'LI-VUX + S-ALS + SE3'  : '/home/eugeniu/zz_zx_final/GNSS_s-ALS_rel',

    'LI-VUX + D-ALS'        : '/home/eugeniu/zz_zx_final/HeliALS',

    
}

methods_data = {
    'Reference trajectory'      : ['#d62728','8'],

    'GNSS-INS'                  : ['#2ca02c','1'],
    'LI'                        : ['#1f77b4','2'],    
    'LI-VUX'                    : ['#ff7f0e','3'],

    'LI-VUX + SE3'              : ['#9467bd','4'], 
    'LI-VUX + S-ALS'            : ['#17becf','5'],
    'LI-VUX + S-ALS + SE3'      : ['#7f7f7f','6'],

    'LI-VUX + D-ALS'            : ['#8c564b','7'],



    'GNSS_INS-alone' : ["#6c2416",'G'],
    'test' : ["#648099",'Z'],
    'test2' : ["#6F1E6A",'f'],

    'LI-*'                      : ["#4d6a7f",'2'],    
    'LI-VUX-*'                  : ['#ff7f0e','3'],
    'LI-**'                     : ["#94afc3",'2'],    
    'LI-VUX-**'                 : ['#ff7f0e','3'],
    'Fast-LIO2'                 : ['#2ca02c','1'],
    'DLIO'                       : ['#d62728','8'],

     'test-prev'      : ["#651e1e",'D'],

     'LI_iesekf': ["#996487",'r'],
     'LI_VUX_iesekf': ["#996487",'v'],

     'x1': ["#690C4A",'v'],
     'x2': ["#649995",'s'],
     'x3': ["#996487",'j'],
     'x4': ["#64998E",'r'],
     'x5': ["#999664",'w'],
     'x6': ["#686499",'q'],

    'z1': ["#690C4A",'v'],
     'z2': ["#649995",'s'],
     'z3': ["#996487",'j'],
     'z4': ["#64998E",'r'],
     'z5': ["#999664",'w'],
     'z6': ["#686499",'q'],

     'xv3': ["#996487",'j'],
     'xv4': ["#64998E",'r'],
     'xv5': ["#999664",'w'],
     'xv6': ["#686499",'q'],
}   



methods = {
    # 'point-lio'             : '/home/eugeniu/zz_zx_final/point-lio',

    # 'Fast-LIO2'             : '/home/eugeniu/zz_zx_final/fast_lio2', 
    # 'DLIO'             : '/home/eugeniu/zz_zx_final/dlo', 

    # 'LI-**'             : '/home/eugeniu/zz_zx_final/a0',    #our implementation with no robust no cov, with fixed Fx and Fw 
    # 'LI-VUX-**'             : '/home/eugeniu/zz_zx_final/b0',    #a0 with vux

    # 'LI-*'             : '/home/eugeniu/zz_zx_final/a1',    #prev a0 with robust kernel only 
    # 'LI-VUX-*'             : '/home/eugeniu/zz_zx_final/b1',    #a1 with vux 

    # 'LI'             : '/home/eugeniu/zz_zx_final/a4',    #similar to a2 but with final best weighting 
    # 'x1'             : '/home/eugeniu/zz_zx_final/x1',  #MAP with analytical J derivation 
    #'z1'             : '/home/eugeniu/zz_zx_final/z1',  #MAP with analytical J derivation 

    # 'LI-VUX'         : '/home/eugeniu/zz_zx_final/b4',    #similar to b2 but with final best weighting 
    # 'x2'             : '/home/eugeniu/zz_zx_final/x2',  #MAP with analytical J derivation  
    # 'z2'             : '/home/eugeniu/zz_zx_final/z2',  #MAP with analytical J derivation 

    
    # 'test'             : '/home/eugeniu/zz_zx_final/test',  #MAP with analytical J derivation  
    # 'GNSS_INS-alone'        : '/home/eugeniu/zz_zx_final/GNSS_INS-alone',
    # 'GNSS-INS'              : '/home/eugeniu/z_tighly_coupled/0',  # '/home/eugeniu/zz_zx_final/GNSS_INS-alone',
    'LI-VUX + SE3'          : '/home/eugeniu/zz_zx_final/GNSS_INS',    #a4 + GNSS-INS   3 sigma of gnss-ins
    'x3'             : '/home/eugeniu/zz_zx_final/x3',
    'xv3'             : '/home/eugeniu/zz_zx_final/xv3',
    #'z3'             : '/home/eugeniu/zz_zx_final/z3',

    'LI-VUX + S-ALS'        : '/home/eugeniu/zz_zx_final/s_ALS',
    'x4'             : '/home/eugeniu/zz_zx_final/x4',
    'xv4'             : '/home/eugeniu/zz_zx_final/xv4',

    'LI-VUX + S-ALS + SE3'  : '/home/eugeniu/zz_zx_final/GNSS_s-ALS_rel',
    'x5'             : '/home/eugeniu/zz_zx_final/x5',
    'xv5'             : '/home/eugeniu/zz_zx_final/xv5',

    'LI-VUX + D-ALS'        : '/home/eugeniu/zz_zx_final/HeliALS',
    'x6'             : '/home/eugeniu/zz_zx_final/x6',
    'xv6'             : '/home/eugeniu/zz_zx_final/xv6',


}   

'''solutions with MAP 
-x1 - LI
-x2 - LI-VUX

the rest are done without VUX 

-x3 - LI-VUX + SE3    
-x4 - LI-VUX + S-ALS
-x5 - LI-VUX + S-ALS + SE3
-x6 - LI-VUX + D-ALS

xv3 - LI-VUX + SE3    
xv4 - LI-VUX + S-ALS
xv5 - LI-VUX + S-ALS + SE3
xv6 - LI-VUX + D-ALS




do everything with MAP 


the ieekf - as adapted from fast-lio2

-z1 - LI
z2 - LI-VUX

the rest are done without VUX 

z3 - LI-VUX + SE3    
z4 - LI-VUX + S-ALS
z5 - LI-VUX + S-ALS + SE3


'''

methods_ = {
    'LI-VUX + S-ALS'        : '/home/eugeniu/zz_zx_final/s_ALS',
    # 'LI-VUX + S-ALS (no scalling)'             : '/home/eugeniu/zz_zx_final/s_ALS_no_Scalling',
    # 'LI-VUX + SE3'          : '/home/eugeniu/zz_zx_final/GNSS_INS',
    # 'LI-VUX + SE3 (no scalling)'          : '/home/eugeniu/zz_zx_final/GNSS_INS_no_Scalling',
    
    'LI-VUX + S-ALS (l-coupled)' : '/home/eugeniu/z_tighly_coupled/7',
    # 'LI-VUX + S-ALS (t-coupled)' : '/home/eugeniu/z_tighly_coupled/8',
}

methods_data_ = {
    
    'LI-VUX + S-ALS'            : ['#17becf','1'],
    'LI-VUX + S-ALS (no scalling)'                       : ['#d62728','2'],

    'LI-VUX + SE3'              : ['#9467bd','3'], 
    'LI-VUX + SE3 (no scalling)'                 : ['#2ca02c','4'],

    'LI-VUX + S-ALS (l-coupled)' : ['#ff7f0e','2'],
    'LI-VUX + S-ALS (t-coupled)' : ["#352c3e",'2'],

}   


# raw_mat = file_interface.read_vel(path_gt)

def funct_debug():
    file_path = "/home/eugeniu/zz_zx_final/madgwick/debugMLS.txt"

    data = np.loadtxt(file_path)
    l = len(data)
    print("data:", np.shape(data))

    data = data[:100000, :]

    time = data[:, 0]

    vel = data[:, 1:4]   # vx, vy, vz
    ba  = data[:, 4:7]   # bax, bay, baz
    bw  = data[:, 7:10]  # bwx, bwy, bwz
    dt  = data[:, 10]

    # Create figure
    # plt.figure(figsize=(12, 8))

    # --- 1. Time ---
    # plt.subplot(2, 2, 1)
    # plt.plot(dt, linewidth=1.5)
    # plt.xlabel("Index")
    # plt.ylabel("Time [s]")
    # plt.title("dt")
    # plt.grid(True)

    # --- 2. Velocity ---
    # plt.subplot(2, 2, 2)
    # vel_vector = np.linalg.norm(vel, axis = 1)
    # print("vel_vector:", np.shape(vel_vector))
    # # plt.plot(time, vel[:, 0], 'r', label='v_x')
    # # plt.plot(time, vel[:, 1], 'g', label='v_y')
    # # plt.plot(time, vel[:, 2], 'b', label='v_z')
    # plt.plot(time, vel_vector, 'k', label='vel_vector')

    # max_vel_m_s = 20 / 3.6
    # vel_line = np.full_like(time, max_vel_m_s)
    # print("vel_line:", np.shape(vel_line))
    # plt.plot(time, vel_line, '--', label='20 km/h')

    # plt.xlabel("Time [s]")
    # plt.ylabel("Velocity [m/s]")
    # plt.title("Velocity")
    # plt.legend()
    # plt.grid(True)
    # plt.draw()

    vel_vector = np.linalg.norm(vel, axis=1)
    print("vel_vector:", vel_vector.shape)

    max_vel_m_s = 30.0 / 3.6  # 30 km/h in m/s
    vel_line = np.full_like(time, max_vel_m_s)

    # --- gain computation ---
    beta_min = 0.001
    beta_max = 0.1

    v_n = vel_vector / max_vel_m_s
    v_n = np.clip(v_n, 0.0, 1.0)

    beta = beta_min + (1.0 - v_n) * (beta_max - beta_min)

    # --- figure ---
    fig, ax1 = plt.subplots(figsize=(12, 8))


    # Differentiate to get acceleration
    acceleration = np.abs(np.diff(vel_vector) / dt[:-1])
    print("acceleration:", np.shape(acceleration))





    # Left Y-axis: velocity
    ax1.plot(time, vel_vector, 'k', label='|v| [m/s]')
    

    ax1.plot(time, vel_line, '--', label='30 km/h')
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Velocity [m/s]")
    ax1.grid(True)

    # Right Y-axis: beta
    ax2 = ax1.twinx()
    ax2.plot(time, beta, 'r', label='β (Madgwick gain vel)')
    ax2.set_ylabel("Gain β")

    # v_n = acceleration / .1 #2 max acceleration
    # v_n = np.clip(v_n, 0.0, 1.0)
    # beta = beta_min + (1.0 - v_n) * (beta_max - beta_min)
    # ax2.plot(time[:-1], beta, 'b', label='β (Madgwick gain acc)')

    # --- combine legends ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title("Velocity and Adaptive Madgwick Gain")
    plt.tight_layout()
    plt.draw()

    plt.figure()
    plt.plot(time[:-1], acceleration, label="acc")
    plt.legend()
    plt.grid(True)

    # --- 3. Accelerometer bias ---
    # plt.subplot(2, 2, 3)
    # plt.figure(figsize=(12, 8))
    # beta_min = 0.001
    # beta_max = 0.1
    # v_n = vel_vector / max_vel_m_s
    # v_n = np.clip(v_n, 0.0, 1.0)

    # beta = beta_min + (1.0 - v_n) * (beta_max - beta_min)
    # plt.plot(time, beta, 'r', label='beta')
    # plt.plot(time, ba[:, 0], 'r', label='ba_x')
    # plt.plot(time, ba[:, 1], 'g', label='ba_y')
    # plt.plot(time, ba[:, 2], 'b', label='ba_z')
    # plt.xlabel("Time [s]")
    # plt.ylabel("Accel Bias")
    # plt.title("Accelerometer Bias (ba)")
    # plt.legend()
    # plt.grid(True)

    # --- 4. Gyroscope bias ---
    # plt.subplot(2, 2, 4)
    # plt.plot(time, bw[:, 0], 'r', label='bw_x')
    # plt.plot(time, bw[:, 1], 'g', label='bw_y')
    # plt.plot(time, bw[:, 2], 'b', label='bw_z')
    # plt.xlabel("Time [s]")
    # plt.ylabel("Gyro Bias")
    # plt.title("Gyroscope Bias (bw)")
    # plt.legend()
    # plt.grid(True)

    plt.tight_layout()
    plt.show()

# funct_debug()
# exit()

methods_data_old = {
    '0_LI' :  ["#1f77b4",'M'],
    '1_LI_robust_adaptive'      : ['#9467bd','D'],
    'GNSS_s-ALS_rel'             : ['#2ca02c','A'],
    '3_LI_robust_adaptive_p2p_p2pl'                 : ['#ff7f0e','C'],
    'HeliALS' : ['#17becf','F'],
    '5_LI_robust_adaptive_backwardPass_p2p_p2pl' : ['#e377c2','H'],

        'LC_fixed'      : ['#9467bd','D'],




    'test' : ["#648099",'Z'],
    'GNSS_INS-alone' : ['#8c564b','G'],

    'Reference trajectory' : ['#d62728','L'],
    
    'LI-VUX + S-ALS (l-coupled)' : ['#bcbd22','E'],
    'LI-VUX + S-ALS (t-coupled)' : ['#17becf','F'],

    'GNSS_LC' : ['#bcbd22','E'],
    'Sparse ALS' : ["#808019",'E'],
    'Sparse ALS VUX': ["#196880",'E'],
    'GNSS_s-ALS': ["#194280",'E'],

    'test-prev'      : ["#651e1e",'D'],
    'b1' : ["#808019",'E'],
    'fast_lio2': ["#194280",'E'],

    'a0' : ["#808019",'E'],
    'a1' : ["#e377c2",'S'],
    'a2' : ["#ff7f0e",'N'],
    'a3' : ['#2ca02c','A'],
    'a4': ["#194280",'E'],
    'b4': ["#194280",'E'],

    'GNSS_'             : ['#2ca02c','A'],
    'GNSS_VUX'          : ["#284828",'A'],
    # # # 'LI-VUX-(raw)GNSS'      : ['#7f7f7f','D'],

    


    

    'b0' : ["#ff7f0e",'N'], #['#04f810','N'],

    'g_p2p_p2pl' : ["#26391B",'Z'],

    'dlo-5' : ["#26391B",'Z'],
    'GNSS_INS-alone-back' : ["#a86b40",'S'],
    'b2' : ["#433123",'S'],
     'dlo' : ["#433123",'S'],
    'a2_bar' : ["#e377c2",'S'],
}


colors = ['tab:brown', 'tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange', 'cyan', 'lime','orange','gray']

linestyles = ['-', '--', '-.', ':', '-', '--', '-.', '-', '--', ':','--', '-.', '-', '--', ':',]
# lab = ['A','B','C','D','E','F','G','H','K','X']
lab = ['0','1','2','3','4','5','6','7','8','9']
lt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

obj_gt = TrajectoryReader(path_gt, path_gt)
all_traj = [obj_gt.traj_gt]
# 3D Plot
fig_3d = plt.figure(figsize=(10, 7))
ax_3d = fig_3d.add_subplot(111, projection='3d')
ax_3d.set_title("3D Trajectories")
ax_3d.set_xlabel("East [m]")
ax_3d.set_ylabel("North [m]")
ax_3d.set_zlabel("Height [m]")


data_ape_t = {}
data_rpe_t = {}
data_ape_r = {}
data_rpe_r = {}
data_time = {}
data_iterations = {}

positions = all_traj[0].positions_xyz 
ax_3d.plot(positions[:, 0], positions[:, 1], positions[:, 2],label="Reference trajectory", color="black")
# ax_2d.plot(positions[:, 0], positions[:, 1],label="Reference trajectory", color="black",) 
x_limits = [np.min(positions[:, 0]), np.max(positions[:, 0])]
y_limits = [np.min(positions[:, 1]), np.max(positions[:, 1])]
z_limits = [np.min(positions[:, 2]), np.max(positions[:, 2])]

# Compute the global range and center
x_range = x_limits[1] - x_limits[0]
y_range = y_limits[1] - y_limits[0]
z_range = z_limits[1] - z_limits[0]
max_range = max(x_range, y_range, z_range)
x_center = np.mean(x_limits)
y_center = np.mean(y_limits)
z_center = np.mean(z_limits)
# Set equal limits for each axis
ax_3d.set_xlim(x_center - max_range/2, x_center + max_range/2)
ax_3d.set_ylim(y_center - max_range/2, y_center + max_range/2)
ax_3d.set_zlim(z_center - max_range/2, z_center + max_range/2)


for idx, (label, path) in enumerate(methods.items()):
    #for label, path in methods.items():
    print('\n\n Model ',label," ==============================================================")
    path_model=path+"/MLS.txt"
    path_time=path+"/MLS_time.txt"
    path_iterations=path+"/MLS_iter.txt"
    obj = TrajectoryReader(path_gt, path_model, label, path_time, path_iterations)

    obj.APE_translation()
    obj.RPE_translation()
    # obj.APE_rotation()
    # obj.RPE_rotation()
    all_traj.append(obj.traj_model)

    data_ape_t[label] = obj.ape_t_error_vectors
    data_rpe_t[label] = obj.rpe_t_error_vectors

    # data_ape_r[label] = obj.ape_r_error_vectors
    # data_rpe_r[label] = obj.rpe_r_error_vectors

    data_time[label] = obj.time
    data_iterations[label] = obj.iterations

    positions = obj.traj_model.positions_xyz  # N x 3 numpy array

    ax_3d.plot(
        positions[:, 0], positions[:, 1], positions[:, 2],
        label=label, color=colors[idx % len(colors)], linestyle=linestyles[idx % len(linestyles)]
    )

ax_3d.legend(loc='upper center', bbox_to_anchor=bbox_to_anchor, #title="Method",
          ncol = ncol, fancybox=True, shadow=True)
ax_3d.grid(True)
plt.draw()

# plt.show()
# exit()

use_hatches = False # True 

hatches = ['','', 'x', '*','x', '*','x', '*' ]

def plot_box_old(data, metric = '',  show_legend = True, show_cumulative = False, add_arrows = False):
    print('plot_box for ',metric)
    labels = list(data.keys())

    labels_local = lab[0:len(labels)]
    lt_local = lt[0:len(labels)]

    plt.figure(figsize=(10, 6))
    

    # values = [data[label] for label in labels]
    values = [data[label][100:] for label in labels] #skip some outliers in the begining 
    # values = [data[label][100:] for label in labels] #skip some outliers in the begining

    box = plt.boxplot(values, patch_artist=True, 
                    # showmeans=False,
                    # meanline=False, 
                    showmeans=True,
                    meanline=True,
                    meanprops=dict(color='black', linestyle='--', linewidth=2.),
                    showfliers=False, notch=False) 

    ind = 0
    legend_handles = []
    labels_local = []
    colors_ = [methods_data[label][0] for label in labels]

    for patch, color, label in zip(box['boxes'], colors_, labels):
    # for patch, label in zip(box['boxes'], labels):
        # color = (random.random(), random.random(), random.random())
        patch.set_facecolor(color)

        if use_hatches:
            hatch = hatches[ind]
            patch.set_hatch(hatch)
            # patch.set_edgecolor('white')   # hatch color


        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
        patch_legend = mpatches.Patch(
            facecolor=color,
            edgecolor='black',
            label=methods_data[label][1] + " : " + label
            # label = label
        )
        
        labels_local.append(methods_data[label][1])
        
        # mean_value = np.mean(data[label])
        # mean_value = np.sqrt(np.mean(data[label]**2)) #rmse

        median_value = np.median(data[label])

        # Median (white dot)
        plt.scatter(
                ind + 1, median_value,
                color='white', edgecolor='black',
                zorder=3, s=40
            )

            # Mean (black dashed line)
        # plt.plot(
        #         [ind + 0.85, ind + 1.15],
        #         [mean_value, mean_value],
        #         color='red', linestyle='--', linewidth=2
        #     )

        # if '*' in label:
        #     patch.set_hatch('\\')  # or 'xx', '\\',  //etc.
        #     patch.set_edgecolor('white') 
        #     patch_legend = mpatches.Patch(
        #         facecolor=color,
        #         edgecolor='white',   
        #         hatch='\\',          
        #         # label=methods_data[label][1] + " : " + label
        #         label = label
        #     )
        
        # legend_handles.append(mpatches.Patch(color=color, label = methods_data[label][1]+" : "+label))
        legend_handles.append(patch_legend)
        if add_arrows:
            mean_value = np.mean(data[label])
            #If the mean is greater than 2, annotate it
            if mean_value > 0.9:
                box['boxes'][ind].set_visible(False)
                # Hide whiskers (2 per box)
                box['whiskers'][2*ind].set_visible(False)
                box['whiskers'][2*ind + 1].set_visible(False)

                box['caps'][2*ind].set_visible(False)
                box['caps'][2*ind + 1].set_visible(False)
                box['medians'][ind].set_visible(False)
                box['means'][ind].set_visible(False)


                x_pos = ind + 1  # boxplot x positions are 1-indexed
                y_base = 0.85       # arrow starts at y = y_base

                # Draw upward arrow and label
                plt.annotate("", xy=(x_pos, y_base + 0.15), xytext=(x_pos, y_base),
                            arrowprops=dict(arrowstyle="->", lw=3, color='black'),
                            xycoords='data', annotation_clip=False)

                plt.text(x_pos - 0.26, y_base + 0.15, f"{mean_value:.2f}",
                        fontsize=font-2, weight=600, color='black')

        ind += 1

    for median_line in box['medians']:
        median_line.set_alpha(0)  # or median_line.set_visible(False)
        #median_line.set(color='grey', linewidth=2, linestyle='dotted')

    for mean_line in box['means']:
        mean_line.set(color='black', linewidth=2, linestyle='--')

    for line in box['whiskers'] + box['caps']:
        line.set(color='black', linewidth=1.2)


    plt.title(f'Box plot of {metric}')
    
    # plt.xticks(np.arange(1, len(labels) + 1), labels, labelsize=font)
    if show_legend:
        plt.legend(handles=legend_handles,  loc='upper center', bbox_to_anchor=bbox_to_anchor, #title="Method",
            ncol = ncol, fancybox=True, shadow=True, fontsize=font_legend)
    plt.grid(True)
    try:
        plt.xticks(lt_local, labels_local, fontsize = font) 
    except:
        pass 
    # if first_got_legend:
    #     plt.legend().set_visible(False)
    
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:.2f}"))   
    
    plt.ylabel(metric, fontsize=font)
    plt.tick_params(axis='both', which='major', labelsize=font)
    plt.draw()

    # plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:.2f}"))  

    if show_cumulative:
        plt.figure(figsize=(10, 5))

        values = [data[label] for label in labels]

        i=0
        for patch, color, label in zip(box['boxes'], colors_, labels):
        # for patch, label in zip(box['boxes'], labels):
            #for i, label in enumerate(data):
            values = np.sort(data[label])
            cdf = np.linspace(0, 1, len(values))
            plt.plot(values, cdf, linewidth=2, color = color, label=label, linestyle = linestyles[i])

            #plt.title(f'Cumulative Distribution of {metric_name}')
            plt.xlabel(metric, fontsize=font)
            plt.ylabel("Cumulative Probability", fontsize=font)
            if show_legend:
                plt.legend( loc='upper center', bbox_to_anchor=bbox_to_anchor, #title="Method",
                ncol = ncol, fancybox=True, shadow=True, fontsize=font_legend)
            plt.grid(True)
            #plt.tight_layout()
            plt.tick_params(axis='both', which='major', labelsize=font)
            plt.draw()
            i+=1

def plot_box(data, metric='', show_legend=True, add_arrows=False):
    print('plot_violin for ', metric)


    labels = list(data.keys())
    values = [data[label] for label in labels]

    values = [data[label][100:] for label in labels] #skip some outliers in the begining 


    def clip_top_percentile(arr, percentile=95):
        """
        Clip array values to the given percentile
        (values above percentile are set to the percentile value)
        """
        arr = np.asarray(arr)
        max_val = np.percentile(arr, percentile)
        return np.clip(arr, None, max_val)

    values = [clip_top_percentile(d, 99) for d in values]

    labels_local = lab[0:len(labels)]
    lt_local = lt[0:len(labels)]

    plt.figure(figsize=(10, 6))

    # Create violin plot
    parts = plt.violinplot(
        values,
        # quantiles = [0.25, 0.75] * len(values),
        showmeans= False,
        showmedians = False,
        showextrema= False,
        # points = 100, #The number of points to evaluate each of the gaussian kernel density estimations at.
        # bw_method = 1.5
    )

    colors_ = [methods_data[label][0] for label in labels]

    legend_handles = []

    # Color each violin
    for i, (body, color, label) in enumerate(zip(parts['bodies'], colors_, labels)):
        body.set_facecolor(color)
        body.set_edgecolor('black')
        body.set_alpha(0.8)
        body.set_linewidth(1.2)
        
        # Legend entry
        patch_legend = mpatches.Patch(
            facecolor=color,
            label=methods_data[label][1] + " : " + label,
            edgecolor='black',
        )
        legend_handles.append(patch_legend)

        # Statistics
        mean_value = np.mean(values[i])
        median_value = np.median(values[i])

        # Optional arrows for large mean values
        if add_arrows and mean_value > 0.9:
            y_base = 0.85
            plt.annotate(
                "",
                xy=(i + 1, y_base + 0.15),
                xytext=(i + 1, y_base),
                arrowprops=dict(arrowstyle="->", lw=3, color='black'),
                annotation_clip=False
            )
            plt.text(
                i + 1 - 0.26, y_base + 0.15,
                f"{mean_value:.2f}",
                fontsize=font - 2, weight=600
            )

            body.set_visible(False)
            continue

        

        # Median (white dot)
        plt.scatter(
            i + 1, median_value,
            color='white', edgecolor='black',
            zorder=3, s=40
        )

        # Mean (black dashed line)
        plt.plot(
            [i + 0.85, i + 1.15],
            [mean_value, mean_value],
            color='black', linestyle='--', linewidth=2
        )

    

    plt.title(f'Violin plot of {metric}')
    plt.ylabel(metric, fontsize=font)

    plt.xticks(lt_local, labels_local, fontsize=font)
    plt.tick_params(axis='both', which='major', labelsize=font)

    plt.grid(True)

    if show_legend:
        plt.legend(
            handles=legend_handles,
            loc='upper center',
            bbox_to_anchor=bbox_to_anchor,
            ncol=ncol,
            fancybox=True,
            shadow=True,
            fontsize=font_legend
        )

    plt.tight_layout()
    plt.draw()


    # plt.figure()
    # for i, (errors, color, label) in enumerate(zip(values, colors_, labels)):
    #     plt.plot(errors, label = label, color = color)

    # plt.legend()
    # plt.draw()

def plot_lines(data, metric = '',  show_legend = True):
    print('plot_lines for ',metric)
    labels = list(data.keys())

    lt_local = lt[0:len(labels)]
    plt.figure(figsize=(10, 6))
    
    for label in labels:
        mean = np.mean(data[label])
        std = np.std(data[label])
        plt.plot(data[label], linewidth=2, color = methods_data[label][0], label = methods_data[label][1] + " : " + label+ " ( mean:" + str(round(mean, 2)) + " ) ( std:" + str(round(std, 2)) + " )")

    plt.title(f'Box plot of {metric}')
    
    # if show_legend:
    #     plt.legend(handles=legend_handles,  loc='upper center', bbox_to_anchor=bbox_to_anchor, #title="Method",
    #         ncol = ncol, fancybox=True, shadow=True, fontsize=font_legend)
    plt.grid(True)
    plt.ylabel(metric, fontsize=font)
    plt.tick_params(axis='both', which='major', labelsize=font)
    plt.draw()
    plt.legend()


def plot_box_violin(
    data,
    metric='',
    show_legend=True,
    add_arrows=False,
    skip_start=100,
    # bw_method=1.5,
    figsize=(10, 6)
):
    labels = list(data.keys())
    values = [np.asarray(data[label][skip_start:]) for label in labels]

    labels_local = lab[0:len(labels)]
    lt_local = lt[0:len(labels)]
    colors_ = [methods_data[label][0] for label in labels]

    plt.figure(figsize=figsize)

    positions = np.arange(1, len(values) + 1)

    # -----------------------------
    # 1) Transparent box plot
    # -----------------------------
    box = plt.boxplot(
        values,
        positions=positions,
        widths=0.15,
        patch_artist=True,
        showfliers=False
    )

    for patch, color in zip(box['boxes'], colors_):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)

    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    for whisker in box['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.2)

    for cap in box['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.2)

    # -----------------------------
    # 2) Violin plot (distribution overlay)
    # -----------------------------
    vp = plt.violinplot(
        values,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        # bw_method=bw_method  bw_method=1.5,
        # points=1000
    )

    legend_handles = []

    for i, (body, color, label) in enumerate(zip(vp['bodies'], colors_, labels)):
        body.set_facecolor(color)
        body.set_edgecolor('black')
        body.set_alpha(0.5)
        body.set_linewidth(1.2)

        # Legend
        patch_legend = mpatches.Patch(
            facecolor=color,
            edgecolor='black',
            label=methods_data[label][1] + " : " + label
        )
        legend_handles.append(patch_legend)

        # Median & mean markers
        median_value = np.median(values[i])
        mean_value = np.mean(values[i])

        plt.scatter(
            i + 1, median_value,
            color='white', edgecolor='black',
            zorder=3, s=40
        )

        plt.plot(
            [i + 0.85, i + 1.15],
            [mean_value, mean_value],
            color='black', linestyle='--', linewidth=2
        )

        # Optional arrow for large means
        if add_arrows and mean_value > 0.9:
            y_base = 0.85
            plt.annotate(
                "",
                xy=(i + 1, y_base + 0.15),
                xytext=(i + 1, y_base),
                arrowprops=dict(arrowstyle="->", lw=3, color='black'),
                annotation_clip=False
            )
            plt.text(
                i + 1 - 0.26, y_base + 0.15,
                f"{mean_value:.2f}",
                fontsize=font - 2, weight=600
            )
            body.set_visible(False)

    # -----------------------------
    # 3) Axes, labels, legend
    # -----------------------------
    plt.xticks(lt_local, labels_local, fontsize=font)
    plt.ylabel(metric, fontsize=font)
    plt.title(f"Box + Violin plot of {metric}")
    plt.tick_params(axis='both', which='major', labelsize=font)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    if show_legend:
        plt.legend(
            handles=legend_handles,
            loc='upper center',
            bbox_to_anchor=bbox_to_anchor,
            ncol=ncol,
            fancybox=True,
            shadow=True,
            fontsize=font_legend
        )

    plt.tight_layout()
    plt.draw()


plot_box_old(data_ape_t, 'ATE translation (m)', show_legend = True, add_arrows = False, show_cumulative = True)
plot_box_old(data_rpe_t, 'RTE translation (%)', show_legend = True, show_cumulative = True)


# plot_lines(data_ape_t, 'ATE translation (m)', show_legend = True)


#to show the scalling effect 
def plot_line_(data, metric = '',  show_legend = True):
    print('plot_line_ for ',metric)
    labels = list(data.keys())

    labels_local = lab[0:len(labels)]
    lt_local = lt[0:len(labels)]

    plt.figure(figsize=(10, 6))
    
    values = [data[label] for label in labels]
    colors_ = [methods_data[label][0] for label in labels]

    for val, color, label in zip(values, colors_, labels):
        plt.plot(val, color = color, linewidth=2, label = methods_data[label][1] + " : " + label)

    
    if show_legend:
        plt.legend( loc='best', #ncol = ncol,  #bbox_to_anchor=bbox_to_anchor, #title="Method",
        fancybox=True, shadow=True, fontsize=font_legend)
            
    plt.grid(True)
    plt.ylabel(metric, fontsize=font)
    plt.xlabel("Scans", fontsize=font)
    # plt.xticks(lt_local, labels_local, fontsize = font) 
    plt.tick_params(axis='both', which='major', labelsize=font)

    plt.draw()

    plt.figure()
    planes = np.loadtxt("/home/eugeniu/zz_zx_final/planes.txt")

    print("planes ", np.shape(planes))

    plt.plot(planes[::4, 0], label = "MLS")
    plt.plot(planes[::4, 1], label = "ALS")
    plt.xlabel('Scans')
    plt.ylabel('Planes')
    plt.legend()
    plt.grid(True)
    plt.draw()

    

# plot_line_(data_ape_t, 'ATE translation (m)',  show_legend = True)
    
    

#to compare the Tightly coupled vs loosely coupled
def f_kernel(data_ate, data_rte, metric_ate = '', metric_rte = ''):

    labels = list(data_ate.keys())
    values_ate = [data_ate[label] for label in labels]
    values_rte = [data_rte[label] for label in labels]

    colors_ = [methods_data[label][0] for label in labels]


    fig, ax1 = plt.subplots(figsize=(7, 4))

    # --- Positions ---
    ate_pos = [0.85, 1.15]   # Absolute (ATE)
    rte_pos = [1.85, 2.15]   # Relative (RTE)
    positions_ate = ate_pos
    positions_rte = rte_pos

    color_t = colors_[0]
    color_l = colors_[1]
    

    # --- Absolute error (ATE) ---
    bp_abs = ax1.boxplot(
        [values_ate[0], values_ate[1]],
        positions=ate_pos,
        widths=0.25,
        showmeans=True,
        meanline=True,
        patch_artist=True,
        # medianprops=dict(marker='o', markerfacecolor='white',
        #                 markeredgecolor='black', markersize=6),
        meanprops=dict(color='black', linestyle='--', linewidth=1.5),
        boxprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        flierprops=dict(marker='.'),
        showfliers=False,
        medianprops=dict(linewidth=0)  # hide default median line
    )
    ax1.set_ylabel(metric_ate)

    for ind, (patch, color, label) in enumerate(zip(bp_abs['boxes'], [color_t, color_l], [labels[0], labels[1]])):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)

        # --- Median (white dot) ---
        median_value = np.median(values_ate[ind])
        plt.scatter(
            positions_ate[ind], median_value,
            color='white', edgecolor='black', zorder=3, s=40
        )

    # --- Relative error (RTE) ---
    ax2 = ax1.twinx()
    bp_rel = ax2.boxplot(
        [values_rte[0], values_rte[1]],
        positions=rte_pos,
        widths=0.25,
        showmeans=True,
        patch_artist=True,
        meanline=True,
        # medianprops=dict(marker='o', markerfacecolor='white',
        #                 markeredgecolor='black', markersize=6),
        meanprops=dict(color='black', linestyle='--', linewidth=1.5),
        boxprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        flierprops=dict(marker='.'),
        showfliers=False,
        medianprops=dict(linewidth=0)
    )

    ax2.set_ylabel(metric_rte)

    
    for median in bp_abs['medians']:
        median.set_visible(False)

    for median in bp_rel['medians']:
        median.set_visible(False)


    

    for ind, (patch, color, label) in enumerate(zip(bp_rel['boxes'], [color_t, color_l], [labels[0], labels[1]])):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)

        # --- Median (white dot) ---
        median_value = np.median(values_rte[ind])
        plt.scatter(
            positions_rte[ind], median_value,
            color='white', edgecolor='black', zorder=3, s=40
        )
        
    legend_handles = [
        mpatches.Patch(
            facecolor=color_t,
            edgecolor='black',
            label=labels[0]
        ),
        mpatches.Patch(
            facecolor=color_l,
            edgecolor='black',
            label=labels[1]
        )
    ]
   
    plt.legend(handles=legend_handles,  loc='upper center', bbox_to_anchor=bbox_to_anchor, #title="Method",
            ncol = ncol, fancybox=True, shadow=True, fontsize=font_legend)

    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(["ATE ", "RTE"])
    ax1.set_xlim(0.5, 2.5)

    ax1.grid(True, axis='y', linestyle=':', linewidth=0.8)
    ax2.grid(False)

    ax1.axvline(x=1.5, color='black', linewidth=0.8, alpha=0.6)


    plt.tight_layout()
    plt.draw()
# f_kernel(data_ape_t, data_rpe_t, 'ATE translation (m)','RTE translation (%)')

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

#to show the effect of weighting and state-of-the-art methods
def plot_box_LI_vs_LIVUX_matplotlib_extended(data, metric):
    """
    Matplotlib boxplot comparing:
    - Fast-LIO2 and DLIO as individual boxes
    - LI and LI-VUX variants grouped by **, *, baseline
    """

    # --- Extra standalone boxes ---
    extra_labels = ['Fast-LIO2', 'DLIO']
    extra_colors = [methods_data[l][0] for l in extra_labels]
    extra_values = [data[l] for l in extra_labels]

    # --- Grouped LI/LI-VUX boxes ---
    families = ['LI', 'LI-VUX']
    variants_ = ['Our(**)', 'Our(*)', 'Our best']  # x-axis labels
    variants = ['**', '*', '']  # used to build labels

    box_width = 0.3
    spacing = 1.0  # space between groups

    # --- Compute x positions ---
    x_positions = []

    # First: positions for extra boxes
    extra_positions = [i * (box_width + 0.2) for i in range(len(extra_labels))]
    x_positions.extend(extra_positions)

    # Next: positions for grouped boxes
    for i, var in enumerate(variants):
        for j, fam in enumerate(families):
            pos = max(extra_positions) + spacing + i * spacing + j * box_width
            x_positions.append(pos)

    # --- Gather values and colors in order ---
    box_values = extra_values.copy()
    box_colors = extra_colors.copy()

    # LI/LI-VUX grouped boxes
    for var in variants:
        for fam in families:
            label = fam if var=='' else f"{fam}-{var}"
            if label in data:
                box_values.append(data[label])
                box_colors.append(methods_data[fam][0])
            else:
                box_values.append([])
                box_colors.append(methods_data[fam][0])

    plt.figure(figsize=(12,6))

    # --- Plot all boxes ---
    box = plt.boxplot(
        box_values,
        positions=x_positions,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        showmeans=False,
        meanline=False,
        notch=False
    )

    # Hide median lines
    for median_line in box['medians']:
        median_line.set_visible(False)

    # Color boxes and add median/mean markers
    for patch, color, values in zip(box['boxes'], box_colors, box_values):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)

    for i, values in enumerate(box_values):
        if len(values) == 0:
            continue
        median_val = np.median(values)
        plt.scatter(
            x_positions[i],
            median_val,
            color='white', edgecolor='black',
            zorder=3, s=40
        )

        mean_val = np.mean(values)
        plt.plot(
            [x_positions[i]-box_width/2, x_positions[i]+box_width/2],
            [mean_val, mean_val],
            color='black', linestyle='--', linewidth=2
        )

    # --- X-ticks ---
    xticks_pos = []

    # Extra boxes
    xticks_pos.extend(extra_positions)
    xticks_labels = extra_labels.copy()

    # Grouped boxes
    for i, var_label in enumerate(variants_):
        # center position between LI and LI-VUX boxes
        left_pos = max(extra_positions) + spacing + i * spacing
        right_pos = left_pos + box_width
        xticks_pos.append((left_pos + right_pos)/2)
        xticks_labels.append(var_label)

    plt.xticks(xticks_pos, xticks_labels, fontsize=font)
    plt.ylabel(metric, fontsize=font)
    plt.title(f'Comparison — {metric}', fontsize=font+2)
    plt.grid(True)

    # --- Legend ---
    legend_handles = []

    # LI and LI-VUX
    for fam in families:
        legend_handles.append(
            mpatches.Patch(facecolor=methods_data[fam][0], edgecolor='black', label=fam)
        )
    # Extra methods
    for lab in extra_labels:
        legend_handles.append(
            mpatches.Patch(facecolor=methods_data[lab][0], edgecolor='black', label=lab)
        )

    plt.legend(handles=legend_handles, fontsize=font_legend)
    plt.tight_layout()
    plt.draw()

# plot_box_LI_vs_LIVUX_matplotlib_extended(data_ape_t, 'ATE translation (m)')
# plot_box_LI_vs_LIVUX_matplotlib_extended(data_rpe_t, 'RTE translation (%)')


# plot_lines(data_time, 'Time (ms)', show_legend = True)
# plot_lines(data_iterations, 'Iterations', show_legend = True)

plt.show()