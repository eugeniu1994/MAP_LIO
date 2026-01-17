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

max_diff = 0.000001  #s

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

from xyzservices import TileProvider
from PIL import Image

sns.set()

font = 14

font = 22 #used for google map images 
# font = 26 #for plots 
font_legend = 18
font = 22 



font_legend = 16
font = 16

# font = 28

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
bbox_to_anchor=(0.5, 1.3)   #top

bbox_to_anchor=(0.5, 1.2)   #top

ncol = 3
# ncol = 4

# ncol = 2

def plot_trajectory_(xyz_enu, etrs_tm35fin = 'EPSG:3067', epsg = 3067):
    start = xyz_enu[0]
    print('start:',start)
    end = xyz_enu[len(xyz_enu) - 1]
    # Create a GeoDataFrame with the EVO location point in the ETRS-TM35FIN projection

    middle_index = len(xyz_enu) // 2
    middle_point = xyz_enu[middle_index]

    evo_point = gpd.GeoDataFrame(
        #{'geometry': [Point(start[0],start[1])]},
        {'geometry': [Point(middle_point[0],middle_point[1])]},
        crs=etrs_tm35fin
    )

    evo_area = evo_point.buffer(500)  # 500 meters 

    trajectory_points_etrstm35fin = xyz_enu[::30,:2]

    # Create GeoDataFrame from the points (ETRS-TM35FIN)
    trajectory_gdf = gpd.GeoDataFrame(
        {'geometry': [Point(x, y) for x, y in trajectory_points_etrstm35fin]},
        crs=etrs_tm35fin
    )

    # Reproject the trajectory back to Web Mercator (EPSG:3857) for plotting with the basemap
    trajectory_gdf_mercator = trajectory_gdf.to_crs(epsg=epsg)  

    fig, axis = plt.subplots(figsize=(20, 20))

    # Plot the area boundary
    evo_area.boundary.plot(ax=axis, color='gray', linewidth=0)
    # Plot the trajectory points in Web Mercator (projected for plotting)
    trajectory_gdf_mercator.plot(ax=axis, color='red', marker='o', markersize=markersize, label='Reference trajectory', zorder=10)

    #ctx.add_basemap(axaxis, source=ctx.providers.Esri.WorldImagery, crs=etrs_tm35fin)
    #ctx.add_basemap(axis, source=ctx.providers.OpenStreetMap.Mapnik, crs=etrs_tm35fin)

    
    google_sat = TileProvider(
        name="Google Satellite",
        url="http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attribution="Google",
    )
    ctx.add_basemap(axis, source=google_sat, crs=trajectory_gdf.crs)

    axis.set_xlabel('East (m)')
    axis.set_ylabel('North (m)')
    plt.legend()
    plt.grid(False)

    # Set the formatter for x and y axis to avoid scientific notation
    # axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
    # axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))

    # Specify the data coordinates where to place the image
    xy = (end[0]-5, end[1])

    image = mpimg.imread('/home/eugeniu/x_vux_mls_als_paper/car_small.png')  # Use a small PNG file
    image = rotate(image, angle=95, reshape=True)
    imagebox = OffsetImage(image, zoom=.2)  # Adjust zoom to make the image smaller or larger

    ab = AnnotationBbox(imagebox, xy,
                        xycoords='data',
                        frameon=False,
                        zorder=10)  # No frame around image

    axis.add_artist(ab)

    return fig, axis

def plot_trajectory(xyz_enu, other_traj = [], other_labels=[], etrs_tm35fin = 'EPSG:3067', epsg = 3067):
    start = xyz_enu[0]
    print('start:',start)
    end = xyz_enu[len(xyz_enu) - 1]
    # Create a GeoDataFrame with the EVO location point in the ETRS-TM35FIN projection

    middle_index = len(xyz_enu) // 2
    middle_point = xyz_enu[middle_index]

    evo_point = gpd.GeoDataFrame(
        #{'geometry': [Point(start[0],start[1])]},
        {'geometry': [Point(middle_point[0],middle_point[1])]},
        crs=etrs_tm35fin
    )

    evo_area = evo_point.buffer(500)  # 500 meters 

    trajectory_points_etrstm35fin = xyz_enu[::30,:2]

    # Create GeoDataFrame from the points (ETRS-TM35FIN)
    trajectory_gdf = gpd.GeoDataFrame(
        {'geometry': [Point(x, y) for x, y in trajectory_points_etrstm35fin]},
        crs=etrs_tm35fin
    )
    # Reproject the trajectory back to Web Mercator (EPSG:3857) for plotting with the basemap
    trajectory_gdf_mercator = trajectory_gdf.to_crs(epsg=epsg)  

    fig, axis = plt.subplots(figsize=(20, 20))

    # Plot the area boundary
    evo_area.boundary.plot(ax=axis, color='gray', linewidth=0)
    # Plot the trajectory points in Web Mercator (projected for plotting)
    trajectory_gdf_mercator.plot(ax=axis, color='red', marker='o', markersize=markersize, label='Reference trajectory',  zorder=10) #

    colors = ["#1f77b4",  
            '#ff7f0e',   
            "#EFD700",  
            "#04f810"]  
    
    markers = ['o','^','D','d']
    
    colors = ["#1f77b4",  
            '#ff7f0e',   
            "#e377c2"]  
    
    markers = ['o','o','o',]
    for i in range(len(other_traj)):
        traj = other_traj[i]
        l = other_labels[i]

        points_etrstm35fin = traj[::15,:2]
        trajectory_gdf = gpd.GeoDataFrame(
            {'geometry': [Point(x, y) for x, y in points_etrstm35fin]},
            crs=etrs_tm35fin
        )
        trajectory_gdf_mercator = trajectory_gdf.to_crs(epsg=epsg)
        alpha = .7
        # alpha = 1
        trajectory_gdf_mercator.plot(ax=axis, color=colors[i], marker=markers[i], markersize=markersize, alpha=alpha, label=l, zorder=11)


    #ctx.add_basemap(axaxis, source=ctx.providers.Esri.WorldImagery, crs=etrs_tm35fin)
    #ctx.add_basemap(axis, source=ctx.providers.OpenStreetMap.Mapnik, crs=etrs_tm35fin)

    google_sat = TileProvider(
        name="Google Satellite",
        url="http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attribution="Google",
    )
    ctx.add_basemap(axis, source=google_sat, crs=trajectory_gdf.crs)

    axis.set_xlabel('East (m)')
    axis.set_ylabel('North (m)')
    
    # plt.legend()
    plt.grid(False)


    # Set the formatter for x and y axis to avoid scientific notation
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))

    # Specify the data coordinates where to place the image
    xy = (end[0]-5, end[1])

    image = mpimg.imread('/home/eugeniu/x_vux_mls_als_paper/car_small.png')  # Use a small PNG file
    image = rotate(image, angle=95, reshape=True)
    imagebox = OffsetImage(image, zoom=.2)  # Adjust zoom to make the image smaller or larger

    ab = AnnotationBbox(imagebox, xy,
                        xycoords='data',
                        frameon=False,
                        zorder=10)  # No frame around image

    axis.add_artist(ab)
    plt.legend(markerscale=3) 

    return fig, axis

#used so far
search_radius = 3.0
min_time_diff = 100
distance_upper_bound = 1.0 #distance_upper_bound xy max distance between the points 


#new better config - redoo the tests
# search_radius = 1. # 3.0
# min_time_diff = 250
# distance_upper_bound = .2 # xy max distance between the points 


# search_radius = 1.0
# min_time_diff = 250
# distance_upper_bound = 1.0 #distance_upper_bound xy max distance between the points 

def show_time():
    import glob
    path = "/home/eugeniu/z_tighly_coupled/_0aaa/"
    # loosely_times = read_times(path+"mls_alone_0*")
    # tightly_times = read_times(path+"mls_alone_1*")
    def read_all_runs(pattern):
        """Read all txt files matching pattern and return as 2D array (runs x timesteps)."""
        files = sorted(glob.glob(pattern))
        runs = []
        for f in files:
            data = np.loadtxt(f)

            data -= 25 #minus 10ms

            runs.append(data)
        return np.vstack(runs)  # shape: (n_runs, n_timesteps)

    # Read data 3 and 0 
    mls_runs = read_all_runs(path+"mls_alone_*.txt")
    loosely_runs = read_all_runs(path+"l_*.txt")
    tightly_runs = read_all_runs(path+"t_*.txt")

    # Compute mean and std across runs (axis=0 = across runs, per timestep)
    mls_mean = np.mean(mls_runs, axis=0)
    mls_std  = np.std(mls_runs, axis=0)

    loosely_mean = np.mean(loosely_runs, axis=0)
    loosely_std  = np.std(loosely_runs, axis=0)

    tightly_mean = np.mean(tightly_runs, axis=0)
    tightly_std  = np.std(tightly_runs, axis=0)

    # X axis = timesteps
    timesteps = np.arange(1, loosely_runs.shape[1] + 1)
    timesteps_mls = np.arange(1, mls_runs.shape[1] + 1)
    timesteps_t = np.arange(1, tightly_runs.shape[1] + 1)

    plt.figure(figsize=(10,6))

    fontsize = 22 # 24
    # Loosely coupled
    # for run in loosely_runs:
    #     plt.plot(timesteps, run, color="skyblue", alpha=0.3)
    plt.plot(timesteps, loosely_mean, color="tab:blue", linewidth=2, label="Loosely coupled")
    plt.fill_between(timesteps,
                    loosely_mean - loosely_std,
                    loosely_mean + loosely_std,
                    color="tab:blue", alpha=0.2)

    # Tightly coupled
    # for run in tightly_runs:
    #     plt.plot(timesteps_t, run, color="lightcoral", alpha=0.3)
    plt.plot(timesteps_t, tightly_mean, color="tab:red", linewidth=2, label="Tightly coupled")
    plt.fill_between(timesteps_t,
                    tightly_mean - tightly_std,
                    tightly_mean + tightly_std,
                    color="tab:red", alpha=0.2)

    plt.xlabel("Scans",fontsize=fontsize)
    plt.ylabel("Runtime (ms)", fontsize=fontsize)
    #plt.title("Runtime Comparison: Loosely vs Tightly Coupled (10 runs)")
    plt.legend()
    plt.grid(True) #, alpha=0.6
    plt.tight_layout()



    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.legend(fontsize=fontsize)

    plt.show()

# show_time()

class TrajectoryReader(object):
    def __init__(self, path_gt, path_model, model_name = '', align = True):
        self.path_gt = path_gt
        self.path_model = path_model
        self.model_name = model_name

        #the prev values
        # traj_gt = file_interface.read_custom_trajectory_file(self.path_gt)
        # traj_model = file_interface.read_custom_trajectory_file(self.path_model)

        traj_gt = file_interface.read_custom_trajectory_file2(self.path_gt)
        traj_model = file_interface.read_custom_trajectory_file2(self.path_model)

        self.traj_gt, self.traj_model = sync.associate_trajectories(traj_gt, traj_model, max_diff)

        if align:
            self.traj_model.align(self.traj_gt, correct_scale=False, correct_only_scale=False, n=-1) 
            
            # self.traj_model.align_origin(traj_ref=self.traj_gt)
        
        self.T_origin_to_ENU = np.eye(4)
        self.T_origin_to_ENU[:3, :3] = R_origin_to_ENU
        self.T_origin_to_ENU[:3, 3] = t_origin_to_ENU

        self.traj_gt = self.transform_trajectory_to_ENU(self.traj_gt, self.T_origin_to_ENU)
        self.traj_model = self.transform_trajectory_to_ENU(self.traj_model, self.T_origin_to_ENU)

        print("Reference trajectory:", self.traj_gt.positions_xyz.shape)
        print("Model:", self.traj_model.positions_xyz.shape)

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

    def plot_data(self):
        # fig = plt.figure(figsize=(10, 6))  
        # plot.trajectories(
        #     fig,  
        #     [self.traj_gt, self.traj_model], 
        #     plot.PlotMode.xyz,
        #     title="Aligned Trajectories")
                
        ref = self.traj_gt.positions_xyz
        est = self.traj_model.positions_xyz
        print('self.ape_t_error_vectors:', np.shape(self.ape_t_error_vectors))
        print('ref:', np.shape(ref))

        # Create segments for the reference trajectory (colored by error)
        ref_points = ref[:, :2]  # (N, 2)
        ref_segments = np.stack([ref_points[:-1], ref_points[1:]], axis=1)  # (N-1, 2, 2)
        ref_colors = self.ape_t_error_vectors[:-1]  # (N-1,)

        # Line collection for the reference trajectory
        ref_lc = LineCollection(ref_segments, cmap='viridis', linewidth=2.5,
                                norm=plt.Normalize(vmin=ref_colors.min(), vmax=ref_colors.max()))
        ref_lc.set_array(ref_colors)

        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        est_points = est[:, :2]
        ax.plot(est_points[:, 0], est_points[:, 1], color='black', alpha=0.8, linewidth=1, label="{}".format(self.model_name))
        ax.add_collection(ref_lc)

        # Colorbar for error values
        cbar = fig.colorbar(ref_lc, ax=ax)
        cbar.set_label('Error Norm [m]')

        # Aesthetics
        ax.set_title("Trajectory with Absolute Errors (XY plane) for {}".format(self.model_name))
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        #ax.legend()
        ax.legend( loc='upper center', bbox_to_anchor=bbox_to_anchor, #title="Method",
          ncol = ncol, fancybox=True, shadow=True)

        plt.draw()

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

        ape_metric.process_data((self.traj_gt, self.traj_model))
        self.ape_statistics_t = ape_metric.get_all_statistics()
        print("Model ape_statistics_t:", self.ape_statistics_t)

        self.ape_t_error_vectors = ape_metric.error  # shape (N, 3)
    
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
                
        # rpe_metric = metrics.RPE(metrics.PoseRelation.translation_part, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)
        # rpe_metric = metrics.RPE(metrics.PoseRelation.point_distance_error_ratio, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)
        rpe_metric = metrics.RPE(metrics.PoseRelation.point_distance, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)

        rpe_metric.process_data((self.traj_gt, self.traj_model))
        self.rpe_statistics_t = rpe_metric.get_all_statistics()
        print("Model rpe_statistics_t:", self.rpe_statistics_t)

        self.rpe_t_error_vectors = rpe_metric.error  # shape (N, 3)

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

    
    def overlap_error(self, est_xyz, label, search_radius = search_radius, min_time_diff = min_time_diff, plot = False):
        traj = est_xyz
        xy = traj[:, :2]
        # Build KD-tree on XY coordinates
        tree = cKDTree(xy)
        
        # Step 1: Find overlapping point pairs
        self.overlap_pairs = []
        for i, point in enumerate(xy):
            # Query neighbors within radius (exclude self)
            idxs = tree.query_ball_point(point, r=search_radius)
            # Filter neighbors that are too close in time (to avoid sequential points)
            idxs = [j for j in idxs if abs(j - i) > min_time_diff]
            for j in idxs:
                self.overlap_pairs.append((i, j))

        print('overlap_pairs:', len(self.overlap_pairs))

        # Step 2: Build graph and find connected components
        G = nx.Graph()
        G.add_edges_from(self.overlap_pairs)
        components = list(nx.connected_components(G))
        min_segment_size = 5
        overlap_segments = [sorted(list(c)) for c in components if len(c) >= min_segment_size]
        print(f"Found {len(overlap_segments)} overlapping segments")

        # Step 3: Plot trajectory with highlighted overlaps
        f_map,axis_map = plot_trajectory(est_xyz, etrs_tm35fin = 'EPSG:3067', epsg = 3067)

        plt.draw()

        cmap = plt.cm.get_cmap('tab10')

        cols_ = np.array(['tab:orange', "tab:blue"])

        cols_ = np.array(['tab:orange', "tab:cyan"])

        if plot:
            plt.figure(figsize=(12, 8))
            plt.plot(xy[:, 0], xy[:, 1], label='Trajectory: {}'.format(label), color='gray', alpha=0.5)
            colors = plt.cm.get_cmap('tab10', len(overlap_segments))
            i = 0
            import matplotlib.lines as mlines
            legend_marker_trajectory = mlines.Line2D(
                [], [],
                color='red',
                marker='o',
                linestyle='None',
                linewidth=2,
                markersize=10    # <-- legend marker size here
            )
            handles = []
            labels_ = []

            handles.append(legend_marker_trajectory)
            labels_.append("Reference trajectory")
            for idx, segment in enumerate(overlap_segments):
                pts = xy[segment]
                plt.plot(pts[:, 0], pts[:, 1], '.', color=colors(idx), label=f'Overlapped segment {idx+1}')
                print('idx:', idx, ', pts:', np.shape(pts), ', segment:', np.shape(segment))
                label = f'Overlapped segment {idx+1}'
                axis_map.plot(pts[::30, 0], pts[::30, 1], '*', markersize=markersize, alpha=0.5, color=cols_[i], zorder = 0, label=label)
                

                legend_marker = mlines.Line2D(
                    [], [],
                    color=cols_[i],
                    marker='*',
                    linestyle='None',
                    linewidth=2,
                    markersize=15    # <-- legend marker size here
                )
                handles.append(legend_marker)
                labels_.append(label)

                i+=1

            plt.xlabel('East [m]')
            plt.ylabel('North [m]')
            plt.title('Trajectory with Overlapping Segments Highlighted')
            #plt.legend()

            axis_map.legend(
                handles=handles,
                labels=labels_
            )


            plt.legend( loc='upper center', bbox_to_anchor=bbox_to_anchor, ncol = ncol, fancybox=True, shadow=True) #title="Method",
            plt.axis('equal')
            plt.grid(False)
            plt.draw()

            # axis_map.legend()
            plt.draw()

            

            # plt.show()

        #-----------------------------------------------------------------------
        if plot: # and False:
            fig_3d = plt.figure(figsize=(12, 8))
            ax3d = fig_3d.add_subplot(111, projection='3d')

            # Plot full trajectory in gray
            ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='gray', alpha=0.3, label='Full Trajectory')

        
        self.segment_passes = []  # Stores tuples: (forward_pass_idxs, backward_pass_idxs)

        for idx, segment in enumerate(overlap_segments):
            segment = sorted(segment)
            # Use clustering in index space to separate passes
            diffs = np.diff(segment)
            split_idx = np.argmax(diffs) + 1  # split at largest gap

            forward_pass = segment[:split_idx]
            backward_pass = segment[split_idx:]

            self.segment_passes.append((forward_pass, backward_pass))
            print(f"Segment with {len(segment)} points -> Forward: {len(forward_pass)}, Backward: {len(backward_pass)}")

            pts1 = traj[forward_pass]
            pts2 = traj[backward_pass]

            #pts1 = pts1[::10]
            #pts2 = pts2[::10]

            if plot: # and False:
                color1 = cmap((2 * idx) % 10)
                color2 = cmap((2 * idx + 1) % 10)

                ax3d.scatter(pts1[:, 0], pts1[:, 1], pts1[:, 2], 
                        color=color1, marker='o', alpha=0.7, s=3,
                        label=f'Segment {idx+1} - Forward Pass')

                ax3d.scatter(pts2[:, 0], pts2[:, 1], pts2[:, 2], 
                                color=color2, marker='^', alpha=0.7, s=3,
                                label=f'Segment {idx+1} - Backward Pass')

                ax3d.set_xlabel("East [m]")
                ax3d.set_ylabel("North [m]")
                ax3d.set_zlabel("Height [m]")
                ax3d.set_title(f"3D Trajectory with Overlapping Segments ({label})")
                ax3d.grid(True)

        if plot: # and False:
            def set_axes_equal(ax):
                x_limits = ax.get_xlim3d()
                y_limits = ax.get_ylim3d()
                z_limits = ax.get_zlim3d()

                x_range = abs(x_limits[1] - x_limits[0])
                y_range = abs(y_limits[1] - y_limits[0])
                z_range = abs(z_limits[1] - z_limits[0])

                x_middle = np.mean(x_limits)
                y_middle = np.mean(y_limits)
                z_middle = np.mean(z_limits)

                radius = 0.5 * max(x_range, y_range, z_range)
                ax.set_xlim3d([x_middle - radius, x_middle + radius])
                ax.set_ylim3d([y_middle - radius, y_middle + radius])
                ax.set_zlim3d([z_middle - radius, z_middle + radius])

            set_axes_equal(ax3d)
            ax3d.legend(loc='best') #title="Method",



            plt.draw()

        #--------------------------------------------------------------------------------
        # Step 4: Compare Z-values using NN match between forward/backward pass
        num_plots = len(self.segment_passes)
        if num_plots == 0:
            print("No valid segments for z-axes error plotting.")
            return 0

        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots), sharex=False)
        
        if num_plots == 1:
            axes = [axes]  # ensure it's iterable

        avg_z = 0
        number = 0
        all_z_diffs = []
        first_time = True
        colors_ = ['tab:orange','tab:blue', 'red','blue','green',]
        for idx, ((forward_pass, backward_pass), ax) in enumerate(zip(self.segment_passes, axes)):

            if len(forward_pass) < 5 or len(backward_pass) < 5:
                ax.set_title(f"Segment {idx+1}: Skipped (too small)")
                continue

            # Get 3D points
            fwd_pts = traj[forward_pass]
            bwd_pts = traj[backward_pass]

            print("Traveleld distance fwd_pts:", self.traveled_distance(fwd_pts))
            print("Traveleld distance bwd_pts:", self.traveled_distance(bwd_pts))

            # KD-Tree to find nearest backward point for each forward point
            # bwd_tree = cKDTree(bwd_pts[:, :3])
            # distances, indices = bwd_tree.query(fwd_pts[:, :3], distance_upper_bound=distance_upper_bound)
            # KD-Tree on XY only
            bwd_tree = cKDTree(bwd_pts[:, :2])  # only (x, y)
            distances, indices = bwd_tree.query(fwd_pts[:, :2], distance_upper_bound=distance_upper_bound)


            valid = distances != np.inf
            fwd_valid = fwd_pts[valid]
            bwd_valid = bwd_pts[indices[valid]]

            z1 = fwd_valid[:, 2]
            z2 = bwd_valid[:, 2]
            z_diff = z2 - z1
            abs_z_diff = np.abs(z_diff)
            rmse = np.sqrt(np.mean(z_diff ** 2))
            mean = np.mean(np.abs(z_diff))
            all_z_diffs.append(np.abs(z_diff))
            avg_z += mean
            number+= 1
            print(f"Segment {idx+1}: Matched {len(z_diff)} pairs — z-RMSE = {rmse:.2f} m, mean = {mean:.2f} m")

            # ax.plot(z1, label='Forward Pass $z$-axes',  color='tab:blue')
            ax.scatter(range(len(z1)), z1, label='Forward Pass $z$-axes', color='tab:blue', s=1)
            ax.scatter(range(len(z2)), z2, label='Backward Pass $z$-axes', color='tab:orange', s=1)
            ax.fill_between(range(len(z_diff)), z1, z2, color='gray', alpha=0.3, label='$z$-axes error')
            ax.set_ylabel("$z$-axes (m)")
            #ax.set_xlabel("Points")
            ax.set_title(f"Segment {idx+1} — Mean Δz-axes: {mean:.2f} m, RMSE: {rmse:.2f} m")
            ax.grid(True)

            f, a = plt.subplots()
            # a.plot(z1, label='Forward Pass $z$-axes', linestyle='--', color='tab:blue')
            # a.plot(z2, label='Backward Pass $z$-axes', linestyle='-.', color='tab:orange')
            # a.fill_between(range(len(z_diff)), z1, z2, color='gray', alpha=0.3, label='$z$-axes error')
            a.plot(abs_z_diff, label=' abs $z$-axes error', linestyle='--', color='tab:blue')
            a.set_ylabel("$z$-axes (m)")
            a.set_xlabel("Points")
            #a.set_title(f"Segment {idx+1} — Mean Δz-axes: {mean:.2f} m, RMSE: {rmse:.2f} m")
            a.grid(True)
            if first_time:
                first_time = False
                a.legend(loc='upper center', bbox_to_anchor=bbox_to_anchor,
                ncol = ncol, fancybox=True, shadow=True)

            
            f2, a2 = plt.subplots()
            a2.plot(z_diff, label='$z$-axes error', color=colors_[idx])
            a2.set_ylabel("$z$-axes (m)")
            a2.set_xlabel("Points")
            a2.set_title(f"Segment {idx+1} — Mean Δz-axes: {mean:.2f} m, RMSE: {rmse:.2f} m")
            a2.grid(True)
            # if first_time:
            # first_time = False
            a2.legend(loc='upper center', bbox_to_anchor=bbox_to_anchor,
                ncol = ncol, fancybox=True, shadow=True)

            
            
            

        
        axes[1].legend(loc='upper center', bbox_to_anchor=bbox_to_anchor,
          ncol = ncol, fancybox=True, shadow=True)

        fig.suptitle(f"$z$-axes Comparison Across Overlapping Segments ({label})")
        #fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.draw()

        all_z_diffs_concatenated = np.concatenate(all_z_diffs)
        return avg_z/number, all_z_diffs_concatenated

def overlap_error(est_xyz, label, segment_passes,  plot = False):    
    # if label in ['LI', 'LI-VUX']:
    #     return 0, [0]
    
    #segment_passes = []  # Stores tuples: (forward_pass_idxs, backward_pass_idxs)

    axes = segment_passes
    num_plots = len(segment_passes)
    if num_plots == 0:
        print("No valid segments for Z error plotting.")
        return 0, [0]

    max_index = est_xyz.shape[0] 
    if plot:
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots), sharex=False)
        if num_plots == 1:
            axes = [axes]  # ensure it's iterable

        fig2, axes2 = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots), sharex=False)
        if num_plots == 1:
            axes2 = [axes2]  # ensure it's iterable

    avg_z = 0
    number = 0
    all_z_diffs = []
    for idx, ((forward_pass, backward_pass), ax) in enumerate(zip(segment_passes, axes)):

        if plot:
            if len(forward_pass) < 5 or len(backward_pass) < 5:
                ax.set_title(f"Segment {idx+1}: Skipped (too small)")
                continue
        
        if max(forward_pass) < len(est_xyz) and max(backward_pass) < len(est_xyz):
            fwd_pts = est_xyz[forward_pass]
            bwd_pts = est_xyz[backward_pass]
        else:
            print("One or more index lists contain out-of-bounds indices.")
            return 0, [0]


        # Get 3D points
        # fwd_pts = est_xyz[forward_pass]
        # bwd_pts = est_xyz[backward_pass]

        # KD-Tree to find nearest backward point for each forward point
        # bwd_tree = cKDTree(bwd_pts[:, :3])
        # distances, indices = bwd_tree.query(fwd_pts[:, :3], distance_upper_bound=distance_upper_bound) #1.0
        # KD-Tree on XY only
        bwd_tree = cKDTree(bwd_pts[:, :2])  # only (x, y)
        distances, indices = bwd_tree.query(fwd_pts[:, :2], distance_upper_bound=distance_upper_bound)
        
        valid = (distances != np.inf) 
        fwd_valid = fwd_pts[valid]
        bwd_valid = bwd_pts[indices[valid]]

        z1 = fwd_valid[:, 2]
        z2 = bwd_valid[:, 2]
        z_diff = z2 - z1

        std = np.std(np.abs(z_diff))
        rmse = np.sqrt(np.mean(z_diff ** 2))
        mean = np.mean(np.abs(z_diff))
        all_z_diffs.append(np.abs(z_diff))
        avg_z += mean
        number+= 1
        print(f"Segment {idx+1}: Matched {len(z_diff)} pairs — z-RMSE = {rmse:.2f} m, mean = {mean:.2f} m")

        if plot:
            # ax.plot(z1, label='Forward Pass $z$-axes',linestyle='--', color='tab:blue')
            # ax.plot(z2, label='Backward Pass $z$-axes', linestyle='-.', color='tab:orange')
            ax.scatter(range(len(z1)), z1, label='Forward Pass $z$-axes', color='tab:blue', s=1)
            ax.scatter(range(len(z2)), z2, label='Backward Pass $z$-axes', color='tab:orange', s=1)
            
            ax.fill_between(range(len(z_diff)), z1, z2, color='gray', alpha=0.3, label='$z$-axes error')
            ax.set_ylabel("$z$-axes [m]")
            ax.set_xlabel("Points")
            ax.set_title(f"Segment {idx+1} — Mean Δz-axes: {mean:.2f} m, RMSE: {rmse:.2f} m, std: {std:.2f} m")
            ax.grid(True)

            axes2[idx].plot(np.abs(z_diff),label='$z$-axes error')
            axes2[idx].set_ylabel("$z$-axes [m]")
            axes2[idx].set_xlabel("Points")
            axes2[idx].set_title(f"Segment {idx+1} — Mean Δz-axes: {mean:.2f} m, RMSE: {rmse:.2f} m, std: {std:.2f} m")
            axes2[idx].grid(True)
    
    if plot:
        # axes[1].legend(loc='upper center', bbox_to_anchor=bbox_to_anchor,
        #       ncol = ncol, fancybox=True, shadow=True)
        
        # axes2[1].legend(loc='upper center', bbox_to_anchor=bbox_to_anchor,
        #       ncol = ncol, fancybox=True, shadow=True)

        fig.suptitle(f"z-axes Overlapping Segments ({label})")
        fig2.suptitle(f"z-axes Overlapping Segments ({label})")
        plt.draw()

    all_z_diffs_concatenated = np.concatenate(all_z_diffs)
    return avg_z/number, all_z_diffs_concatenated



path_gt =  "/home/eugeniu/z_tighly_coupled/ref/MLS.txt" #prev
path_gt =  "/home/eugeniu/zz_zx_final/ref/MLS.txt"      #new one

methods = {
    'Reference trajectory' : '/home/eugeniu/z_tighly_coupled/ref',

    'GNSS-INS'             : '/home/eugeniu/z_tighly_coupled/0',
    'LI'                    : '/home/eugeniu/z_tighly_coupled/1',
    'LI-VUX'                : '/home/eugeniu/z_tighly_coupled/2',
    'LI-VUX-ALS(l-coupled)' : '/home/eugeniu/z_tighly_coupled/3',
    'LI-VUX-ALS(t-coupled)' : '/home/eugeniu/z_tighly_coupled/4',
    ## # # do not use this at all'LI-VUX-(raw)GNSS'      : '/home/eugeniu/z_tighly_coupled/5',
    'LI-VUX-GNSS'      : '/home/eugeniu/z_tighly_coupled/6',
    'LI-VUX-sparse-ALS(l-coupled)' : '/home/eugeniu/z_tighly_coupled/7',
    'LI-VUX-sparse-ALS(t-coupled)' : '/home/eugeniu/z_tighly_coupled/8',
}

#-robust comes from plane uncertanties 

methods = {
    'Reference trajectory' : '/home/eugeniu/z_tighly_coupled/ref',
    # 'test_now'                         : '/home/eugeniu/z_tighly_coupled/_last', #/home/eugeniu/z_tighly_coupled',
    # 'test'                         : '/home/eugeniu/z_tighly_coupled/test',

    'GNSS-INS'             : '/home/eugeniu/z_tighly_coupled/0',

    #the methods with fixed covariance as before 
    # '*-LI'                    : '/home/eugeniu/z_tighly_coupled/1',
    # 'LI'                        : '/home/eugeniu/z_tighly_coupled/1.1',    #robust
    
    
    # '*-LI-VUX'                : '/home/eugeniu/z_tighly_coupled/2',
    # 'LI-VUX'                    : '/home/eugeniu/z_tighly_coupled/2.1', #robust
    
    

    ## # # do not use this at all 'LI-VUX-(raw)GNSS'      : '/home/eugeniu/z_tighly_coupled/5',
    'LI-VUX + GNSS'      : '/home/eugeniu/z_tighly_coupled/6',

    
    'LI-VUX + S-ALS (l-coupled)' : '/home/eugeniu/z_tighly_coupled/7',
    'LI-VUX + S-ALS (t-coupled)' : '/home/eugeniu/z_tighly_coupled/8',

    'LI-VUX + D-ALS (l-coupled)' : '/home/eugeniu/z_tighly_coupled/3',

    # '*-LI-VUX + D-ALS (t-coupled)': '/home/eugeniu/z_tighly_coupled/_last',

    'LI-VUX + D-ALS (t-coupled)' : '/home/eugeniu/z_tighly_coupled/4',
}

methods_data = {
    'GNSS-INS'             : ['#2ca02c','A'],

    'LI'                    : ['#1f77b4','B'],    #robust
    'LI-VUX'                 : ['#ff7f0e','C'],#robust

    
    # # # 'LI-VUX-(raw)GNSS'      : ['#7f7f7f','D'],
    'LI-VUX + GNSS'      : ['#9467bd','D'],

    'LI-VUX + S-ALS (l-coupled)' : ['#bcbd22','E'],
    'LI-VUX + S-ALS (t-coupled)' : ['#17becf','F'],

    'LI-VUX + D-ALS (l-coupled)' : ['#8c564b','G'],
    'LI-VUX + D-ALS (t-coupled)' : ['#e377c2','H'],

    'Reference trajectory' : ['#d62728','L'],

    '*-LI' :  ["#1f77b4",'M'],# ['#EFD700','M'],
    '*-LI-VUX' : ["#ff7f0e",'N'], #['#04f810','N'],

    'test' : ["#648099",'Z'],
    'test2' : ["#26391B",'Z'],
    'test_now' : ["#a86b40",'S'],
    '*-LI-VUX + D-ALS (t-coupled)' : ["#e377c2",'S'],
}


colors = ['tab:brown', 'tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange', 'cyan', 'lime','orange','gray']
colors2 = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#7f7f7f',  # Gray
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#bcbd22',  # Yellow-green
    '#17becf'   # Cyan
    '#d62728',  # Red
]
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', '-', '--', ':',]
lab = ['A','B','C','D','E','F','G','H','K','X']
lt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


methods = {
    #'Reference trajectory' : '/home/eugeniu/z_tighly_coupled/ref',
    'Reference trajectory' :  "/home/eugeniu/zz_zx_final/ref",
    #  'rko'             : '/home/eugeniu/zz_zx_final/rko',
    # 'point-lio'             : '/home/eugeniu/zz_zx_final/point-lio',

    # 'fast_lio2'             : '/home/eugeniu/zz_zx_final/fast_lio2', 
    # 'dlo'             : '/home/eugeniu/zz_zx_final/dlo', 
    # 'dlo-5'             : '/home/eugeniu/zz_zx_final/dlo.5', 

    # 'a0'             : '/home/eugeniu/zz_zx_final/a0',    #our implementation with no robust no cov, with fixed Fx and Fw 
    # 'a1'             : '/home/eugeniu/zz_zx_final/a1',    #prev a0 with robust kernel only 
    # 'a2'             : '/home/eugeniu/zz_zx_final/a2',    #our best with fixed prediction and robust & adaptive 
    # 'a2_bar'             : '/home/eugeniu/zz_zx_final/a2_bar',    #a2 with update using original iekf code 
    # 'a3'             : '/home/eugeniu/zz_zx_final/a3',    #a2 with gravity
    'a4'             : '/home/eugeniu/zz_zx_final/a4',    #similar to a2 but with final best weighting 
    
    # 'b0'             : '/home/eugeniu/zz_zx_final/b0',    #a0 with vux
    # 'b1'             : '/home/eugeniu/zz_zx_final/b1',    #a1 with vux 
    # 'b2'             : '/home/eugeniu/zz_zx_final/b2',    #a2 with added vux
    # 'b4'             : '/home/eugeniu/zz_zx_final/b4',    #similar to b2 but with final best weighting 

    'HeliALS'             : '/home/eugeniu/zz_zx_final/HeliALS',  # a4 LI + HeliALS
    'Sparse ALS'             : '/home/eugeniu/zz_zx_final/s_ALS', # a4 LI + s-ALS

    'LI-VUX + S-ALS (l-coupled)' : '/home/eugeniu/z_tighly_coupled/7',
    # 'LI-VUX + S-ALS (t-coupled)' : '/home/eugeniu/z_tighly_coupled/8',


    'GNSS_'             : '/home/eugeniu/zz_zx_final/GNSS_INS',   # a4 + GNSS-INS   9 sigma of gnss-ins
    'GNSS_INS-alone'             : '/home/eugeniu/zz_zx_final/GNSS_INS-alone',
    
    

    'GNSS_s-ALS'             : '/home/eugeniu/zz_zx_final/GNSS_s-ALS',     #  test gnss + li + sALS
}

methods_data = {
    '0_LI' :  ["#1f77b4",'M'],
    '1_LI_robust_adaptive'      : ['#9467bd','D'],
    '2_LI_robust_adaptive_g'             : ['#2ca02c','A'],
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

    # # # 'LI-VUX-(raw)GNSS'      : ['#7f7f7f','D'],

    


    

    'b0' : ["#ff7f0e",'N'], #['#04f810','N'],

    'g_p2p_p2pl' : ["#26391B",'Z'],

    'dlo-5' : ["#26391B",'Z'],
    'test_now' : ["#a86b40",'S'],
    'b2' : ["#433123",'S'],
     'dlo' : ["#433123",'S'],
    'a2_bar' : ["#e377c2",'S'],
}

obj_gt = TrajectoryReader(path_gt, path_gt)


if False:
    est_xyz = obj_gt.traj_model.positions_xyz

    xyz_1 = TrajectoryReader(path_gt, '/home/eugeniu/z_tighly_coupled/1.1/MLS.txt', 'LI', False).traj_model.positions_xyz
    xyz_2 = TrajectoryReader(path_gt, '/home/eugeniu/z_tighly_coupled/2.1/MLS.txt', 'LI-VUX', False).traj_model.positions_xyz

    before_xyz_1 = TrajectoryReader(path_gt, '/home/eugeniu/z_tighly_coupled/1/MLS.txt', '*-LI', False).traj_model.positions_xyz
    before_xyz_2 = TrajectoryReader(path_gt, '/home/eugeniu/z_tighly_coupled/2/MLS.txt', '*-LI-VUX', False).traj_model.positions_xyz

    other_traj = [ before_xyz_1, xyz_1, before_xyz_2, xyz_2]
    other_labels = ['*-LI', 'LI',  '*-LI-VUX',  'LI-VUX']



    # before_xyz_3 = TrajectoryReader(path_gt, '/home/eugeniu/z_tighly_coupled/_last/MLS.txt', '*-LI-VUX + D-ALS (t-coupled)', False).traj_model.positions_xyz

    other_traj = [ before_xyz_1,  before_xyz_2,] #before_xyz_3
    other_labels = ['*-LI', '*-LI-VUX', '*-LI-VUX + D-ALS (t-coupled)']

    f_map,axis_map = plot_trajectory(est_xyz, other_traj=other_traj, other_labels = other_labels)
    plt.show()


obj_gt.overlap_error(obj_gt.traj_model.positions_xyz, "Reference trajectory", plot = True)
all_traj = [obj_gt.traj_gt]
# 3D Plot
fig_3d = plt.figure(figsize=(10, 7))
ax_3d = fig_3d.add_subplot(111, projection='3d')
ax_3d.set_title("3D Trajectories")
ax_3d.set_xlabel("East [m]")
ax_3d.set_ylabel("North [m]")
ax_3d.set_zlabel("Height [m]")


# 2D Plot (XY Plane)
# fig_2d = plt.figure(figsize=(10, 7))
# ax_2d = fig_2d.add_subplot(111)
# ax_2d.set_title("XY Plane Trajectories")
# ax_2d.set_xlabel("East [m]")
# ax_2d.set_ylabel("North [m]")

data_ape_t = {}
data_ape_r = {}
data_rpe_t = {}
data_rpe_r = {}
data_z_overlap = {}
data_z_overlap2 = {}

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


def test(points):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial import KDTree
    from scipy.interpolate import UnivariateSpline
    from scipy.spatial.distance import cdist
    import networkx as nx

    def find_overlap_segments(points, search_radius=1.0, min_time_diff=100, distance_upper_bound=1.0):
        """
        Find overlapping segments in the trajectory where the robot revisits similar areas.
        """
        n_points = len(points)
        
        # Build KDTree for efficient spatial queries
        tree = KDTree(points[:, :2])  # Only use x,y for spatial queries
        
        # Find nearby points using KDTree
        nearby_pairs = []
        for i in range(n_points):
            # Query all points within search_radius
            indices = tree.query_ball_point(points[i, :2], search_radius)
            
            for j in indices:
                if i != j and abs(i - j) > min_time_diff:  # Ensure temporal separation
                    dist = np.linalg.norm(points[i, :2] - points[j, :2])
                    if dist <= distance_upper_bound:
                        nearby_pairs.append((min(i, j), max(i, j), dist))
        
        # Remove duplicates and sort
        nearby_pairs = list(set(nearby_pairs))
        nearby_pairs.sort()
        
        return nearby_pairs

    def group_segments(nearby_pairs, min_segment_length=5):
        """
        Group nearby pairs into continuous segments.
        """
        if not nearby_pairs:
            return []
        
        # Create a graph to find connected components
        G = nx.Graph()
        for i, j, dist in nearby_pairs:
            G.add_edge(i, j)
        
        segments = []
        for component in nx.connected_components(G):
            nodes = sorted(component)
            if len(nodes) >= min_segment_length:
                # Split into forward and backward passes
                segments.append(nodes)
        
        return segments

    def fit_spline_and_compute_errors(points, segments):
        """
        Fit splines to segments and compute z-axis errors between forward and backward passes.
        """
        results = []
        
        for seg_idx, segment in enumerate(segments):
            if len(segment) < 10:  # Need enough points for spline fitting
                continue
                
            # Extract segment points
            seg_points = points[segment]
            
            # Project points to 2D (x,y) and parameterize by arc length
            xy_points = seg_points[:, :2]
            
            # Compute arc length parameter
            arc_length = np.zeros(len(xy_points))
            for i in range(1, len(xy_points)):
                arc_length[i] = arc_length[i-1] + np.linalg.norm(xy_points[i] - xy_points[i-1])
            
            # Normalize arc length to [0, 1]
            t = arc_length / arc_length[-1] if arc_length[-1] > 0 else np.linspace(0, 1, len(arc_length))
            
            try:
                # Fit spline for z-coordinate
                # spline = UnivariateSpline(t, seg_points[:, 2]) #, s=len(seg_points)*0.1
                # Much tighter fit - reduce smoothing factor
                spline = UnivariateSpline(t, seg_points[:, 2], s=len(seg_points)*0.001)  # 0.1% instead of 10%
                # or even tighter:
                # spline = UnivariateSpline(t, seg_points[:, 2], s=0)  # Interpolating spline - goes through all points

                # Evaluate spline at all parameter points
                z_spline = spline(t)
                
                # Compute errors (difference between actual and spline)
                errors = seg_points[:, 2] - z_spline
                
                # For overlapping analysis, we need to identify forward/backward passes
                # This is simplified - in practice you'd need more sophisticated logic
                # to separate forward and backward passes in the same segment
                
                results.append({
                    'segment_id': seg_idx,
                    'indices': segment,
                    'points': seg_points,
                    'spline': spline,
                    'arc_length': arc_length,
                    'z_spline': z_spline,
                    'errors': errors,
                    'mean_error': np.mean(np.abs(errors)),
                    'std_error': np.std(errors)
                })
                
            except Exception as e:
                print(f"Error fitting spline for segment {seg_idx}: {e}")
                continue
        
        return results

    def analyze_forward_backward(points, search_radius=1.0, min_time_diff=100, distance_upper_bound=1.0):
        """
        Main analysis function for forward-backward z-axis error analysis.
        """
        print("Finding overlap segments...")
        nearby_pairs = find_overlap_segments(points, search_radius, min_time_diff, distance_upper_bound)
        print(f"Found {len(nearby_pairs)} nearby point pairs")
        
        print("Grouping into segments...")
        segments = group_segments(nearby_pairs)
        print(f"Found {len(segments)} segments")
        
        print("Fitting splines and computing errors...")
        results = fit_spline_and_compute_errors(points, segments)
        
        return results

    def plot_results(points, results):
        """
        Create comprehensive plots of the analysis results.
        """
        fig = plt.figure(figsize=(20, 15))
        
        # Plot 1: Original trajectory with overlap segments highlighted
        # fig, ax1 = plt.subplots(figsize=(10, 8),projection='3d')
        # ax1 = fig.add_subplot(3, 2, 1, projection='3d')
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.plot(points[:, 0], points[:, 1], points[:, 2], 'b-', alpha=0.3, label='Full Trajectory')
        
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, result in enumerate(results):
            color = colors[i % len(colors)]
            seg_points = result['points']
            ax1.plot(seg_points[:, 0], seg_points[:, 1], seg_points[:, 2], 
                    color=color, linewidth=2, label=f'Segment {result["segment_id"]}')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Trajectory with Overlap Segments')
        ax1.legend()

        x_limits = [np.min(points[:, 0]), np.max(points[:, 0])]
        y_limits = [np.min(points[:, 1]), np.max(points[:, 1])]
        z_limits = [np.min(points[:, 2]), np.max(points[:, 2])]

        # Compute the global range and center
        x_range = x_limits[1] - x_limits[0]
        y_range = y_limits[1] - y_limits[0]
        z_range = z_limits[1] - z_limits[0]
        max_range = max(x_range, y_range, z_range)
        x_center = np.mean(x_limits)
        y_center = np.mean(y_limits)
        z_center = np.mean(z_limits)
        # Set equal limits for each axis
        ax1.set_xlim(x_center - max_range/2, x_center + max_range/2)
        ax1.set_ylim(y_center - max_range/2, y_center + max_range/2)
        ax1.set_zlim(z_center - max_range/2, z_center + max_range/2)
        
        # Plot 2: Z-coordinate along trajectory
        fig, ax2 = plt.subplots(figsize=(10, 8))
        # ax2 = fig.add_subplot(3, 2, 2)
        ax2.plot(range(len(points)), points[:, 2], 'b-', alpha=0.5, label='Original Z')
        
        for i, result in enumerate(results):
            color = colors[i % len(colors)]
            indices = result['indices']
            ax2.plot(indices, points[indices, 2], 'o', color=color, linewidth=2, 
                    label=f'Segment {result["segment_id"]}')
        
        ax2.set_xlabel('Point Index')
        ax2.set_ylabel('Z coordinate')
        ax2.set_title('Z-coordinate along Trajectory')
        ax2.legend()
        
        # Plot 3: Spline fits for each segment
        # ax3 = fig.add_subplot(3, 2, 3)
        fig, ax3 = plt.subplots(figsize=(10, 8))
        for i, result in enumerate(results):
            color = colors[i % len(colors)]
            arc_length = result['arc_length']
            actual_z = result['points'][:, 2]
            spline_z = result['z_spline']
            
            ax3.plot(arc_length, actual_z, color=color, linestyle='-', alpha=0.7, 
                    label=f'Seg {result["segment_id"]} - Actual')
            ax3.plot(arc_length, spline_z, color=color, linestyle='--', linewidth=2,
                    label=f'Seg {result["segment_id"]} - Spline')
        
        ax3.set_xlabel('Arc Length')
        ax3.set_ylabel('Z coordinate')
        ax3.set_title('Spline Fits for Each Segment')
        ax3.legend()
        
        # Plot 4: Errors for each segment
        fig, ax4 = plt.subplots(figsize=(10, 8))
        # ax4 = fig.add_subplot(3, 2, 4)
        for i, result in enumerate(results):
            color = colors[i % len(colors)]
            arc_length = result['arc_length']
            errors = result['errors']
            
            ax4.plot(arc_length, errors, color=color, linewidth=2,
                    label=f'Seg {result["segment_id"]} (mean: {result["mean_error"]:.4f})')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax4.set_xlabel('Arc Length')
        ax4.set_ylabel('Z Error (Actual - Spline)')
        ax4.set_title('Z-axis Errors for Each Segment')
        ax4.legend()
        
        # Plot 6: Overall statistics
        # ax6 = fig.add_subplot(3, 2, 6)
        fig, ax6 = plt.subplots(figsize=(10, 8))
        if results:
            all_errors = np.concatenate([result['errors'] for result in results])
            
            ax6.hist(all_errors, bins=30, alpha=0.7, edgecolor='black')
            ax6.axvline(x=np.mean(all_errors), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(all_errors):.4f}')
            ax6.axvline(x=np.mean(all_errors) + np.std(all_errors), color='orange', 
                    linestyle='--', alpha=0.7, label=f'±1 STD')
            ax6.axvline(x=np.mean(all_errors) - np.std(all_errors), color='orange', 
                    linestyle='--', alpha=0.7)
            
            ax6.set_xlabel('Z Error')
            ax6.set_ylabel('Frequency')
            ax6.set_title('Distribution of All Z-axis Errors')
            ax6.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        if results:
            all_errors = np.concatenate([result['errors'] for result in results])
            print(f"Overall mean absolute error: {np.mean(np.abs(all_errors)):.6f}")
            print(f"Overall error standard deviation: {np.std(all_errors):.6f}")
            print(f"Overall error range: [{np.min(all_errors):.6f}, {np.max(all_errors):.6f}]")
            print(f"Number of segments analyzed: {len(results)}")
            print(f"Total points in analysis: {len(all_errors)}")
            
            for result in results:
                print(f"\nSegment {result['segment_id']}:")
                print(f"  Points: {len(result['indices'])}")
                print(f"  Mean absolute error: {result['mean_error']:.6f}")
                print(f"  Error std: {result['std_error']:.6f}")
        else:
            print("No segments found for analysis.")
    
    # Analyze the trajectory
    results = analyze_forward_backward(
        points, 
        search_radius=3.0, 
        min_time_diff=100, 
        distance_upper_bound=1.0
    )
    
    # Plot results
    plot_results(points, results)

# test(obj_gt.traj_model.positions_xyz)

for idx, (label, path) in enumerate(methods.items()):
    #for label, path in methods.items():
    print('\n Model ',label)
    path_model=path+"/MLS.txt"
    obj = TrajectoryReader(path_gt, path_model, label)

    obj.APE_translation()
    obj.APE_rotation()
    obj.RPE_translation()
    obj.RPE_rotation()

    
    if False:    
        row = "" #table_data[label]
        row += f"& \\multicolumn{{1}}{{c|}}{{ {round(obj.ape_statistics_t['mean'],3)} }} \n"
        row += f"& \\multicolumn{{1}}{{c|}}{{ {round(obj.ape_statistics_t['median'],3)} }} \n"
        row += f"& \\multicolumn{{1}}{{c|}}{{ {round(obj.ape_statistics_t['rmse'],3)} }} \n"
        row += f"& \\multicolumn{{1}}{{c|}}{{ {round(obj.ape_statistics_t['std'],3)} }} \n"
        row += f"& \\multicolumn{{1}}{{c|}}{{ {round(obj.rpe_statistics_t['mean'],3)} }} \n"
        row += f"& \\multicolumn{{1}}{{c|}}{{ {round(obj.rpe_statistics_t['median'],3)} }} \n"
        row += f"& \\multicolumn{{1}}{{c|}}{{ {round(obj.rpe_statistics_t['rmse'],3)} }} \n"
        row += f"& {{{round(obj.rpe_statistics_t['std'],3)}}}" 
        print('\n\n',label,'\n',row,'\n\n')


    #obj.plot_data()

    all_traj.append(obj.traj_model)

    data_ape_t[label] = obj.ape_t_error_vectors
    data_ape_r[label] = obj.ape_r_error_vectors
    data_rpe_t[label] = obj.rpe_t_error_vectors
    data_rpe_r[label] = obj.rpe_r_error_vectors

    positions = obj.traj_model.positions_xyz  # N x 3 numpy array

    ax_3d.plot(
        positions[:, 0], positions[:, 1], positions[:, 2],
        label=label, color=colors[idx % len(colors)], linestyle=linestyles[idx % len(linestyles)]
    )

    # ax_2d.plot(
    #     positions[:, 0], positions[:, 1],
    #     label=label, color=colors[idx % len(colors)], linestyle=linestyles[idx % len(linestyles)]
    # )

    z_overlap_error, all_z_diffs = overlap_error(obj.traj_model.positions_xyz, label, obj_gt.segment_passes,  plot = False)
    data_z_overlap[label] = z_overlap_error
    data_z_overlap2[label] = all_z_diffs




#ax_3d.legend()
#ax_2d.legend()
ax_3d.legend(loc='upper center', bbox_to_anchor=bbox_to_anchor, #title="Method",
          ncol = ncol, fancybox=True, shadow=True)
# ax_2d.legend(title="Method",loc='upper center', bbox_to_anchor=bbox_to_anchor,
#   ncol = ncol, fancybox=True, shadow=True)

ax_3d.grid(True)
# ax_2d.grid(True)
plt.draw()

# plt.show()
# exit()

def plot_box(data, metric = '',  show_legend = True, show_cumulative = False, add_arrows = False):
    print('plot_box for ',metric)
    labels = list(data.keys())

    #labels_local = lab[0:len(labels)]
    lt_local = lt[0:len(labels)]

    plt.figure(figsize=(10, 6))
    

    values = [data[label] for label in labels]

    box = plt.boxplot(values, patch_artist=True, showmeans=True, meanline=True, showfliers=False, notch=False) 
    ind = 0
    legend_handles = []
    labels_local = []
    colors_ = [methods_data[label][0] for label in labels]

    for patch, color, label in zip(box['boxes'], colors_, labels):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
        patch_legend = mpatches.Patch(
            facecolor=color,
            # edgecolor='black',
            label=methods_data[label][1] + " : " + label
        )
        
        labels_local.append(methods_data[label][1])
        
        if '*' in label:
            patch.set_hatch('\\')  # or 'xx', '\\',  //etc.
            patch.set_edgecolor('white') 
            patch_legend = mpatches.Patch(
                facecolor=color,
                edgecolor='white',   
                hatch='\\',          
                label=methods_data[label][1] + " : " + label
            )
        
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
    #plt.tight_layout()
    #plt.xticks(rotation=90)
    #plt.xticks([])
    plt.xticks(lt_local, labels_local, fontsize = font) 
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
            #for i, label in enumerate(data):
            values = np.sort(data[label])
            cdf = np.linspace(0, 1, len(values))
            plt.plot(values, cdf, label=label, color = color, linestyle = linestyles[i])

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

def plot_z_overlap(data, metric_name=''):
    plt.figure(figsize=(10, 6))

    labels = list(data.keys())
    x = np.arange(len(labels))
    width = 0.8 / len(data)
    
    for i, (label, value) in enumerate(data.items()):
        plt.bar(i , value, width, label=label)

        print(label, ':', value)

    plt.ylabel(metric_name)
    #plt.xticks(x, labels)
    plt.xticks([])
    plt.legend( loc='upper center', bbox_to_anchor=bbox_to_anchor, #title="Method",
          ncol = ncol, fancybox=True, shadow=True)
    #plt.xticks(rotation=90)
    #plt.tight_layout()
    plt.grid(True)
        
    plt.draw()


plot_box(data_ape_t, 'ATE translation (m)', show_legend = True, add_arrows = True)
#plot_box(data_ape_r, 'ATE rotation (deg)')
plot_box(data_rpe_t, 'RTE translation (%)', show_legend = True)
#plot_box(data_rpe_r, 'RTE rotation (deg)')

# label = "Reference trajectory"
# z_overlap_error, all_z_diffs = overlap_error(obj_gt.traj_model.positions_xyz, label, obj_gt.segment_passes,  plot = False)
# data_z_overlap[label] = z_overlap_error
# data_z_overlap2[label] = all_z_diffs

# data_z_overlap2.pop('LI')
# data_z_overlap2.pop('LI-VUX')

plot_box(data_z_overlap2, 'Overlap $z$-axes error (m)',  show_legend = True, show_cumulative = True)

#plot_z_overlap(data_z_overlap, 'Mean z error overlap')


plt.show()