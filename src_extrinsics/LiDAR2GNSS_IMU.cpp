

#include "DataHandler_vux.hpp"

#include "IMU.hpp"
#include "GNSS.hpp"

#ifndef USE_EKF
#include "PoseGraph.hpp"
#endif

#ifdef USE_ALS
#include "ALS.hpp"
#endif

#include <chrono>

#include "Vux_reader.hpp"
#include <GeographicLib/UTMUPS.hpp>
#include <liblas/liblas.hpp>

#include <visualization_msgs/Marker.h>
#include <pcl/registration/icp.h>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

volatile bool flg_exit = false;
void SigHandle(int sig)
{
    ROS_WARN("Caught signal %d, stopping...", sig);
    flg_exit = true; // Set the flag to stop the loop
}

void publishPose(const Sophus::SE3 &_pose, const double &msg_time)
{
    tf::Transform transformGPS;
    tf::Quaternion q;
    static tf::TransformBroadcaster br;

    auto t = _pose.translation();
    auto R_yaw = _pose.so3().matrix();

    transformGPS.setOrigin(tf::Vector3(t[0], t[1], t[2]));
    Eigen::Quaterniond quat_(R_yaw);
    q = tf::Quaternion(quat_.x(), quat_.y(), quat_.z(), quat_.w());
    transformGPS.setRotation(q);
    br.sendTransform(tf::StampedTransform(transformGPS, ros::Time().fromSec(msg_time), "world", "MLS"));

    // tf::Transform transform_inv = transformGPS.inverse();
    // static tf::TransformBroadcaster br2;
    // br2.sendTransform(tf::StampedTransform(transform_inv, ros::Time().fromSec(msg_time), "PPK_GNSS", "world"));
}

void publish_ppk_gnss(const Sophus::SE3 &_pose, const double &msg_time)
{
    tf::Transform transformGPS;
    tf::Quaternion q;
    static tf::TransformBroadcaster br;

    auto t = _pose.translation();
    auto R_yaw = _pose.so3().matrix();

    transformGPS.setOrigin(tf::Vector3(t[0], t[1], t[2]));
    Eigen::Quaterniond quat_(R_yaw);
    q = tf::Quaternion(quat_.x(), quat_.y(), quat_.z(), quat_.w());
    transformGPS.setRotation(q);
    br.sendTransform(tf::StampedTransform(transformGPS, ros::Time().fromSec(msg_time), "world", "PPK_GNSS"));

    // tf::Transform transform_inv = transformGPS.inverse();
    // static tf::TransformBroadcaster br2;
    // br2.sendTransform(tf::StampedTransform(transform_inv, ros::Time().fromSec(msg_time), "PPK_GNSS", "world"));
}

bool readSE3FromFile(const std::string &filename, Sophus::SE3 &transform_out)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file " << filename << " for reading.\n";
        return false;
    }

    std::string line;
    Eigen::Matrix4d mat;
    int row = 0;

    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        if (line.find("T_als2mls") != std::string::npos)
            continue;

        std::istringstream iss(line);
        for (int col = 0; col < 4; ++col)
        {
            iss >> mat(row, col);
        }
        ++row;
        if (row == 4)
            break;
    }

    if (row != 4)
    {
        std::cerr << "Failed to read full 4x4 matrix from " << filename << "\n";
        return false;
    }

    transform_out = Sophus::SE3(mat.block<3, 3>(0, 0), mat.block<3, 1>(0, 3));
    return true;
}

void publishAccelerationArrow(ros::Publisher &marker_pub, const Eigen::Vector3d &acceleration, const double &msg_time)
{
    visualization_msgs::Marker arrow;

    arrow.header.frame_id = "PPK_GNSS"; // the location of the ppk gnss imu
    arrow.header.stamp = ros::Time().fromSec(msg_time);
    arrow.ns = "acceleration_arrow";
    arrow.id = 0;
    arrow.type = visualization_msgs::Marker::ARROW;
    arrow.action = visualization_msgs::Marker::ADD;

    // Define arrow start (origin) and end (acceleration direction)
    geometry_msgs::Point start, end;
    start.x = 0.0;
    start.y = 0.0;
    start.z = 0.0;

    end.x = acceleration.x();
    end.y = acceleration.y();
    end.z = acceleration.z();

    arrow.points.push_back(start);
    arrow.points.push_back(end);

    // Set arrow properties
    arrow.scale.x = 0.3;
    arrow.scale.y = 0.3;
    arrow.scale.z = 0.3;

    arrow.color.r = 1.0; // Full red
    arrow.color.g = 0.5; // Medium green
    arrow.color.b = 0.0; // No blue
    arrow.color.a = 1.0; // Fully opaque

    marker_pub.publish(arrow);
}

// #include "../src3/clean_registration3.hpp"

#include "TrajectoryReader.hpp"

// Solves AX = XB problem

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/expressions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/ExpressionFactor.h>


using namespace gtsam;
using gtsam::symbol_shorthand::X; // extrinsic key X(0)
using Matrix6 = Eigen::Matrix<double, 6, 6>;

// Factor constraining extrinsic X so that:
// delta_I_meas ≈ X * delta_L_meas * X^{-1}
class ExtrinsicSE3Factor : public NoiseModelFactor1<Pose3> {
private:
    Pose3 delta_I_;
    Pose3 delta_L_;

public:
    ExtrinsicSE3Factor(Key key, const Pose3 &delta_I, const Pose3 &delta_L,
                       const SharedNoiseModel &model)
        : NoiseModelFactor1<Pose3>(model, key), delta_I_(delta_I), delta_L_(delta_L) {}

    Vector evaluateError(const Pose3 &T_I_L, OptionalMatrixType H) const override {
        // predicted delta_I from lidar delta and extrinsic
        Pose3 predicted = T_I_L * delta_L_ * T_I_L.inverse();

        // error = Logmap( delta_I^{-1} * predicted ) = Pose3::Logmap(delta_I.between(predicted))
        Pose3 errPose = delta_I_.between(predicted);
        Vector error = Pose3::Logmap(errPose); // 6x1 (rot(3), trans(3))

        if (H) {
            // numerical Jacobian for simplicity/robustness
            auto fun = [this](const Pose3 &X) -> Vector {
                Pose3 pred = X * delta_L_ * X.inverse();
                Pose3 e = delta_I_.between(pred);
                return Pose3::Logmap(e);
            };
            *H = numericalDerivative11<Vector, Pose3>(fun, T_I_L, 1e-6);
        }
        return error;
    }
};

#include <unsupported/Eigen/KroneckerProduct>
#include <Eigen/LU>
#include <Eigen/QR>

Eigen::Matrix3d skew(Eigen::Vector3d u)
{
    Eigen::Matrix3d u_hat = Eigen::MatrixXd::Zero(3, 3);
    u_hat(0, 1) = u(2);
    u_hat(1, 0) = -u(2);
    u_hat(0, 2) = -u(1);
    u_hat(2, 0) = u(1);
    u_hat(1, 2) = u(0);
    u_hat(2, 1) = -u(0);

    return u_hat;
}

Sophus::SE3 ConventionalAXXBSVDSolver(std::vector<Sophus::SE3> &A_, std::vector<Sophus::SE3> &B_)
{
    Eigen::MatrixXd m = Eigen::MatrixXd::Zero(12 * A_.size(), 12);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(12 * A_.size());
    for (int i = 0; i < A_.size(); i++)
    {
        // extract R,t from homogophy matrix
        M3D Ra = A_[i].so3().matrix(); // topLeftCorner(3,3);
        V3D Ta = A_[i].translation();  // topRightCorner(3,1);
        M3D Rb = B_[i].so3().matrix(); // topLeftCorner(3,3);
        V3D Tb = B_[i].translation();  // topRightCorner(3,1);

        m.block<9, 9>(12 * i, 0) = Eigen::MatrixXd::Identity(9, 9) - Eigen::kroneckerProduct(Ra, Rb);

        Eigen::Matrix3d Ta_skew = skew(Ta);

        m.block<3, 9>(12 * i + 9, 0) = Eigen::kroneckerProduct(Eigen::MatrixXd::Identity(3, 3), Tb.transpose());
        m.block<3, 3>(12 * i + 9, 9) = Eigen::MatrixXd::Identity(3, 3) - Ra;
        b.block<3, 1>(12 * i + 9, 0) = Ta;
    }

    Eigen::Matrix<double, 12, 1> x = m.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    Eigen::Matrix3d R = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(x.data()); // row major

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(R, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix4d handeyetransformation = Eigen::Matrix4d::Identity(4, 4);
    handeyetransformation.topLeftCorner(3, 3) = svd.matrixU() * svd.matrixV().transpose();
    handeyetransformation.topRightCorner(3, 1) = x.block<3, 1>(9, 0);

    Eigen::Vector3d t = handeyetransformation.block<3, 1>(0, 3);
    Sophus::SO3 SO3(handeyetransformation.block<3, 3>(0, 0));

    return Sophus::SE3(SO3, t);
}

void DataHandler::Subscribe()
{
    // // ---- ground truth extrinsic: T_I_L (maps L -> I) ----
    // Rot3 R_true = Rot3::RzRyRx(-0.2, -0.5, 0.05); // rx, ry, rz
    // Point3 t_true(0.5, -1.2, 0.1);
    // Pose3 T_I_L_true(R_true, t_true);

    // std::cout << "Ground truth extrinsic (T_I_L_true):\n" << T_I_L_true << std::endl;

    // // ---- synthetic L trajectory (poses of lidar in world) ----
    // const size_t N = 30;
    // std::vector<Pose3> poses_w_L;
    // poses_w_L.reserve(N);
    // for (size_t i = 0; i < N; ++i) {
    //     double x = 0.2 * double(i);
    //     double y = 0.05 * std::sin(double(i) * 0.3);
    //     double z = 0.0;
    //     double roll = 0.01 * double(i % 5);
    //     double pitch = 0.005 * std::sin(double(i) * 0.2);
    //     double yaw = 0.02 * double(i);
    //     Rot3 R = Rot3::RzRyRx(roll, pitch, yaw);
    //     poses_w_L.emplace_back(R, Point3(x, y, z));
    // }

    // // ---- CORRECT: compute corresponding I poses using T_wI = T_wL * T_LI
    // // T_LI = T_I_L_true.inverse()
    // std::vector<Pose3> poses_w_I; poses_w_I.reserve(N);
    // Pose3 T_L_I = T_I_L_true.inverse(); // T_LI
    // for (size_t i = 0; i < N; ++i) {
    //     // **correct composition**: world_I = world_L * T_LI
    //     poses_w_I.push_back( poses_w_L[i].compose(T_L_I) );
    //     // (equivalently: poses_w_L[i] * T_L_I)
    // }

    // // ---- Build graph with many relative-motion factors ----
    // NonlinearFactorGraph graph;
    // Values initial;

    // double meas_sigma = 0.02; // rad/m synthetic measurement sigma
    // auto noise = noiseModel::Isotropic::Sigma(6, meas_sigma);

    // for (size_t i = 0; i + 1 < N; ++i) {
    //     Pose3 delta_L_true = poses_w_L[i].between(poses_w_L[i + 1]);
    //     Pose3 delta_I_true = poses_w_I[i].between(poses_w_I[i + 1]);

    //     graph.add( std::make_shared<ExtrinsicSE3Factor>(X(0), delta_I_true, delta_L_true, noise) );
    // }

    // // Weak prior (very large sigmas = weak)
    // auto priorNoise = noiseModel::Diagonal::Sigmas((Vector(6) << 10., 10., 10., 10., 10., 10.).finished());
    // graph.add( PriorFactor<Pose3>(X(0), Pose3::Identity(), priorNoise) );

    // // initial guess (identity or slightly perturbed)
    // Pose3 init_guess = Pose3::Identity();
    // initial.insert(X(0), init_guess);
    // std::cout << "Initial guess for extrinsic:\n" << init_guess << std::endl;

    // // Optimize
    // LevenbergMarquardtParams params;
    // params.setVerbosity("ERROR");
    // LevenbergMarquardtOptimizer optimizer(graph, initial); //, params
    // Values result = optimizer.optimize();

    // Pose3 T_I_L_est = result.at<Pose3>(X(0));
    // std::cout << "\nEstimated extrinsic (T_I_L_est):\n" << T_I_L_est << std::endl;

    // // Compare to ground truth
    // Pose3 errPose = T_I_L_true.between(T_I_L_est);
    // Vector errXi = Pose3::Logmap(errPose);
    // std::cout << "\nEstimation error (Pose3 Logmap) [rotX,rotY,rotZ, tx,ty,tz]:\n" << errXi.transpose() << std::endl;

    // Covariance
    //Marginals marginals(graph, result);
    //Eigen::MatrixXd cov = marginals.marginalCovariance(X(0));
    //std::cout << "\nMarginal covariance for X(0) (6x6):\n" << cov << std::endl;
    // Marginals for covariance of estimated extrinsic
    //gtsam::Marginals marginals(graph, result);
    // Matrix cov = marginals.marginalCovariance(X(0));
    // std::cout << "\nMarginal covariance for X(0) (6x6):\n" << cov << std::endl;


    std::cout << "Run test" << std::endl;
    std::cout << std::fixed << std::setprecision(12);
    std::cerr << std::fixed << std::setprecision(12);

    std::cout << "\n=============================== Georeference the data ===============================" << std::endl;

#ifdef USE_EKF
    std::shared_ptr<IMU_Class> imu_obj(new IMU_Class());
#else
    std::shared_ptr<Graph> imu_obj(new Graph());
#endif

    std::shared_ptr<GNSS> gnss_obj(new GNSS());

    Sophus::SE3 Lidar_wrt_IMU = Sophus::SE3(Lidar_R_wrt_IMU, Lidar_T_wrt_IMU); // this is the known extrinsic transformation
    imu_obj->set_param(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU, V3D(gyr_cov, gyr_cov, gyr_cov), V3D(acc_cov, acc_cov, 10. * acc_cov),
                       V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov), V3D(b_acc_cov, b_acc_cov, 10. * b_acc_cov));
    gnss_obj->set_param(GNSS_T_wrt_IMU, GNSS_IMU_calibration_distance, postprocessed_gnss_path);

    Sophus::SE3 Lidar_wrt_IMU_estim = Sophus::SE3();

    // #define USE_ALS

    // #ifdef USE_ALS
    //     std::shared_ptr<ALS_Handler> als_obj = std::make_shared<ALS_Handler>(folder_root, downsample, closest_N_files, filter_size_surf_min);
    //     ros::Publisher pubLaserALSMap = nh.advertise<sensor_msgs::PointCloud2>("/ALS_map", 1000);
    // #endif

    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudLocal = nh.advertise<sensor_msgs::PointCloud2>("/cloud_local", 100000);

    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100);

    ros::Publisher pubLaserCloudDebug = nh.advertise<sensor_msgs::PointCloud2>("/cloud_debug", 10);
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("acceleration_marker", 10);

    Sophus::SE3 curr_mls;
    bool perform_mls_registration = true;

    Eigen::Matrix4d T_lidar2gnss;
    T_lidar2gnss << 0.0131683606, -0.9998577263, 0.0105414145, 0.0154123047,
        0.9672090675, 0.0100627670, -0.2537821120, -2.6359450601,
        0.2536399297, 0.0135376461, 0.9672039693, -0.5896374492,
        0.0, 0.0, 0.0, 1.0;
    M3D R_lidar2gnss = T_lidar2gnss.block<3, 3>(0, 0); // Rotation
    V3D t_lidar2gnss = T_lidar2gnss.block<3, 1>(0, 3); // Translation
    // gnss should be rtansformed to mls frame
    Sophus::SE3 lidar2gnss(R_lidar2gnss, t_lidar2gnss); // FROM LIDAR 2 GNSS   T_lidar = T_gnss * lidar2gnss.inverse()
    Sophus::SE3 gnss2lidar = lidar2gnss.inverse();      // THIS FOR THE BACK ANTENA - TO TRANSFORM TO FRONT imu frame
    //----------------------------------------------------------------------------

    std::cout << "\n\nStart reading the data..." << std::endl;
    //------------------------------------------------------------------------------
    TrajectoryReader reader;
    // //for drone we had this
    // M3D T;
    // T << -1,  0,  0,
    //       0,  0, -1,
    //       0, -1,  0;
    // gnss2lidar = Sophus::SE3(T, V3D::Zero()); //this was required for the drone

    // an extrinsic transformation is passed here to transform the ppk gnss-imu orientaiton into mls frame
    reader.read(postprocessed_gnss_path, gnss2lidar);

    // --- Access measurements ---
    const auto &measurements = reader.measurements();
    std::cout << "Parsed " << measurements.size() << " measurements." << std::endl;
    int total_m = measurements.size();

    if (!measurements.empty())
    {
        const auto &m = measurements[0]; // first measurement
        std::cout << "First measurement:" << std::endl;
        std::cout << "  GPSTime = " << m.GPSTime << " sec" << std::endl;
        std::cout << "  Position (E,N,H) = ("
                  << m.Easting << ", "
                  << m.Northing << ", "
                  << m.H_Ell << ")" << std::endl;
        std::cout << "  Orientation (Phi, Omega, Kappa) = ("
                  << m.Phi << ", "
                  << m.Omega << ", "
                  << m.Kappa << ")" << std::endl;
        std::cout << "  AccBias (X,Y,Z) = ("
                  << m.AccBiasX << ", "
                  << m.AccBiasY << ", "
                  << m.AccBiasZ << ")" << std::endl;

        std::cout << "  AngRate (X,Y,Z) = ("
                  << m.AngRateX << ", "
                  << m.AngRateY << ", "
                  << m.AngRateZ << ")" << std::endl;

        std::cout << "  VelBdy (X,Y,Z) = ("
                  << m.VelBdyX << ", "
                  << m.VelBdyY << ", "
                  << m.VelBdyZ << ")" << std::endl;

        std::cout << "First measurement m.utc_usec :" << m.utc_usec << std::endl;
        std::cout << "First measurement m.utc_usec2:" << m.utc_usec2 << std::endl;

        // First measurement m.utc_usec :   1721898390000000.000000000000
        // First measurement m.utc_usec2:   1721898390.000000000000

        // pcl_cbk msg->  .stamp.toSec():    1721900923.978538036346
    }

    auto m0 = measurements[0];
    V3D raw_gyro;
    V3D raw_acc, gravity_free_acc = V3D(m0.AccBdyX, m0.AccBdyY, m0.AccBdyZ);
    reader.addEarthGravity(measurements[reader.curr_index], raw_gyro, raw_acc, G_m_s2); // this will add the world gravity
    // reader.addGravity(measurements[reader.curr_index], se3, raw_gyro, raw_acc, G_m_s2); //gravity in curr body frame

    std::cout << "gravity_free_acc:" << gravity_free_acc.transpose() << std::endl;
    std::cout << "raw_acc:" << raw_acc.transpose() << std::endl;

    Sophus::SE3 first_ppk_gnss_pose_inverse = Sophus::SE3();
    reader.toSE3(m0, first_ppk_gnss_pose_inverse);
    first_ppk_gnss_pose_inverse = first_ppk_gnss_pose_inverse.inverse();

    int tmp_index = 0;
    ros::Rate rate(500);

    int scan_id = 0;

    bool use_als = true, als_integrated = false;
    using namespace std::chrono;

    bool ppk_gnss_synced = false;
    Sophus::SE3 se3 = Sophus::SE3();
    // Alignment transform: GNSS -> LiDAR
    Sophus::SE3 T_LG = Sophus::SE3();
    std::vector<std::string> topics{lid_topic, imu_topic, gnss_topic};

    std::vector<std::string> bag_files = expandBagPattern(bag_file);
    std::cout << "bag_files:" << bag_files.size() << std::endl;
    if (bag_files.size() == 0)
    {
        std::cerr << "Error: Bag file does not exist or is not accessible: " << bag_file << std::endl;
        return;
    }
    for (auto &f : bag_files)
        std::cout << "Matched: " << f << std::endl;

    // Open all bags
    std::vector<std::shared_ptr<rosbag::Bag>> bags;
    for (const auto &file : bag_files)
    {
        auto bag = std::make_shared<rosbag::Bag>();
        bag->open(file, rosbag::bagmode::Read);
        bags.push_back(bag);
        ROS_INFO_STREAM("Opened bag: " << file);
    }

    // Build a single view from all bags
    rosbag::View view;
    for (auto &b : bags)
    {
        view.addQuery(*b, rosbag::TopicQuery(topics));
    }

    signal(SIGINT, SigHandle); // Handle Ctrl+C (SIGINT)
    flg_exit = false;
    perform_mls_registration = true;
    bool ppk_gnss_oriented = false;

    Sophus::SE3 tmp_pose = Sophus::SE3();

    std::vector<PointCloudXYZI::Ptr> original_scans_;
    std::vector<Sophus::SE3> lidar_poses_;
    std::vector<Sophus::SE3> gnss_imu_poses_;

    bool has_prev_cloud_ = false;
    PointCloudXYZI::Ptr prev_cloud_;
    prev_cloud_.reset(new PointCloudXYZI());

    Sophus::SE3 global_pose_ = Sophus::SE3();                   // Global pose
    Sophus::SE3 last_relative_motion_estimate_ = Sophus::SE3(); // Constant velocity model

    for (const rosbag::MessageInstance &m : view)
    {
        ros::spinOnce();
        if (flg_exit || !ros::ok())
            break;

        std::string topic = m.getTopic();
        if (topic == imu_topic)
        {
            sensor_msgs::Imu::ConstPtr imu_msg = m.instantiate<sensor_msgs::Imu>();
            if (imu_msg)
            {
                imu_cbk(imu_msg);
                continue;
            }
        }
        else if (topic == gnss_topic)
        {
            gps_common::GPSFix::ConstPtr gps_msg = m.instantiate<gps_common::GPSFix>();
            if (gps_msg)
            {
                gps_cbk(gps_msg);
                continue;
            }
        }
        else if (topic == lid_topic)
        {
            sensor_msgs::PointCloud2::ConstPtr pcl_msg = m.instantiate<sensor_msgs::PointCloud2>();
            if (pcl_msg)
            {
                pcl_cbk(pcl_msg);
            }
        }

        if (sync_packages(Measures))
        {
            scan_id++;
            std::cout << "scan_id:" << scan_id << std::endl;
            if (scan_id > 8000) // 500 1050 used for data before
            {
                std::cout << "Stop here... enough data 8000 scans" << std::endl;
                break;
            }

            {
                double t00 = omp_get_wtime();

                gnss_obj->Process(gps_buffer, lidar_end_time, state_point.pos);
                Sophus::SE3 gnss_pose = (gnss_obj->use_postprocessed_gnss) ? gnss_obj->postprocessed_gps_pose : gnss_obj->gps_pose;
                // gnss_pose.so3() = state_point.rot; // use the MLS orientation
                // if (als_integrated) // if (use_gnss)
                publish_gnss_odometry(gnss_pose);

                double time_of_day_sec = gnss_obj->tod; // this is at the end of the scan

                if (!reader.initted)
                {
                    if (!reader.init(time_of_day_sec))
                    {
                        std::cerr << "Cannot initialize the GNSS-IMU reader..." << std::endl;
                        throw std::runtime_error("Cannot initialize the GNSS-IMU reader...: time_of_day_sec " + std::to_string(time_of_day_sec));
                    }
                    else
                    {
                        reader.initted = true;
                        tmp_index = reader.curr_index;
                        std::cout << "init Initialization succeeded.." << std::endl;
                        std::cout << "tmp_index:" << tmp_index << std::endl;
                    }
                    continue;
                }

                if (reader.initted)
                {
                    tmp_index = reader.curr_index;
                    if (!ppk_gnss_synced)
                    {
                        const auto &m = measurements[tmp_index];
                        Sophus::SE3 interpolated_pose;
                        reader.toSE3(m, interpolated_pose);

                        // take only the position of the first pose - keeps the orientation as it it, so gravity = earth gravity
                        first_ppk_gnss_pose_inverse = Sophus::SE3(M3D::Identity(), interpolated_pose.translation()).inverse();

                        // doing this we put everything in the frame of the first pose - gravity here is not the earth gravity
                        // first_ppk_gnss_pose_inverse = interpolated_pose.inverse(); //this will rotate the world - so that gravity
                        // the earth gravity can be added using the current system rotation in the world frame

                        tmp_pose = first_ppk_gnss_pose_inverse * interpolated_pose;

                        // Convert to Euler (ZYX: yaw-pitch-roll)
                        V3D euler = interpolated_pose.so3().matrix().eulerAngles(2, 1, 0);
                        // euler[0] = yaw (around Z), euler[1] = pitch (around Y), euler[2] = roll (around X)
                        std::cout << "Euler angles (rad): " << euler.transpose() << std::endl;
                        std::cout << "Euler angles (deg): " << euler.transpose() * 180.0 / M_PI << std::endl;

                        ppk_gnss_synced = true;
                        std::cout << "\nsynchronised\n, press enter..." << std::endl;
                        std::cin.get();
                        continue;
                    }

                    double time_start = time_of_day_sec - .1;
                    double time_end = time_of_day_sec;

                    auto interpolated_pose = reader.closestPose(time_start);

                    tmp_index = reader.curr_index;
                    const auto &msg_time = measurements[tmp_index].tod;

                    se3 = T_LG * first_ppk_gnss_pose_inverse * interpolated_pose; // in first frame
                    publish_ppk_gnss(se3, msg_time);

                    // reader.addEarthGravity(measurements[reader.curr_index], raw_gyro, raw_acc, G_m_s2); //this will add the world gravity
                    reader.addGravity(measurements[reader.curr_index], se3, raw_gyro, raw_acc, G_m_s2); // gravity in curr body frame

                    // todo - we can do it the other way around and add the gravity in IMU body frame
                    publishAccelerationArrow(marker_pub, -raw_acc, msg_time);

                    {
                        // main stuff here
                        /*

                        std::vector<PointCloudXYZI::Ptr> original_scans_;
                        std::vector<Sophus::SE3> lidar_poses_;
                        std::vector<Sophus::SE3> gnss_imu_poses_;


                        ---setup the gnss reader
                        take a list of scans - 100m
                        for each scan find the closest pose on time
                        georeference the scan and display it
                        find relative transform between every 2 scan

                        find the extrinsics - based on relative pieces

                        try scan to scan to get the absolute pose

                        AT EACH STEP - COMPUTE THE RESIDUAL WITH GT EXTRINSIC
                            SEPARATE FOR TRANSLATION AND ROTATION


                        for a number of iterations
                            use the prev values of extrinsics
                            undistort the scan
                            georeference it - and plot it
                            redoo the registration and extrinsic estimation

                        */

                        auto dist = se3.translation().norm();
                        std::cout << "dist:" << dist << std::endl;
                        if (dist < 40)
                        {
                            *feats_undistort = *Measures.lidar; // lidar frame
                            {
                                downSizeFilterSurf.setInputCloud(feats_undistort);
                                downSizeFilterSurf.filter(*feats_down_body);

                                if (!has_prev_cloud_)
                                {
                                    *prev_cloud_ = *feats_down_body;
                                    has_prev_cloud_ = true;
                                    continue;
                                }

                                // --- ICP Setup ---
                                pcl::IterativeClosestPoint<PointType, PointType> icp;
                                icp.setMaximumIterations(50);
                                icp.setMaxCorrespondenceDistance(1.0);
                                icp.setInputSource(feats_down_body);
                                icp.setInputTarget(prev_cloud_);

                                // --- Constant velocity model as initial guess ---
                                Eigen::Matrix4f init_guess = last_relative_motion_estimate_.matrix().cast<float>();

                                PointCloudXYZI::Ptr aligned(new PointCloudXYZI());
                                icp.align(*aligned, init_guess);

                                if (!icp.hasConverged())
                                {
                                    ROS_WARN("ICP did not converge!");
                                    return;
                                }

                                // --- Get relative transform from ICP ---
                                Eigen::Matrix4f relative_transform_f = icp.getFinalTransformation();
                                Eigen::Matrix4d relative_transform = relative_transform_f.cast<double>();

                                Sophus::SE3 relative_motion(relative_transform.block<3, 3>(0, 0),
                                                            relative_transform.block<3, 1>(0, 3));

                                // --- Update global pose ---
                                global_pose_ = global_pose_ * relative_motion;

                                // --- Update constant velocity model ---
                                last_relative_motion_estimate_ = relative_motion;

                                publishPose(global_pose_, lidar_end_time);

                                *prev_cloud_ = *feats_down_body;

                                original_scans_.push_back(feats_undistort);
                                gnss_imu_poses_.push_back(se3);
                                lidar_poses_.push_back(global_pose_);
                            }

                            TransformPoints(Lidar_wrt_IMU_estim, feats_undistort); // lidar to IMU frame - front IMU

                            downSizeFilterSurf.setInputCloud(feats_undistort);
                            downSizeFilterSurf.filter(*feats_down_body);

                            TransformPoints(se3, feats_down_body); // georeference with se3 in IMU frame

                            publish_frame_debug(pubLaserCloudDebug, feats_down_body);
                        }
                        else
                        {
                            // for a number of iteration do:
                            // transform scan to imu with latest estimated
                            // undistort with const vel model - or imu
                            // redoo the icp - for better T_rel
                            // re-estimate the extrinsic
                            // plot the merged cloud with the fine-tuned extrinsics after every iteration
                            // stop when no more change happens
                            // find the error w.r.t. for known T - plot


                            // ---- Build graph with many relative-motion factors ----
                            std::vector<Sophus::SE3> A_,B_;

//todo here 

//TODO first \

using the GT pose try to simulate the other sensor motion by using the transform 

for a number of iterations 
    using the current estimate transform all the scans into imu frame 
    find all the relative transforms 
    estimate the extrinsic 

    transform scans to imu frame and plot all toghether - enter

    iter 2 
    transform the scans again into imu fram with lates extrinsic 
    undistory - const vel model
    estimate relative transform 
    estimate extrinsic

    .and so on untill the error is not longer changing 

    compute the error with the GT pose 


                    NonlinearFactorGraph graph;
                    Values initial;

                    double meas_sigma = 0.02; // rad/m synthetic measurement sigma
                    auto noise = noiseModel::Isotropic::Sigma(6, meas_sigma);
                    
                    int N = gnss_imu_poses_.size();
                    for (size_t i = 0; i + 1 < N; ++i) {

                        auto delta_L = lidar_poses_[i].inverse() * lidar_poses_[i+1];
                        auto delta_I = gnss_imu_poses_[i].inverse() * gnss_imu_poses_[i+1];

                        Pose3 delta_L_true = Pose3(delta_L.matrix());
                        Pose3 delta_I_true = Pose3(delta_I.matrix());

                        graph.add( std::make_shared<ExtrinsicSE3Factor>(X(0), delta_I_true, delta_L_true, noise) );

                        A_.push_back(delta_L);
                        B_.push_back(delta_I);
                    }

                    // Weak prior (very large sigmas = weak)
                    auto priorNoise = noiseModel::Diagonal::Sigmas((Vector(6) << 10., 10., 10., 10., 10., 10.).finished());
                    graph.add( PriorFactor<Pose3>(X(0), Pose3::Identity(), priorNoise) );

                    // initial guess (identity or slightly perturbed)
                    Pose3 init_guess = Pose3::Identity();
                    initial.insert(X(0), init_guess);
                    std::cout << "Initial guess for extrinsic:\n" << init_guess << std::endl;

                    // Optimize
                    LevenbergMarquardtParams params;
                    params.setVerbosity("ERROR");
                    LevenbergMarquardtOptimizer optimizer(graph, initial); //, params
                    Values result = optimizer.optimize();

                    Pose3 T_I_L_est = result.at<Pose3>(X(0));
                    std::cout << "\nEstimated extrinsic (T_I_L_est):\n" << T_I_L_est << std::endl;

                    // Compare to ground truth
                    Pose3 T_I_L_true = Pose3(Lidar_wrt_IMU.matrix());
                    Pose3 errPose = T_I_L_true.between(T_I_L_est);
                    Vector errXi = Pose3::Logmap(errPose);
                    std::cout << "\nEstimation error (Pose3 Logmap) [rotX,rotY,rotZ, tx,ty,tz]:\n" << errXi.transpose() << std::endl;


                    Sophus::SE3 out =  ConventionalAXXBSVDSolver(B_, A_);
                    std::cout<<"out\n:"<<out.matrix()<<std::endl;


                            std::cout << "\nFinished, press enter..." << std::endl;
                            std::cin.get();

                            return;
                        }

                        // reader.undistort_const_vel(time_start, feats_undistort); // const vel model
                        // reader.undistort_imu(time_start, feats_undistort); //imu measurements
                    }
                }
                else
                {
                    std::cout << "GNSS reader not initted..." << std::endl;
                    throw std::runtime_error("GNSS reader not initted...");
                }

                double t11 = omp_get_wtime();
                std::cout << "Mapping time(ms):  " << (t11 - t00) * 1000 << ", feats_down_size: " << feats_down_size << ", lidar_end_time:" << lidar_end_time << "\n"
                          << std::endl;

                // std::this_thread::sleep_for(std::chrono::milliseconds(50)); // to simulate lidar measurements
            }
        }
    }
    // bag.close();
    for (auto &b : bags)
        b->close();

    // cv::destroyAllWindows(); */
}
