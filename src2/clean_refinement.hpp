#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/PreintegrationCombinedParams.h>
#include <gtsam/navigation/PreintegratedCombinedMeasurements.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/base/Matrix.h>
#include <pcl/point_types.h>

using namespace gtsam;
using symbol_shorthand::B; // Bias
using symbol_shorthand::V; // Velocity
using symbol_shorthand::X; // Pose

// Custom PointToPlaneFactor same as before
class PointToPlaneFactor : public NoiseModelFactor1<Pose3>
{
  Point3 point_;
  Point3 ref_;
  Unit3 normal_;

public:
  PointToPlaneFactor(Key key, const Point3 &point, const Point3 &ref, const Unit3 &normal, const SharedNoiseModel &model)
      : NoiseModelFactor1<Pose3>(model, key), point_(point), ref_(ref), normal_(normal) {}

  Vector evaluateError(const Pose3 &pose, boost::optional<Matrix &> H = boost::none) const override
  {
    Point3 transformed = pose.transformFrom(point_, H);
    double error = (transformed - ref_).dot(normal_.unitVector());
    if (H)
      H->row(0) = normal_.unitVector().transpose() * H->block<3, 6>(0, 0);
    return Vector1(error);
  }
};

int main()
{
  NonlinearFactorGraph graph;
  Values initialEstimate;
  ISAM2 isam;

  // IMU parameters
  auto imuParams = PreintegrationCombinedParams::MakeSharedD(Vector3(0, 0, -9.81));
  imuParams->accelerometerCovariance = I_3x3 * 1e-3;
  imuParams->gyroscopeCovariance = I_3x3 * 1e-4;
  imuParams->integrationCovariance = I_3x3 * 1e-4;

  imuBias::ConstantBias priorBias; // zero bias
  auto priorPoseNoise = noiseModel::Diagonal::Sigmas((Vector(6) << 1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3).finished());
  auto priorVelNoise = noiseModel::Isotropic::Sigma(3, 1e-2);
  auto priorBiasNoise = noiseModel::Isotropic::Sigma(6, 1e-3);

  Pose3 prevPose = Pose3::identity();
  Vector3 prevVel(0, 0, 0);
  imuBias::ConstantBias prevBias;

  PreintegratedCombinedMeasurements imuIntegrator(imuParams, prevBias);

  initialEstimate.insert(X(0), prevPose);
  initialEstimate.insert(V(0), prevVel);
  initialEstimate.insert(B(0), prevBias);

  graph.add(PriorFactor<Pose3>(X(0), prevPose, priorPoseNoise));
  graph.add(PriorFactor<Vector3>(V(0), prevVel, priorVelNoise));
  graph.add(PriorFactor<imuBias::ConstantBias>(B(0), prevBias, priorBiasNoise));

  size_t scanIdx = 1;

  while (getNextScan())
  {
    double dt = getImuToLidarDeltaT(); // Time interval between last and current LiDAR scan

    // Integrate IMU measurements
    for (auto &imu : getImuMeasurementsForInterval())
    {
      imuIntegrator.integrateMeasurement(imu.accel, imu.gyro, imu.dt);
    }

    auto pim = imuIntegrator;
    auto imuFactor = CombinedImuFactor(X(scanIdx - 1), V(scanIdx - 1),
                                       X(scanIdx), V(scanIdx),
                                       B(scanIdx - 1), B(scanIdx),
                                       pim);
    graph.add(imuFactor);

    // Predict initial estimate for ISAM
    NavState prevState(prevPose, prevVel);
    NavState predicted = pim.predict(prevState, prevBias);
    Pose3 initPose = predicted.pose();
    Vector3 initVel = predicted.v();

    initialEstimate.insert(X(scanIdx), initPose);
    initialEstimate.insert(V(scanIdx), initVel);
    initialEstimate.insert(B(scanIdx), prevBias); // assuming no bias change

    // Point-to-plane constraints
    auto lidarScan = getLidarScan();
    for (const auto &pt : lidarScan.points)
    {
      Point3 p(pt.x, pt.y, pt.z);
      Point3 ref;
      Unit3 normal;
      if (findClosestPlane(p, initPose, ref, normal))
      {
        auto noise = noiseModel::Isotropic::Sigma(1, 0.05);
        graph.add(boost::make_shared<PointToPlaneFactor>(X(scanIdx), p, ref, normal, noise));
      }
    }

    isam.update(graph, initialEstimate);
    Values result = isam.calculateEstimate();

    // Prepare for next scan
    prevPose = result.at<Pose3>(X(scanIdx));
    prevVel = result.at<Vector3>(V(scanIdx));
    prevBias = result.at<imuBias::ConstantBias>(B(scanIdx));

    imuIntegrator.resetIntegrationAndSetBias(prevBias);
    graph.resize(0);
    initialEstimate.clear();
    scanIdx++;
  }

  return 0;
}



//   Eigen::Vector3d point_on_line = center;
    //             Eigen::Vector3d point_a, point_b;
    //             point_a = 0.1 * unit_direction + point_on_line;
    //             point_b = -0.1 * unit_direction + point_on_line;
      

      //alternative
    //   Point3 lp = pose.transformFrom(curr_point_); // apply SE(3) transform
    //     Vector3 nu = (lp - last_point_a).cross(lp - last_point_b);
    //     Vector3 de = last_point_a - last_point_b;

    //     Vector3 residual = nu / de.norm();

    //     if (H) {
    //         // Compute Jacobian manually or use numerical approx for prototype
    //         // For now: set numerical derivative placeholder
    //         *H = numericalDerivative11<Vector3, Pose3>(
    //                 std::bind(&PointToLineFactor::evaluateError, this, std::placeholders::_1, boost::none), pose);
    //     }

      //ALTERNATIVE
    //   Eigen::Vector3d l = last_point_b_ - last_point_a_;
    //     Eigen::Vector3d vec = lp - last_point_a_;

    //     Eigen::Vector3d cross = vec.cross(l);
    //     double norm_l = l.norm();

    //     if (H) {
    //     Eigen::Matrix3d R = pose.rotation().matrix();
    //     Eigen::Matrix3d skew_cp = gtsam::skewSymmetric(R * curr_point_);
    //     Eigen::Matrix<double, 3, 6> d_lp_d_pose;
    //     d_lp_d_pose << -skew_cp, Eigen::Matrix3d::Identity();

    //     Eigen::Matrix<double, 1, 3> d_norm = cross.normalized().transpose();
    //     Eigen::Matrix<double, 3, 3> d_cross_vec = gtsam::skewSymmetric(l);

    //     *H = (d_norm * d_cross_vec * d_lp_d_pose) / norm_l;
    //     }

    //     return gtsam::Vector1(cross.norm() / norm_l);







    // H_point_wrt_pose (3×6 matrix)
            /*
            [ ∂x/∂tx ∂x/∂ty ∂x/∂tz ∂x/∂rx ∂x/∂ry ∂x/∂rz ]
            [ ∂y/∂tx ∂y/∂ty ∂y/∂tz ∂y/∂rx ∂y/∂ry ∂y/∂rz ]
            [ ∂z/∂tx ∂z/∂ty ∂z/∂tz ∂z/∂rx ∂z/∂ry ∂z/∂rz ]
            */
            // First 3 columns: Derivatives wrt translation (simple)
            // Last 3 columns: Derivatives wrt rotation (involves cross products)
            //[ nx ny nz ] (1×3)  *  [ 3×6 Jacobian ]  =  [ 1×6 Jacobian ]
            //  Jacobian should be 1x6 (1D error, 6D pose)
            //*H = (plane_normal_.transpose() * H_point_wrt_pose); // * weight_;

            // Row Vector (1×6): [n_x, n_y, n_z, (p×n)_x, (p×n)_y, (p×n)_z]
            // Column Vector (6×1): [n_x, n_y, n_z, (p×n)_x, (p×n)_y, (p×n)_z]^T

            // Compute 6x1 column vector Jacobian
            // Eigen::Vector6d J_r;
            // J_r.block<3,1>(0,0) = plane_normal_;                     // Translation part // df/dt
            // //J_r.block<3,1>(3,0) = p_transformed.cross(plane_normal_); // Rotation part   // df/dR
            // J_r.block<3,1>(3,0) = -plane_normal_.cross(p_transformed); // Note negative sign!
            // For robust version:
            // double robust_weight = mEstimator_->weight(std::abs(error));
            // *H = J_r.transpose() * (weight_ * robust_weight);
