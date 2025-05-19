#pragma once

#include "DataHandler_vux.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <deque>
#include <memory>

#include <sensor_msgs/Imu.h>
#include <gps_common/GPSFix.h>


using namespace gtsam;
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

class GnssImuFusion
{
public:
    GnssImuFusion(const Sophus::SE3 &als2mls, const imuBias::ConstantBias &priorBias,
                  const Vector6 &posePriorVar, const Vector3 &velPriorVar,
                  const Vector6 &biasPriorVar)
        : als2mls_(als2mls), priorBias_(priorBias)
    {

        // Set up noise models
        posePriorNoise_ = noiseModel::Diagonal::Sigmas(posePriorVar);
        velPriorNoise_ = noiseModel::Diagonal::Sigmas(velPriorVar);
        biasPriorNoise_ = noiseModel::Diagonal::Sigmas(biasPriorVar);

        // Set up IMU preintegration
        auto imuParams = PreintegrationCombinedParams::MakeSharedU();
        // Set IMU parameters (these should be tuned for your specific IMU)
        imuParams->accelerometerCovariance = Matrix3::Identity() * pow(0.1, 2); // accel noise
        imuParams->gyroscopeCovariance = Matrix3::Identity() * pow(0.01, 2);    // gyro noise
        imuParams->integrationCovariance = Matrix3::Identity() * pow(1e-4, 2);  // integration noise
        imuParams->biasAccCovariance = Matrix3::Identity() * pow(0.1, 2);       // acc bias random walk
        imuParams->biasOmegaCovariance = Matrix3::Identity() * pow(0.01, 2);    // gyro bias random walk
        imuParams->biasAccOmegaInt = Matrix::Identity(6, 6) * 1e-5;             // bias integration

        currentSummarizedMeasurement_ = std::make_shared<PreintegratedCombinedMeasurements>(imuParams, priorBias_);

        // Initialize the graph
        graph_ = std::make_shared<NonlinearFactorGraph>();

        // Add prior factors for the first state
        graph_->addPrior(X(0), Pose3(), posePriorNoise_);
        graph_->addPrior(V(0), Vector3::Zero(), velPriorNoise_);
        graph_->addPrior(B(0), priorBias_, biasPriorNoise_);

        // Initialize values
        values_ = std::make_shared<Values>();
        values_->insert(X(0), Pose3());
        values_->insert(V(0), Vector3::Zero());
        values_->insert(B(0), priorBias_);

        currentPose_ = Pose3();
        currentVelocity_ = Vector3::Zero();
        currentBias_ = priorBias_;
        key_ = 1;
    }

    void processImu(const sensor_msgs::Imu::ConstPtr &imu)
    {
        // Convert IMU message to GTSAM format
        Vector3 measuredAcc(imu->linear_acceleration.x,
                            imu->linear_acceleration.y,
                            imu->linear_acceleration.z);
        Vector3 measuredOmega(imu->angular_velocity.x,
                              imu->angular_velocity.y,
                              imu->angular_velocity.z);

        double deltaT = imu->header.stamp.toSec() - lastImuTime_;
        if (deltaT <= 0)
        {
            lastImuTime_ = imu->header.stamp.toSec();
            return;
        }

        // Integrate the IMU measurement
        currentSummarizedMeasurement_->integrateMeasurement(measuredAcc, measuredOmega, deltaT);

        lastImuTime_ = imu->header.stamp.toSec();
    }

    void processGnss(const gps_common::GPSFix::ConstPtr &gps)
    {
        // Convert GNSS position to GTSAM format (in global frame)
        Point3 globalPosition(gps->latitude, gps->longitude, gps->altitude);

        // Transform to local frame using als2mls
        Sophus::SE3 globalPose = Sophus::SE3(Sophus::SO3d(), globalPosition);
        Sophus::SE3 localPose = als2mls_ * globalPose;

        Pose3 gtsamLocalPose(localPose.rotationMatrix(), localPose.translation());

        // Add IMU factor for the current state
        auto imuFactor = std::make_shared<CombinedImuFactor>(
            X(key_ - 1), V(key_ - 1), X(key_), V(key_), B(key_ - 1), B(key_), *currentSummarizedMeasurement_);
        graph_->add(imuFactor);

        // Add GNSS factor
        auto gpsNoise = noiseModel::Diagonal::Sigmas(Vector3(1.0, 1.0, 2.0)); // Tune these values
        auto gpsFactor = std::make_shared<GPSFactor>(X(key_), gtsamLocalPose.translation(), gpsNoise);
        graph_->add(gpsFactor);

        // Predict the new state
        NavState prevState(currentPose_, currentVelocity_);
        NavState propState = currentSummarizedMeasurement_->predict(prevState, currentBias_);

        // Insert predicted values
        values_->insert(X(key_), propState.pose());
        values_->insert(V(key_), propState.v());
        values_->insert(B(key_), currentBias_);

        // Optimize the graph
        LevenbergMarquardtOptimizer optimizer(*graph_, *values_);
        values_ = std::make_shared<Values>(optimizer.optimize());

        // Update current state
        currentPose_ = values_->at<Pose3>(X(key_));
        currentVelocity_ = values_->at<Vector3>(V(key_));
        currentBias_ = values_->at<imuBias::ConstantBias>(B(key_));

        // Reset the preintegration object for the next step
        auto imuParams = currentSummarizedMeasurement_->params();
        currentSummarizedMeasurement_ = std::make_shared<PreintegratedCombinedMeasurements>(
            imuParams, currentBias_);

        key_++;
    }

    void runFusion(const std::deque<sensor_msgs::Imu::ConstPtr> &imu_buffer,
                   const std::deque<gps_common::GPSFix::ConstPtr> &gps_buffer)
    {
        // Assuming buffers are time-synchronized and ordered
        auto imu_it = imu_buffer.begin();
        auto gps_it = gps_buffer.begin();

        while (imu_it != imu_buffer.end() && gps_it != gps_buffer.end())
        {
            if ((*imu_it)->header.stamp < (*gps_it)->header.stamp)
            {
                processImu(*imu_it);
                ++imu_it;
            }
            else
            {
                processGnss(*gps_it);
                ++gps_it;
            }
        }

        // Process remaining IMU messages
        while (imu_it != imu_buffer.end())
        {
            processImu(*imu_it);
            ++imu_it;
        }
    }

    Pose3 getCurrentPose() const { return currentPose_; }
    Vector3 getCurrentVelocity() const { return currentVelocity_; }
    imuBias::ConstantBias getCurrentBias() const { return currentBias_; }

private:
    Sophus::SE3 als2mls_;
    imuBias::ConstantBias priorBias_;

    std::shared_ptr<PreintegratedCombinedMeasurements> currentSummarizedMeasurement_;
    std::shared_ptr<NonlinearFactorGraph> graph_;
    std::shared_ptr<Values> values_;

    noiseModel::Diagonal::shared_ptr posePriorNoise_;
    noiseModel::Diagonal::shared_ptr velPriorNoise_;
    noiseModel::Diagonal::shared_ptr biasPriorNoise_;

    Pose3 currentPose_;
    Vector3 currentVelocity_;
    imuBias::ConstantBias currentBias_;

    double lastImuTime_ = 0;
    size_t key_ = 0;
};