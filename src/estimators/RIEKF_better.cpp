#include "RIEKF.hpp"

using namespace ekf;

struct landmark
{
    int map_point_index;   // index of the point from the reference map
    int cloud_point_index; // index pf the points from the cloud
    V3D norm;              // the normal of the plane in global frame (normalized)
    double d;              // d parameter of the plane
    double w;              // plane weight - should be 1 over plane measurement variance
    V3D tgt;
    double cost; // the residual p2plane cost value
};

static std::vector<landmark> global_landmarks(100000);
static std::vector<bool> global_valid(100000, false);
static std::vector<PointVector> global_Neighbours(100000);

double cauchy(double residual, double scale = 1.0)
{
    double x = residual / scale;
    return 1.0 / (1.0 + x * x);
}

double huber(double residual, double threshold = 1.0)
{
    double abs_r = std::abs(residual);
    return abs_r <= threshold ? 1.0 : threshold / abs_r;
}

void establishCorrespondences(const double R, const state &x_, const bool &update_neighbours, const PointCloudXYZI::Ptr &src_frame,
                              const PointCloudXYZI::Ptr &map, const pcl::KdTreeFLANN<PointType>::Ptr &tree)
{
    int feats_down_size = src_frame->size();

    auto travelled_distance = x_.pos.norm();

#ifdef MP_EN
    // omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body = src_frame->points[i];
        PointType point_world;

        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(x_.rot * (x_.offset_R_L_I * p_body + x_.offset_T_L_I) + x_.pos); // lidar to imu and then to world

        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        std::vector<int> pointSearchInd(NUM_MATCH_POINTS);
        std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = global_Neighbours[i];
        std::vector<double> point_weights;
        if (update_neighbours) // update the nearest neighbours
        {
            if (tree->nearestKSearch(point_world, NUM_MATCH_POINTS, pointSearchInd, pointSearchSqDis) >= NUM_MATCH_POINTS)
            {
                global_valid[i] = pointSearchSqDis[NUM_MATCH_POINTS - 1] > 1 ? false : true; // true if distance smaller than 1
                points_near.clear();
                for (int j = 0; j < pointSearchInd.size(); j++)
                {
                    // points_near.push_back(map->points[pointSearchInd[j]]);
                    points_near.insert(points_near.begin(), map->points[pointSearchInd[j]]);
                }
            }
            else
            {
                global_valid[i] = false;
            }
        }

        if (!global_valid[i])
            continue;

        Eigen::Matrix<float, 4, 1> pabcd; // plane coefficients [nx, ny, nz, d]
        global_valid[i] = false;
        double plane_var = 1.;
        // if (ekf::esti_plane(pabcd, points_near, .1))
        if (ekf::esti_plane_pca(pabcd, points_near, .03, point_weights, plane_var, false))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            // float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());
            float s = 1 - 0.9 * fabs(pd2) / point_body.intensity;

            if (s > 0.9)
            {
                const auto &p_tgt = points_near[0];
                landmark l;
                l.map_point_index = pointSearchInd[0];
                l.cloud_point_index = i;
                l.norm = V3D_4(pabcd(0), pabcd(1), pabcd(2));
                l.d = pabcd(3);

                double base_weight = 1 / R;
                if (travelled_distance > 5)
                    base_weight = 1 / plane_var;

                double kernel_weight = huber(pd2, 0.1); // cauchy(pd2, 0.2);
                l.w = base_weight * kernel_weight;

                l.tgt = V3D(p_tgt.x, p_tgt.y, p_tgt.z);
                l.cost = pd2;

                global_valid[i] = true;
                global_landmarks[i] = l;
            }
        }
    }
}

using Jacobian_plane = Eigen::Matrix<double, 1, state_size>;
struct ResultTuple
{
    ResultTuple()
    {
        JTJ.setZero();
        JTr.setZero();
        squared_residual = 0;
        points = 0;
    }

    ResultTuple operator+(const ResultTuple &other)
    {
        this->JTJ += other.JTJ;
        this->JTr += other.JTr;
        this->squared_residual += other.squared_residual;
        this->points += other.points;
        return *this;
    }

    cov JTJ;              // state_size x state_size
    vectorized_state JTr; // state_size x 1
    double squared_residual;
    int points;
};

struct AvgValue
{
    AvgValue()
    {
        squared_residual = 0.;
        points = 0.;
    }

    AvgValue operator+(const AvgValue &other)
    {
        this->squared_residual += other.squared_residual;
        this->points += other.points;
        return *this;
    }

    double squared_residual;
    int points;
};

std::tuple<cov, vectorized_state, double> BuildLinearSystem_tbb(const state &x_, const PointCloudXYZI::Ptr &src_frame, const bool extrinsic_est)
{
    const int feats_down_size = src_frame->size();
    // const auto &[JTJ, JTr]
    const ResultTuple &result = tbb::parallel_reduce(
        // Range
        tbb::blocked_range<size_t>{0, feats_down_size},
        // Identity
        ResultTuple(),
        // 1st Lambda: Parallel computation
        [&](const tbb::blocked_range<size_t> &r, ResultTuple J) -> ResultTuple
        {
            auto &[JTJ_private, JTr_private, squared_residual_private, points_private] = J;
            for (auto i = r.begin(); i < r.end(); ++i)
            {
                if (global_valid[i])
                {
                    V3D point_(src_frame->points[i].x,
                               src_frame->points[i].y,
                               src_frame->points[i].z);

                    M3D point_crossmat;
                    point_crossmat << SKEW_SYM_MATRX(point_);

                    // Transform point to IMU frame
                    V3D point_I_ = x_.offset_R_L_I * point_ + x_.offset_T_L_I;
                    M3D point_I_crossmat;
                    point_I_crossmat << SKEW_SYM_MATRX(point_I_);

                    const landmark &lm = global_landmarks[i];

                    // Norm of plane expressed in world -> rotate into IMU frame
                    V3D C = x_.rot.matrix().transpose() * lm.norm;
                    V3D A = point_I_crossmat * C;

                    Jacobian_plane J_r = Jacobian_plane::Zero(); // 1x6 or more
                    J_r.block<1, 3>(0, P_ID) = lm.norm.transpose();
                    J_r.block<1, 3>(0, R_ID) = A.transpose();

                    if (extrinsic_est)
                    {
                        V3D B(point_crossmat * x_.offset_R_L_I.matrix().transpose() * C);
                        J_r.block<1, 3>(0, Re_ID) = B.transpose();
                        J_r.block<1, 3>(0, Te_ID) = C.transpose();
                    }

                    // r(k) = z − h(x(k)), z is zero for p2plane
                    double residual = -lm.cost; // (distance to plane) negative here due to derivative chain rule

                    // Robust weighting
                    // double w = 1.0; // todo - add a robust kernel here

                    JTJ_private.noalias() += J_r.transpose() * lm.w * J_r;
                    JTr_private.noalias() += J_r.transpose() * lm.w * residual;
                    squared_residual_private += lm.w * residual * residual;
                    points_private += 1;
                }
            }
            return J;
        },
        // 2nd Lambda: Parallel reduction of the private Jacboians
        [&](ResultTuple a, const ResultTuple &b) -> ResultTuple
        { return a + b; });

    const auto &[JTJ, JTr, squared_residual, n_points] = result;
    int total_points = 1; // std::max(n_points, 1);

    std::cout << "Registration with:" << n_points << "/" << feats_down_size << std::endl;

    return std::make_tuple(JTJ, JTr, squared_residual / total_points);
}

double estimateCost_tbb(const state &x_, const PointCloudXYZI::Ptr &src_frame)
{
    const int feats_down_size = src_frame->size();

    const AvgValue &result = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, feats_down_size),
        AvgValue(),
        [&](const tbb::blocked_range<int> &r, AvgValue local_cost) -> AvgValue
        {
            auto &[squared_residual_private, points_private] = local_cost;
            for (int i = r.begin(); i < r.end(); ++i)
            {
                if (!global_valid[i])
                    continue;

                const PointType &point_body = src_frame->points[i];
                const landmark &l = global_landmarks[i];

                // Transform point to world
                V3D p_body(point_body.x, point_body.y, point_body.z);
                V3D p_global = x_.rot * (x_.offset_R_L_I * p_body + x_.offset_T_L_I) + x_.pos;

                double residual = l.norm.dot(p_global) + l.d;

                squared_residual_private += l.w * residual * residual;
                points_private += 1;
            }
            return local_cost;
        },
        // 2nd Lambda: Parallel reduction of the private sums
        [&](AvgValue a, const AvgValue &b) -> AvgValue
        { return a + b; });

    const auto &[squared_residual, n_points] = result;
    int total_points = 1; // std::max(n_points, 1);
    return squared_residual / total_points;
}

int RIEKF::update_(double R, PointCloudXYZI::Ptr &feats_down_body, PointCloudXYZI::Ptr &map, int maximum_iter, bool extrinsic_est)
{
    std::cout << "Running standard update_ function" << std::endl;

    residual_struct status;
    status.valid = true;
    status.converge = true;
    int t = 0;

    state x_propagated = x_;
    vectorized_state dx_new = vectorized_state::Zero(); // 24X1

    if (map->points.size() < 5)
    {
        std::cerr << "Error: map Point cloud is empty! : " << map->points.size() << std::endl;
        status.valid = false;
        status.converge = false;
        return false;
    }

    localKdTree_map->setInputCloud(map);

    int iteration_finished = 0;
    const cov P_inv = P_.inverse(); // 24x24

    for (int i = -1; i < maximum_iter; i++) // maximum_iter
    {
        iteration_finished = i + 1;
        status.valid = true;
        establishCorrespondences(R, x_, status.converge, feats_down_body, map, localKdTree_map);

        // JTJ = J^T W J; JTr = J^T W r
        const auto &[JTJ, JTr, current_cost] = BuildLinearSystem_tbb(x_, feats_down_body, extrinsic_est); // faster
        // std::cout << "sq_residual:" << current_cost << ", lambda:" << lambda << std::endl;

        if (!status.valid)
            break;

        vectorized_state dx;
        cov H = (JTJ + P_inv); // matrix H = JTJ + P_inv (+ lambda * H.diagonal().asDiagonal())  //JTJ - 24 x 24  JTr - 24 x 1      H += lambda * H.diagonal().asDiagonal();

        vectorized_state dx_; // vectorized_state dx_ = K * status.innovation + (KJ - cov::Identity()) * boxminus(x_, x_propagated);
        cov KJ = cov::Zero(); //  matrix K * H

        auto H_inv = H.inverse(); // 24x24
        KJ = H_inv * JTJ;
        // dx_.noalias() = H_inv * JTr;       //slower
        dx_.noalias() = H.ldlt().solve(JTr);                                  // faster
        dx_.noalias() += (KJ - cov::Identity()) * boxminus(x_, x_propagated); // iterated error state kalmna filter part

        // Eigen::LDLT<cov> ldltSolver(H);
        // dx_.noalias() = ldltSolver.solve(JTr); // Solve for dx_
        // KJ.noalias() = ldltSolver.solve(JTJ);  // Solve for KJ (multiple right-hand sides)
        // dx_.noalias() += (KJ - cov::Identity()) * boxminus(x_, x_propagated);

        x_ = boxplus(x_, dx_); // GN

        status.converge = true;
        for (int j = 0; j < state_size; j++)
        {
            if (std::fabs(dx_[j]) > ESTIMATION_THRESHOLD_) // If dx_>ESTIMATION_THRESHOLD_ = 0.001 no convergence is considered
            {
                status.converge = false;
                break;
            }
        }

        if (status.converge)
            t++;

        if (!t && i == maximum_iter - 2)
        {
            status.converge = true;
        }

        if (t > 1 || i == maximum_iter - 1)
        {
            // P_ = (cov::Identity() - KJ) * P_;// same as P_ = P_ - KJ*P_ = P_-= KJ*P_
            P_.noalias() -= KJ * P_;
            break;
        }
    }

    std::cout << "update_final in " << iteration_finished << "/" << maximum_iter << " iterations " << std::endl;

    return iteration_finished;
}