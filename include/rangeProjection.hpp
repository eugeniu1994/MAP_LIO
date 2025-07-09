
#include "utils.h"
#include <opencv2/opencv.hpp>

// config of the hesai
const int N_SCAN = 32;         // number of vertical channels
const int Horizon_SCAN = 1500; // 2000 360° / ang_res_x° resolution
const float ang_res_x = 0.18f; // horizontal angular resolution in degrees
const float ang_res_y = 1.;

const float ang_bottom = 15.0; // 360° x 31° Field of View put mid 31/2
// ang_bottom is the vertical angle corresponding to the bottom-most laser ring (lowest vertical FOV limit)
// ang_bottom = last ang_res_y  = vertical_fov / (N_SCAN - 1)​


void projectToRangeImage(const pcl::PointCloud<PointType>::Ptr &cloud, Eigen::MatrixXi &index_out)
{
    // Init with -1 (meaning no point projected at this pixel yet)
    index_out.resize(N_SCAN, Horizon_SCAN);
    // index_out.setZero();
    index_out.setConstant(-1);

    cv::Mat show_image, range_image = cv::Mat::zeros(N_SCAN, Horizon_SCAN, CV_8U);
    for (int idx = 0; idx < cloud->points.size(); idx++)
    {
        const auto &pt = cloud->points[idx];
        float range = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
        if (range < 0.1)
            continue; // skip invalid points

        int row = pt.ring; // from ring
        float verticalAngle = atan2(pt.z, sqrt(pt.x * pt.x + pt.y * pt.y)) * 180 / M_PI;
        row = (verticalAngle + ang_bottom) / ang_res_y;

        if (row < 0 || row >= N_SCAN)
            continue;

        float horizonAngle = atan2(pt.x, pt.y) * 180 / M_PI;
        int col = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN / 2;

        if (col >= Horizon_SCAN)
            col -= Horizon_SCAN;
        if (col < 0 || col >= Horizon_SCAN)
            continue;

        // std::cout<<"row:"<<row<<", col:"<<col<<", range:"<<range<<std::endl;

        // Keep closest point in case of overlaps
        // if (range < range_image.at<float>(row, col))
        //     range_image.at<float>(row, col) = range;

        range_image.at<uchar>(row, col) = (range / 30) * 255;
        index_out(row, col) = idx; // index of the point from the original cloud
    }

    cv::applyColorMap(range_image, show_image, cv::COLORMAP_JET);
    // cv::flip(show_image, show_image, 0);
    cv::imshow("Range Image", show_image);
    cv::waitKey(1);

    //cv::waitKey(0);
}

