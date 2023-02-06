#include "vhop/visualization.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>


void vhop::visualization::drawKeypoints(const cv::Mat& image,
                                        const Eigen::Matrix<int, Eigen::Dynamic, 2> &keypoints,
                                        const cv::Scalar &color) {
  for (int i = 0; i < keypoints.rows(); i++) {
        cv::Point center(keypoints(i, 0), keypoints(i, 1));
        cv::circle(image, center, 2, color, -1);
    }
}

void vhop::visualization::drawKeypoints(const std::string& image_file_name,
                                        const Eigen::Matrix<int, Eigen::Dynamic, 2>& keypoints,
                                        const std::string& output_file_name,
                                        const cv::Scalar &color) {
    cv::Mat image = cv::imread(image_file_name);
    drawKeypoints(image, keypoints, color);
    cv::imwrite(output_file_name, image);
}

void vhop::visualization::drawKeypoints(const std::string& image_file_name,
                                        const Eigen::Matrix<int, Eigen::Dynamic, 2>& pred_keypoints,
                                        const Eigen::Matrix<int, Eigen::Dynamic, 2>& gt_keypoints,
                                        const std::string& output_file_name) {
    cv::Mat image = cv::imread(image_file_name);
    drawKeypoints(image, gt_keypoints, cv::Scalar(0, 255, 0));
    drawKeypoints(image, pred_keypoints, cv::Scalar(0, 0, 255));
    cv::imwrite(output_file_name, image);
}
