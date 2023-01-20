#ifndef VHOP_VISUALIZATION_H
#define VHOP_VISUALIZATION_H

#include <string>
#include "Eigen/Dense"
#include <opencv2/core/core.hpp>


namespace vhop::visualization {

/**
 * @brief Draw keypoints on an image.
 * @param image Image to draw on.
 * @param keypoints Keypoints to draw.
 * @param color Color of the keypoints.
 */
void drawKeypoints(const cv::Mat& image,
                   const Eigen::Matrix<int, Eigen::Dynamic, 2> &keypoints,
                   const cv::Scalar &color = cv::Scalar(0, 0, 255));

/**
 * @brief Load image, draw 2D keypoints and save it.
 * @param image_file_name: image file name.
 * @param keypoints: 2D keypoints, (N, 2).
 * @param output_file_name: output file name.
 */
void drawKeypoints(const std::string &image_file_name,
                   const Eigen::Matrix<int, Eigen::Dynamic, 2> &keypoints,
                   const std::string &output_file_name,
                   const cv::Scalar &color = cv::Scalar(0, 0, 255));

/**
 * @brief Draw prediction and ground-truth 2D keypoints in an image and save it.
 * @param image_file_name: image file name.
 * @param pred_keypoints: 2D keypoints, (N, 2).
 * @param gt_keypoints: 2D keypoints, (N, 2).
 * @param output_file_name: output file name.
 */
void drawKeypoints(const std::string &image_file_name,
                   const Eigen::Matrix<int, Eigen::Dynamic, 2> &pred_keypoints,
                   const Eigen::Matrix<int, Eigen::Dynamic, 2> &gt_keypoints,
                   const std::string &output_file_name);
}

#endif //VHOP_VISUALIZATION_H
