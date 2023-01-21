#ifndef VHOP_INCLUDE_VHOP_UTILITY_IO_H_
#define VHOP_INCLUDE_VHOP_UTILITY_IO_H_

#include <cnpy.h>
#include <Eigen/Dense>

#include "vhop/constants.h"

namespace vhop::utility {

/**
 * @brief Load one dimension of a 3D numpy array from a npz file.
 * @param npzFile Path to the npz file.
 * @param arrayName Name of the array to load.
 * @return The loaded array.
 */
Eigen::MatrixXd loadDoubleMatrix3D(const cnpy::NpyArray &raw, int r, int c, int dim = 0);

/**
 * @brief Load a 2D numpy array from a npz file.
 * @param npzFile Path to the npz file.
 * @param arrayName Name of the array to load.
 * @return The loaded array.
 */
Eigen::MatrixXd loadDoubleMatrix(const cnpy::NpyArray &raw, int r, int c);

/**
 * @brief Load a vector from a binary file.
 * @param filePath Path to binary file.
 */
std::vector<double> loadVector(const std::string& filePath);

/**
 * @brief Write a vector of doubles to a binary file.
 * @param myVector Vector of doubles.
 * @param filePath output filename.
 */
void writeVector(const std::string& filePath, const std::vector<double>& myVector);

/**
 * @brief Write SMPL parameters to a binary file.
 * @param beta SMPL shape parameters (10,).
 * @param theta SMPL pose parameters (72,).
 */
void writeSMPLParameters(const std::string& filePath,
                         const vhop::beta_t<double>& beta,
                         const vhop::theta_t<double>& theta);

/**
 * @brief Rodrigues' formula for rotation matrix from axis-angle representation.
 * @param r anlge-axis representation of a rotation.
 */
Eigen::Matrix3d rodriguesMatrix(const Eigen::Vector3d& rotVec);

/**
 * @brief Pinhole camera projection using given intrinsic parameters.
 * @param p 3D points.
 * @param K Intrinsic parameters.
 * @return 2D points.
 */
Eigen::Matrix<double, Eigen::Dynamic, 2> project(const Eigen::Matrix<double, Eigen::Dynamic, 3>& p, const Eigen::Matrix3d& K);

} // namespace vhop

#endif //VHOP_INCLUDE_VHOP_UTILITY_IO_H_
