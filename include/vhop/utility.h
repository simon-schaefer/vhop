#ifndef VHOP_INCLUDE_VHOP_UTILITY_IO_H_
#define VHOP_INCLUDE_VHOP_UTILITY_IO_H_

#include <cnpy.h>
#include <Eigen/Dense>

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
 * @brief Rodrigues' formula for rotation matrix from axis-angle representation.
 * @param r anlge-axis representation of a rotation.
 */
Eigen::Matrix3d rodriguesMatrix(const Eigen::Vector3d& rotVec);

} // namespace vhop

#endif //VHOP_INCLUDE_VHOP_UTILITY_IO_H_
