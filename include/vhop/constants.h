#ifndef VHOP_INCLUDE_VHOP_CONSTANTS_H_
#define VHOP_INCLUDE_VHOP_CONSTANTS_H_

#include <Eigen/Dense>
#include <vector>

namespace vhop {

constexpr size_t VERTEX_NUM = 6890;
constexpr size_t JOINT_NUM = 24;
constexpr size_t JOINT_NUM_EXTRA = 21;
constexpr size_t JOINT_NUM_TOTAL = JOINT_NUM + JOINT_NUM_EXTRA;
constexpr size_t JOINT_NUM_OP = 25;
constexpr size_t SHAPE_BASIS_DIM = 10;
constexpr size_t POSE_BASIS_DIM = 207;
constexpr size_t THETA_DIM = JOINT_NUM * 3;
const std::vector<int> VERTEX_JOINT_IDXS = {
    332, 6260, 2800, 4071, 583, 3216, 3226, 3387, 6617, 6624, 6787, 2746,
    2319, 2445, 2556, 2673, 6191, 5782, 5905, 6016, 6133};

// Joint conversion from SMPL to OpenPose.
// https://files.is.tue.mpg.de/black/talks/SMPL-made-simple-FAQs.pdf
// https://note.com/appai/n/nfe5d6bbd7bb2
const std::vector<int> OPENPOSE_JOINT_INDEXES = {
  24, 12, 17, 19, 21, 16, 18, 20,  0,  2,  5,  8,  1,  4,  7, 25, 26, 27,
  28, 29, 30, 31, 32, 33, 34};

template<typename VALUE_T>
using AlignedVector = std::vector<VALUE_T, Eigen::aligned_allocator<VALUE_T>>;
template<typename T>
using beta_t = Eigen::Matrix<T, SHAPE_BASIS_DIM, 1>;
template<typename T>
using theta_t = Eigen::Matrix<T, THETA_DIM, 1>;
template<typename T>
using rotMats_t = AlignedVector<Eigen::Matrix<T, 3, 3>>;
template<typename T>
using translation_t = Eigen::Matrix<T, 3, 1>;
template<typename T>
using joint_t = Eigen::Matrix<T, JOINT_NUM * 3, 1>;
template<typename T>
using joint_op_3d_t = Eigen::Matrix<T, JOINT_NUM_OP * 3, 1>;
template<typename T>
using joint_op_2d_t = Eigen::Matrix<T, JOINT_NUM_OP, 2>;
using joint_op_scores_t = Eigen::Matrix<double, JOINT_NUM_OP, 1>;
template<typename T>
using vertex_t = Eigen::Matrix<T, JOINT_NUM_EXTRA * 3, 1>;


} // namespace vhop

#endif //VHOP_INCLUDE_VHOP_CONSTANTS_H_
