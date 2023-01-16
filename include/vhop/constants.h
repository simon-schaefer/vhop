#ifndef VHOP_INCLUDE_VHOP_CONSTANTS_H_
#define VHOP_INCLUDE_VHOP_CONSTANTS_H_

#include <Eigen/Dense>
#include <vector>


namespace vhop {

constexpr size_t VERTEX_NUM = 6890;  // 6890
constexpr size_t JOINT_NUM = 24;// 24
constexpr size_t JOINT_NUM_EXTRA = 21;// 9 for joint regressor
constexpr size_t JOINT_NUM_OP = 25;// 25 openpose joints
constexpr size_t SHAPE_BASIS_DIM = 10;// 10
constexpr size_t POSE_BASIS_DIM = 207;// 207
constexpr size_t THETA_DIM = JOINT_NUM * 3;
const std::vector<int> VERTEX_JOINT_IDXS = {
    332, 6260, 2800, 4071, 583, 3216, 3226, 3387, 6617, 6624, 6787, 2746,
    2319, 2445, 2556, 2673, 6191, 5782, 5905, 6016, 6133};

template<typename VALUE_T>
using AlignedVector = std::vector<VALUE_T, Eigen::aligned_allocator<VALUE_T>>;
template<typename T>
using beta_t = Eigen::Matrix<T, SHAPE_BASIS_DIM, 1>;
template<typename T>
using theta_t = Eigen::Matrix<T, JOINT_NUM * 3, 1>;
template<typename T>
using translation_t = Eigen::Matrix<T, 3, 1>;
template<typename T>
using joint_t = Eigen::Matrix<T, JOINT_NUM * 3, 1>;
template<typename T>
using joint_kp_t = Eigen::Matrix<T, 2, JOINT_NUM>;
template<typename T>
using vertex_t = Eigen::Matrix<T, JOINT_NUM_EXTRA * 3, 1>;

} // namespace vhop

#endif //VHOP_INCLUDE_VHOP_CONSTANTS_H_
