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

typedef Eigen::Matrix<double, SHAPE_BASIS_DIM, 1> beta_t;
typedef AlignedVector<beta_t> betas_t;

typedef Eigen::Matrix<double, JOINT_NUM * 3, 1> theta_t;
typedef AlignedVector<theta_t> thetas_t;

typedef Eigen::Matrix<double, 3, 1> translation_t;
typedef AlignedVector<translation_t> translations_t;

typedef Eigen::Matrix<double, JOINT_NUM * 3, 1> joint_t;
typedef AlignedVector<joint_t> joints_t;

typedef Eigen::Matrix<double, JOINT_NUM_OP * 3, 1> op_joint_t;
typedef AlignedVector<op_joint_t> op_joints_t;

typedef Eigen::Matrix<double, JOINT_NUM_OP * 2, 1> op_kp_t;
typedef AlignedVector<op_kp_t> op_kps_t;

typedef Eigen::Matrix<double, JOINT_NUM_OP, 1> op_conf_t;
typedef AlignedVector<op_conf_t> op_confs_t;

typedef Eigen::Matrix<double, JOINT_NUM_EXTRA * 3, 1> vertex_t;
typedef AlignedVector<vertex_t> vertexs_t;

} // namespace vhop

#endif //VHOP_INCLUDE_VHOP_CONSTANTS_H_
