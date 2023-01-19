#ifndef VHOP_REPROJECTION_ERROR_H
#define VHOP_REPROJECTION_ERROR_H

#include <Eigen/Core>
#include <Eigen/Dense>

#include "vhop/constants.h"
#include "vhop/smpl_model.h"


namespace vhop {

class ReprojectionError {

public:

    ReprojectionError(
            const vhop::beta_t<double> &beta,
            const Eigen::Matrix3d &K,
            const Eigen::Matrix4d &T_C_B,
            const vhop::joint_op_2d_t<double> &joint_kps,
            const vhop::SMPL &smpl_model)
            : beta_(beta), T_C_B_(T_C_B), K_(K), joint_kps_(joint_kps), smpl_model_(smpl_model) {}

    bool operator()(const double *poseData, double *reprojection_error) const {
      vhop::theta_t<double> poses(poseData);
      vhop::joint_op_2d_t<double> joints2d = computeJoints2D(poses);
      reprojection_error[0] = (joint_kps_ - joints2d).rowwise().norm().mean();
      return true;
    }

    template <typename T>
    vhop::joint_op_2d_t<T> computeJoints2D(const vhop::theta_t<T> &poses) const {
      vhop::joint_op_3d_t<T> joints3dFlat;
      smpl_model_.ForwardOpenPose<T>(beta_, poses, &joints3dFlat);
      Eigen::Matrix<T, vhop::JOINT_NUM_OP, 3> joints3d_B = joints3dFlat.reshaped(3, vhop::JOINT_NUM_OP).transpose();

      Eigen::Matrix<T, vhop::JOINT_NUM_OP, 3> joints3d_C;
      for (int i = 0; i < vhop::JOINT_NUM_OP; i++) {
        Eigen::Vector<T, 4> joints3d_Bh_i(joints3d_B(i, 0), joints3d_B(i, 1), joints3d_B(i, 2), 1);
        Eigen::Vector4d joints3d_Ch_i = T_C_B_ * joints3d_Bh_i;
        joints3d_C.row(i) = joints3d_Ch_i.head(3).cast<T>();
      }

      return vhop::utility::project(joints3d_C, K_);
    }

private:
    Eigen::Matrix3d K_;
    Eigen::Matrix4d T_C_B_;
    vhop::beta_t<double> beta_;
    vhop::joint_op_2d_t<double> joint_kps_;
    vhop::SMPL smpl_model_;
};

}  // namespace vhop

#endif //VHOP_REPROJECTION_ERROR_H
