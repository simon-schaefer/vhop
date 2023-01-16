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
            const vhop::translation_t<double> &translation,
            const Eigen::Matrix3d &K,
            const vhop::joint_op_2d_t<double> &joint_kps,
            const vhop::SMPL &smpl_model)
            : beta_(beta), translation_(translation), K_(K), joint_kps_(joint_kps), smpl_model_(smpl_model) {}

    bool operator()(const double *poseData, double *reprojection_error) const {
      vhop::theta_t<double> poses(poseData);
      reprojection_error[0] = evaluate(poses);
      return true;
    }

    double evaluate(const vhop::theta_t<double> &poses) const {
      vhop::joint_op_3d_t<double> joints3dFlat;
      smpl_model_.ForwardOpenPose<double>(beta_, poses, translation_, &joints3dFlat);
      Eigen::Matrix<double, vhop::JOINT_NUM_OP, 3> joints3d = joints3dFlat.reshaped(vhop::JOINT_NUM_OP, 3);

      vhop::joint_op_2d_t<double> joints2d = vhop::utility::project(joints3d, K_);
      return (joint_kps_ - joints2d).squaredNorm();
    }

private:
    Eigen::Matrix3d K_;
    vhop::beta_t<double> beta_;
    vhop::translation_t<double> translation_;
    vhop::joint_op_2d_t<double> joint_kps_;
    vhop::SMPL smpl_model_;
};

}  // namespace vhop

#endif //VHOP_REPROJECTION_ERROR_H
