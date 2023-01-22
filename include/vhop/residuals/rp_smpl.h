#ifndef VHOP_INCLUDE_VHOP_RESIDUALS_RP_SMPL_H_
#define VHOP_INCLUDE_VHOP_RESIDUALS_RP_SMPL_H_

#include <Eigen/Core>
#include <Eigen/Dense>

#include "vhop/constants.h"
#include "vhop/smpl_model.h"


namespace vhop {

class ReprojectionErrorSMPL {

public:

    ReprojectionErrorSMPL(
            const vhop::beta_t<double> &beta,
            const Eigen::Matrix3d &K,
            const Eigen::Matrix4d &T_C_B,
            const vhop::joint_op_2d_t<double> &joint_kps,
            const vhop::joint_op_scores_t &kp_scores,
            const vhop::SMPL &smpl_model)
    : beta_(beta), T_C_B_(T_C_B), K_(K), joint_kps_(joint_kps), joint_kps_scores_(kp_scores), smpl_model_(smpl_model) {}

    bool operator()(const double *poseData, double *reprojection_error) const {
      vhop::theta_t<double> poses(poseData);
      vhop::joint_op_2d_t<double> joints2d;
      smpl_model_.ComputeOpenPoseKP<double>(beta_, poses, T_C_B_, K_, &joints2d);

      for (int i = 0; i < vhop::JOINT_NUM_OP; ++i) {
          double score = joint_kps_scores_(i);
          reprojection_error[i * 2] = score * (joints2d(i, 0) - joint_kps_(i, 0));
          reprojection_error[i * 2 + 1] = score * (joints2d(i, 1) - joint_kps_(i, 1));
      }
      return true;
    }

private:
    Eigen::Matrix3d K_;
    Eigen::Matrix4d T_C_B_;
    vhop::beta_t<double> beta_;
    vhop::joint_op_2d_t<double> joint_kps_;
    vhop::joint_op_scores_t joint_kps_scores_;
    vhop::SMPL smpl_model_;
};

}  // namespace vhop

#endif //VHOP_INCLUDE_VHOP_RESIDUALS_RP_SMPL_H_