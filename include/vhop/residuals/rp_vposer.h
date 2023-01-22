#ifndef VHOP_INCLUDE_VHOP_RESIDUALS_RP_VPOSER_H_
#define VHOP_INCLUDE_VHOP_RESIDUALS_RP_VPOSER_H_

#include <Eigen/Core>
#include <Eigen/Dense>

#include "vhop/constants.h"
#include "vhop/smpl_model.h"
#include "vposer/VPoser.h"
#include "vposer/constants.h"

namespace vhop {

class ReprojectionErrorVPoser {

public:

  ReprojectionErrorVPoser(
      const vhop::beta_t<double> &beta,
      const VPoser& vposer,
      const Eigen::Matrix3d &K,
      const Eigen::Matrix4d &T_C_B,
      const vhop::joint_op_2d_t<double> &joint_kps,
      const vhop::joint_op_scores_t &kp_scores,
      const vhop::SMPL &smpl_model)
      : beta_(beta),
        vposer_(vposer),
        T_C_B_(T_C_B),
        K_(K),
        joint_kps_(joint_kps),
        joint_kps_scores_(kp_scores),
        smpl_model_(smpl_model) {}

  bool operator()(const double *latentZ, double *reprojection_error) const {
      vposer::latent_t<double> z(latentZ);
      vhop::rotMats_t<double> rotMats = vposer_.decode(z, true);

      vhop::joint_op_2d_t<double> joints2d;
      smpl_model_.ComputeOpenPoseKP<double>(beta_, rotMats, T_C_B_, K_, &joints2d);

      for (int i = 0; i < vhop::JOINT_NUM_OP; ++i) {
          double score = joint_kps_scores_(i);
          reprojection_error[i * 2] = score * (joints2d(i, 0) - joint_kps_(i, 0));
          reprojection_error[i * 2 + 1] = score * (joints2d(i, 1) - joint_kps_(i, 1));
      }
      return true;
  }

 private:
  vhop::beta_t<double> beta_;
  VPoser vposer_;
  Eigen::Matrix3d K_;
  Eigen::Matrix4d T_C_B_;
  vhop::joint_op_2d_t<double> joint_kps_;
  vhop::joint_op_scores_t joint_kps_scores_;
  vhop::SMPL smpl_model_;
};

}  // namespace vhop

#endif //VHOP_INCLUDE_VHOP_RESIDUALS_RP_VPOSER_H_
