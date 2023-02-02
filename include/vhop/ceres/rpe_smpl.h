#ifndef VHOP_INCLUDE_VHOP_CERES_RPE_SMPL_H_
#define VHOP_INCLUDE_VHOP_CERES_RPE_SMPL_H_

#include <utility>

#include "vhop/constants.h"
#include "vhop/utility.h"
#include "base_rpe_residual.h"

namespace vhop {

class ReprojectionErrorSMPL : public vhop::ResidualBase {

 public:

  ReprojectionErrorSMPL(const std::string &dataFilePath, const vhop::SMPL &smpl_model)
      : ResidualBase(), smpl_model_(smpl_model) {
      cnpy::npz_t npz = cnpy::npz_load(dataFilePath);
      beta_ = vhop::utility::loadDoubleMatrix(npz.at("betas"), vhop::SHAPE_BASIS_DIM, 1);
      K_ = vhop::utility::loadDoubleMatrix(npz.at("intrinsics"), 3, 3);
      T_C_B_ = vhop::utility::loadDoubleMatrix(npz.at("T_C_B"), 4, 4);
      joint_kps_ = vhop::utility::loadDoubleMatrix(npz.at("keypoints_2d"), vhop::JOINT_NUM_OP, 2);
      joint_kps_scores_ = vhop::utility::loadDoubleMatrix(npz.at("keypoints_2d_scores"), vhop::JOINT_NUM_OP, 1);
  }

  ReprojectionErrorSMPL(
      const vhop::beta_t<double>& beta,
      const Eigen::Matrix3d& K,
      const Eigen::Matrix4d& T_C_B,
      const vhop::joint_op_2d_t<double>& joint_kps,
      const vhop::joint_op_scores_t& kp_scores,
      const vhop::SMPL& smpl_model)
      : ResidualBase(),
      beta_(beta),
      smpl_model_(smpl_model),
      K_(K),
      T_C_B_(T_C_B),
      joint_kps_(joint_kps),
      joint_kps_scores_(kp_scores){};

  bool operator()(const double *poseData, double *reprojection_error) const override {
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

  Eigen::VectorXd x0() override {
      // cnpy::npz_t npz_means = cnpy::npz_load("../data/vposer_mean.npz");
      return vhop::theta_t<double>::Zero();
  }

  void convert2SMPL(const Eigen::VectorXd& z, beta_t<double>& beta, theta_t<double> theta) {
      beta = beta_;
      theta = z;
  }

    bool drawReProjections(const Eigen::VectorXd& z, const std::string& imagePath, const std::string& outputImagePath) {
        beta_t<double> beta;
        theta_t<double> theta;
        convert2SMPL(z, beta, theta);

        vhop::joint_op_2d_t<double> joints_2d;
        smpl_model_.ComputeOpenPoseKP(beta, theta, T_C_B_, K_, &joints_2d);

        vhop::visualization::drawKeypoints(imagePath,
                                           joints_2d.cast<int>(),
                                           joint_kps_.cast<int>(),
                                           outputImagePath);
        return true;
    }

  static constexpr int getNumParams() { return vhop::THETA_DIM; }
  static constexpr int getNumResiduals() { return vhop::JOINT_NUM_OP * 2; }

 private:
    vhop::beta_t<double> beta_;
    vhop::SMPL smpl_model_;
    Eigen::Matrix3d K_;
    Eigen::Matrix4d T_C_B_;
    vhop::joint_op_2d_t<double> joint_kps_;
    vhop::joint_op_scores_t joint_kps_scores_;
};

}

#endif //VHOP_INCLUDE_VHOP_CERES_RPE_SMPL_H_
