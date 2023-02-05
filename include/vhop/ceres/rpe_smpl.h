#ifndef VHOP_INCLUDE_VHOP_CERES_RPE_SMPL_H_
#define VHOP_INCLUDE_VHOP_CERES_RPE_SMPL_H_

#include <utility>

#include "vhop/constants.h"
#include "vhop/utility.h"
#include "base_rpe_residual.h"

namespace vhop {

class ReProjectionErrorSMPL : public vhop::RPEResidualBase {

 public:

  ReProjectionErrorSMPL(const std::string &dataFilePath, const vhop::SMPL &smpl_model, size_t offset = 0)
      : RPEResidualBase(dataFilePath, smpl_model, offset) {
      cnpy::npz_t npz = cnpy::npz_load(dataFilePath);
      beta_ = vhop::utility::loadDoubleMatrix(npz.at("betas"), vhop::SHAPE_BASIS_DIM, 1);
  }

  bool operator()(const double *poseData, double *re_projection_error) const override {
      vhop::theta_t<double> poses(poseData + offset_);
      vhop::joint_op_2d_t<double> joints2d;
      smpl_model_.ComputeOpenPoseKP<double>(beta_, poses, T_C_B_, K_, &joints2d);

      for (int i = 0; i < vhop::JOINT_NUM_OP; ++i) {
          double score = joint_kps_scores_(i);
          re_projection_error[i * 2] = score * (joints2d(i, 0) - joint_kps_(i, 0));
          re_projection_error[i * 2 + 1] = score * (joints2d(i, 1) - joint_kps_(i, 1));
      }
      return true;
  }

  Eigen::VectorXd x0() override {
      // cnpy::npz_t npz_means = cnpy::npz_load("../data/vposer_mean.npz");
      return vhop::theta_t<double>::Zero();
  }

  void convert2SMPL(const Eigen::VectorXd& z, beta_t<double>& beta, theta_t<double>& theta) override {
      beta = beta_;
      theta = z;
  }

  static constexpr int getNumParams() { return vhop::THETA_DIM; }
  static constexpr int getNumResiduals() { return vhop::JOINT_NUM_OP * 2; }

 private:
    vhop::beta_t<double> beta_;
};

}

#endif //VHOP_INCLUDE_VHOP_CERES_RPE_SMPL_H_
