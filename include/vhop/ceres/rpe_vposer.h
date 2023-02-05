#ifndef VHOP_INCLUDE_VHOP_CERES_RPE_VPOSER_H_
#define VHOP_INCLUDE_VHOP_CERES_RPE_VPOSER_H_

#include "vhop/constants.h"
#include "vhop/utility.h"
#include "vposer/VPoser.h"
#include "base_rpe_residual.h"

namespace vhop {

class ReProjectionErrorVPoser : public vhop::RPEResidualBase {

 public:

  ReProjectionErrorVPoser(const std::string &dataFilePath, const vhop::SMPL &smpl_model, size_t offset = 0)
      : RPEResidualBase(dataFilePath, smpl_model, offset),
      vposer_("../data/vposer_weights.npz", 512) {
      cnpy::npz_t npz = cnpy::npz_load(dataFilePath);
      beta_ = vhop::utility::loadDoubleMatrix(npz.at("betas"), vhop::SHAPE_BASIS_DIM, 1);
  }

  bool operator()(const double *latentZ, double *reprojection_error) const override {
      vposer::latent_t<double> z(latentZ + offset_);
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

  Eigen::VectorXd x0() override {
      return vposer::latent_t<double>::Zero();
  }

  void convert2SMPL(const Eigen::VectorXd& z, beta_t<double>& beta, theta_t<double>& theta) override {
      beta = beta_;
      vhop::rotMats_t<double> rotMats = vposer_.decode(z, true);
      for(int i = 0; i < rotMats.size(); i++) {
          theta.segment<3>(i * 3) = vhop::utility::rodriguesVector(rotMats[i]);
      }
  }

  static constexpr int getNumParams() { return vhop::THETA_DIM; }
  static constexpr int getNumResiduals() { return vhop::JOINT_NUM_OP * 2; }

 private:
  VPoser vposer_;
  vhop::beta_t<double> beta_;
};

}

#endif //VHOP_INCLUDE_VHOP_CERES_RPE_VPOSER_H_
