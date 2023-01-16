#include <gtest/gtest.h>
#include <string>

#include "vhop/utility.h"
#include "vhop/smpl_model.h"
#include "vhop/reprojection_error.h"

using namespace vhop;


TEST(TestReprojectionError, TestWGroundtruth) {
  cnpy::npz_t npz = cnpy::npz_load("../data/test/sample.npz");
  beta_t<double> beta = vhop::utility::loadDoubleMatrix(npz.at("betas"), vhop::SHAPE_BASIS_DIM, 1);
  theta_t<double> thetas = vhop::utility::loadDoubleMatrix(npz.at("thetas"), vhop::JOINT_NUM * 3, 1);
  translation_t<double> translation = vhop::utility::loadDoubleMatrix(npz.at("translation"), 3, 1);
  Eigen::Matrix3d K = vhop::utility::loadDoubleMatrix(npz.at("intrinsics"), 3, 3);
  joint_op_2d_t<double> joints_kp = vhop::utility::loadDoubleMatrix(npz.at("keypoints_2d"), 25, 2);

  SMPL smpl_model("../data/smpl_neutral.npz");
  ReprojectionError residual(beta, translation, K, joints_kp, smpl_model);
  std::cout << residual.evaluate(thetas) << std::endl;
}