#include <gtest/gtest.h>
#include <string>

#include "vhop/reprojection_error.h"
#include "vhop/smpl_model.h"
#include "vhop/utility.h"
#include "vhop/visualization.h"

using namespace vhop;


TEST(TestReprojectionError, TestWGroundtruth) {
  cnpy::npz_t npz = cnpy::npz_load("../data/test/sample.npz");
  beta_t<double> beta = vhop::utility::loadDoubleMatrix(npz.at("betas"), vhop::SHAPE_BASIS_DIM, 1);
  theta_t<double> theta = vhop::utility::loadDoubleMatrix(npz.at("thetas"), vhop::JOINT_NUM * 3, 1);
  Eigen::Matrix3d K = vhop::utility::loadDoubleMatrix(npz.at("intrinsics"), 3, 3);
  Eigen::Matrix4d T_C_B = vhop::utility::loadDoubleMatrix(npz.at("T_C_B"), 4, 4);
  joint_op_2d_t<double> joints_kp = vhop::utility::loadDoubleMatrix(npz.at("keypoints_2d"), 25, 2);

  SMPL smpl_model("../data/smpl_neutral.npz");
  ReprojectionError residual(beta, K, T_C_B, joints_kp, smpl_model);

  vhop::joint_op_2d_t<double> joints_2d = residual.computeJoints2D(theta);
  std::cout << "joints_2d: " << std::endl << joints_2d << std::endl;
  std::cout << "joints_kp: " << std::endl << joints_kp << std::endl;

//  std::vector<double> errors(1);
//  residual(theta.data(), errors.data());
//  EXPECT_NEAR(errors[0], 0.0, 0.1);
}
