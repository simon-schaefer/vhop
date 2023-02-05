#include <gtest/gtest.h>
#include <string>

#include "vhop/utility.h"
#include "vhop/smpl_model.h"
#include "vhop/visualization.h"

using namespace vhop;

class TestSMPL : public SMPL {

public:
  explicit TestSMPL(const std::string& npzFile) : SMPL(npzFile) {}

  Eigen::MatrixXd getVTemplate() { return restShape_; }
};

TEST(SMPLModelTest, TestModelLoading) {
    TestSMPL model("../data/smpl_neutral.npz");
    Eigen::MatrixXd vTemplateExpected(4, 3);  // first 4 vertices of vTemplate
    vTemplateExpected << 0.04487304, 0.49407477, 0.08962061,
                         0.03947177, 0.48138342, 0.09960105,
                         0.05001034, 0.47607607, 0.09099279,
                         0.05567625, 0.48491829, 0.07991982;

    Eigen::MatrixXd vTemplate = model.getVTemplate();
    EXPECT_EQ(vTemplate.rows(), VERTEX_NUM);
    EXPECT_EQ(vTemplate.cols(), 3);
    EXPECT_TRUE(vTemplate.block(0, 0, 4, 3).isApprox(vTemplateExpected, 0.001));
}

TEST(SMPLModelTest, TestForward) {
    TestSMPL model("../data/smpl_neutral.npz");
    cnpy::npz_t npz = cnpy::npz_load("../data/test/sample.npz");

    beta_t<double> betas = vhop::utility::loadDoubleMatrix(npz.at("betas"), SHAPE_BASIS_DIM, 1);
    theta_t<double> thetas = vhop::utility::loadDoubleMatrix(npz.at("thetas"), THETA_DIM, 1);
    joint_t<double> joints;
    vertex_t<double> vertices;
    model.Forward(betas, thetas, &joints, &vertices);

    Eigen::Matrix<double, vhop::JOINT_NUM, 3> joints3d = joints.reshaped(3, vhop::JOINT_NUM).transpose();
    Eigen::MatrixXd joints3d_full_exp = vhop::utility::loadDoubleMatrix(npz.at("joints_3d_wo_translation"), vhop::JOINT_NUM_TOTAL, 3);
    Eigen::Matrix<double, vhop::JOINT_NUM, 3> joints3d_exp = joints3d_full_exp.block(0, 0, vhop::JOINT_NUM, 3);
    EXPECT_TRUE((joints3d - joints3d_exp).cwiseAbs().maxCoeff() < 0.001);
}

TEST(TestReprojectionError, TestOpenPoseReprojection) {
  cnpy::npz_t npz = cnpy::npz_load("../data/test/sample.npz");
  beta_t<double> beta = vhop::utility::loadDoubleMatrix(npz.at("betas"), vhop::SHAPE_BASIS_DIM, 1);
  theta_t<double> theta = vhop::utility::loadDoubleMatrix(npz.at("thetas"), vhop::JOINT_NUM * 3, 1);
  Eigen::Matrix3d K = vhop::utility::loadDoubleMatrix(npz.at("intrinsics"), 3, 3);
  Eigen::Matrix4d T_C_B = vhop::utility::loadDoubleMatrix(npz.at("T_C_B"), 4, 4);

  SMPL smpl_model("../data/smpl_neutral.npz");
  joint_op_2d_t<double> joints_2d;
  smpl_model.ComputeOpenPoseKP(beta, theta, T_C_B, K, &joints_2d);

  joint_op_2d_t<double> joints_kp = vhop::utility::loadDoubleMatrix(npz.at("keypoints_2d"), 25, 2);
  for(int i = 0; i < 25; ++i) {
    std::string outputFile = "../data/test/smpl/sample_reprojected_" + std::to_string(i) + ".png";
    vhop::visualization::drawKeypoints("../data/test/sample.jpg",
                                     joints_2d.block<1, 2>(i, 0).cast<int>(),
                                     joints_kp.block<1, 2>(i, 0).cast<int>(),
                                     outputFile);
  }
//  std::vector<double> errors(1);
//  residual(theta.data(), errors.data());
//  EXPECT_NEAR(errors[0], 0.0, 0.1);
}
