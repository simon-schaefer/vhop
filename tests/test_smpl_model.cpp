#include <gtest/gtest.h>
#include <string>

#include "vhop/utility.h"
#include "vhop/smpl_model.h"

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
    cnpy::npz_t npz = cnpy::npz_load("../data/test/smpl_forward.npz");

    beta_t betas = vhop::utility::loadDoubleMatrix(npz.at("betas"), SHAPE_BASIS_DIM, 1);
    theta_t thetas = vhop::utility::loadDoubleMatrix(npz.at("thetas"), JOINT_NUM * 3, 1);
    translation_t translation = vhop::utility::loadDoubleMatrix(npz.at("t"), 3, 1);
    joint_t joints;
    vertex_t vertices;
    model.Forward(betas, thetas, translation, &joints, &vertices);

//    cnpy::npz_t npzLBS = cnpy::npz_load("../data/test/smpl_lbs.npz");
//    Eigen::MatrixXd verticesExpected = vhop::utility::loadDoubleMatrix(npz.at("vertices"), 6890, 3);
//    Eigen::Matrix<double, 24, 3> jointsExpected = vhop::utility::loadDoubleMatrix(npz.at("joints"), 24, 3);
//    EXPECT_TRUE(vertices.isApprox(verticesExpected, 0.001));
//    EXPECT_TRUE(joints.isApprox(jointsExpected, 0.001));
}


int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
