#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "vposer/VPoser.h"
#include "vposer/BatchNorm.h"
#include "vposer/Linear.h"
#include "vposer/LeakyRelu.h"
#include "vhop/utility.h"

/**
 * Test data creation in data/vposer_data.py
 * See all parameters and matrix sizes in the Python file
 */

TEST(TestVPoser, TestLinearLayer){
    cnpy::npz_t npz = cnpy::npz_load("../data/test/linear_layer.npz");
    Eigen::Vector<double, 3> z = vhop::utility::loadDoubleMatrix(npz.at("z"), 3, 1);
    Linear l1 = Linear(3, 2, npz, "layer");

    Eigen::Vector2d outExpected = vhop::utility::loadDoubleMatrix(npz.at("out"), 2, 1);
    Eigen::Vector2d out = l1.forward(z);
    EXPECT_TRUE(out.isApprox(outExpected, 0.001));
}

TEST(TestVPoser, TestBatchNorm){
    cnpy::npz_t npz = cnpy::npz_load("../data/test/batch_norm.npz");
    Eigen::Vector<double, 20> z = vhop::utility::loadDoubleMatrix(npz.at("z"), 20, 1);
    BatchNorm l1 = BatchNorm(20, npz, "layer");

    Eigen::Vector<double, 20> outExpected = vhop::utility::loadDoubleMatrix(npz.at("out"), 20, 1);
    Eigen::Vector<double, 20> out = l1.forward(z);
    EXPECT_TRUE(out.isApprox(outExpected, 0.001));
}

TEST(TestVPoser, TestLeakyRelu){
    cnpy::npz_t npz = cnpy::npz_load("../data/test/leaky_relu.npz");
    Eigen::Vector<double, 20> z = vhop::utility::loadDoubleMatrix(npz.at("z"), 20, 1);
    LeakyRelu l1 = LeakyRelu(0.2);

    Eigen::Vector<double, 20> outExpected = vhop::utility::loadDoubleMatrix(npz.at("out"), 20, 1);
    Eigen::Vector<double, 20> out = l1.forward(z);
    EXPECT_TRUE(out.isApprox(outExpected, 0.001));
}

TEST(TestVPoser, TestForward){
    VPoser vposer = VPoser("../data/vposer_weights.npz", 512);
    cnpy::npz_t npz = cnpy::npz_load("../data/test/vposer_data.npz");
    vposer::latent_t<double> z = vhop::utility::loadDoubleMatrix(npz.at("z"), vposer::LATENT_DIM, 1);

    vhop::AlignedVector<Eigen::Matrix3d> rotMats = vposer.decode(z, false);
    Eigen::Matrix<double, 21, 9> rotMatsExp = vhop::utility::loadDoubleMatrix(npz.at("R_out"), 21, 9);
    for(int i = 0; i < rotMats.size(); i++) {
        Eigen::Matrix3d rotMatExp_i = rotMatsExp.row(i).reshaped(3, 3).transpose();
        EXPECT_TRUE(rotMats[i].isApprox(rotMatExp_i, 0.001));
    }
}
