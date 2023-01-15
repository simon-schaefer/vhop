#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "vposer/VPoser.h"
#include "vposer/LeakyRelu.h"
#include "vhop/utility.h"


TEST(TestVPoser, TestLinearLayer){
    // Linear Layer
    // Input [0.4454, 0.8607, 0.2194, 0.6060, 0.6443]
    // Output [ 0.0576, -0.2601,  0.7225,  0.4670, -0.0571,  0.3613, -0.6810, -0.9362, 0.7772, -0.0737]
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(1,5);
    input << 0.4454, 0.8607, 0.2194, 0.6060, 0.6443;
    Linear l1 = Linear(5, 10);
    std::string weightPath = "../data/vposer/test_weights/linear";
    l1.loadParams(weightPath);

    Eigen::MatrixXd output = l1.forward(input);
    Eigen::MatrixXd expectedOutput(1,10);
    expectedOutput << 0.0576, -0.2601,  0.7225,  0.4670, -0.0571,  0.3613, -0.6810, -0.9362, 0.7772, -0.0737;
    EXPECT_TRUE(output.isApprox(expectedOutput, 0.001));

    // Linear + Leaky Relu
    // Input [-0.24, -24.2, 0.23]
    // Output [ 0.4959, -0.0185, -0.1139, 10.8231, -0.1283]
    input = Eigen::MatrixXd::Random(1,3);
    input(0,0) = -0.24; input(0,1) =  -24.2; input(0,2) =0.23;
    l1 = Linear(3, 5);
    weightPath = "../data/vposer/test_weights/leaky";
    l1.loadParams(weightPath);

    output = l1.forward(input);
    LeakyRelu leakyRelu = LeakyRelu(0.1);
    Eigen::MatrixXd lout = leakyRelu.forward(output);

    Eigen::MatrixXd expectedOutput2(1,5);
    expectedOutput2 << 0.495944, -0.185348, -1.13853, 10.8231, -1.28328;
    EXPECT_TRUE(lout.isApprox(expectedOutput2, 0.001));
}

TEST(TestVPoser, TestForward){
    VPoser vposer = VPoser("../data/vposer/weights/", 512, 32);
    // vposer.printModel();
    vposer.loadParams();

    Eigen::MatrixXd sample_input = vhop::utility::loadDoubleMatrix("../data/vposer/sample_input/amass_body_input.txt", 500, 63);
    vposer.forward(sample_input);
}

int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
