#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>

#include <cnpy.h>
#include "vhop/utility.h"


TEST(TestUtility, TestRodriguez) {
    // conversion from https://www.andre-gaschler.com/rotationconverter/
    Eigen::Vector3d r(1.7289318, 0.0823301, 0.4939805);
    Eigen::Matrix3d R = vhop::utility::rodriguesMatrix(r);
    Eigen::Matrix3d expectedRotation;
    expectedRotation << 0.9050074, -0.2133418,  0.3680312,
                        0.3211713, -0.2246347, -0.9199936,
                        0.2789457,  0.9508020, -0.1347768;
    EXPECT_TRUE(R.isApprox(expectedRotation, 0.001));
}

TEST(TestUtility, TestLoadDoubleMatrix) {
    cnpy::npz_t npz = cnpy::npz_load("../data/test/reading.npz");
    const cnpy::NpyArray& fooMatrix = npz.at("a");
    Eigen::MatrixXd m = vhop::utility::loadDoubleMatrix(fooMatrix, 3, 2);
    Eigen::MatrixXd expectedMatrix(3, 2);
    expectedMatrix << 0.94149614, 0.85808574,
                      0.88884227, 0.35643352,
                      0.35795367, 0.14058801;
    EXPECT_TRUE(m.isApprox(expectedMatrix, 0.0001));
}

int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
