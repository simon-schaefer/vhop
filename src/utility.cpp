#include "vhop/utility.h"

#include <cnpy.h>
#include <Eigen/Dense>


Eigen::MatrixXd vhop::utility::loadDoubleMatrix3D(const cnpy::NpyArray& raw, int r, int c, int dim) {
    std::vector<double> dataVector;
    auto *data = raw.data<double>();
    dataVector.assign(data, data + r * 3 * c);

    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(r, c);
    for (int v = 0; v < r; v++) {
        for (int s = 0; s < c; s++) {
            out(v, s) = dataVector[v * 3 * c + dim * c + s];
        }
    }
    return out;
}

Eigen::MatrixXd vhop::utility::loadDoubleMatrix(const cnpy::NpyArray &raw, int r, int c) {
    std::vector<double> dataVector;
    auto *data = raw.data<double>();
    dataVector.assign(data, data + r * c);

    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(r, c);
    for (int v = 0; v < r; v++) {
        for (int s = 0; s < c; s++) {
            out(v, s) = dataVector[v * c + s];
        }
    }
    return out;
}


Eigen::Matrix<double, 3, 3> vhop::utility::rodriguesMatrix(const Eigen::Vector3d &r) {
    double theta = r.norm();
    Eigen::Matrix<double, 3, 3> R = Eigen::Matrix<double, 3, 3>::Identity();
    if (theta > 1e-8) {
        Eigen::Vector3d r_normalized = r / theta;
        Eigen::Matrix<double, 3, 3> r_cross;
        r_cross << 0, -r_normalized(2), r_normalized(1),
            r_normalized(2), 0, -r_normalized(0),
            -r_normalized(1), r_normalized(0), 0;
        R += sin(theta) * r_cross + (1 - cos(theta)) * r_cross * r_cross;
    }
    return R;
}
