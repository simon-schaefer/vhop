#include "vhop/utility.h"
#include "vhop/constants.h"

#include <cnpy.h>
#include <fstream>
#include <iostream>
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

// Implementation from https://stackoverflow.com/questions/46663046/save-read-double-vector-from-file-c
std::vector<double> vhop::utility::loadVector(const std::string& filePath) {
    std::vector<char> buffer{};
    std::ifstream ifs(filePath, std::ios::in | std::ifstream::binary);
    std::istreambuf_iterator<char> iter(ifs);
    std::istreambuf_iterator<char> end{};
    std::copy(iter, end, std::back_inserter(buffer));
    std::vector<double> newVector(buffer.size() / sizeof(double));
    memcpy(&newVector[0], &buffer[0], buffer.size());
    return newVector;
}

// Implementation from https://stackoverflow.com/questions/46663046/save-read-double-vector-from-file-c
void vhop::utility::writeVector(const std::string& filename, const std::vector<double>& myVector) {
    std::ofstream ofs(filename, std::ios::out | std::ofstream::binary);
    std::ostream_iterator<char> osi{ ofs };
    const char* beginByte = (char*)&myVector[0];

    const char* endByte = (char*)&myVector.back() + sizeof(double);
    std::copy(beginByte, endByte, osi);
}

void vhop::utility::writeSMPLParameters(const std::string& filePath,
                                        const vhop::beta_t<double>& beta,
                                        const vhop::theta_t<double>& theta) {
    std::vector<double> outputs(vhop::SHAPE_BASIS_DIM + vhop::THETA_DIM);
    for(int i = 0; i < vhop::SHAPE_BASIS_DIM; i++) {
        outputs[i] = beta(i);
    }
    for(int i = 0; i < vhop::THETA_DIM; i++) {
        outputs[i + SHAPE_BASIS_DIM] = theta(i);
    }
    writeVector(filePath, outputs);
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

Eigen::Vector3d vhop::utility::rodriguesVector(const Eigen::Matrix3d &R) {
    Eigen::AngleAxisd r(R);
    return r.angle() * r.axis();
}

Eigen::Matrix<double, Eigen::Dynamic, 2> vhop::utility::project(const Eigen::Matrix<double, Eigen::Dynamic, 3>& p, const Eigen::Matrix3d& K) {
    Eigen::Matrix<double, Eigen::Dynamic, 2> p2d(p.rows(), 2);
    for (int i = 0; i < p.rows(); i++) {
        p2d(i, 0) = p(i, 0) * K(0, 0) / p(i, 2) + K(0, 2);
        p2d(i, 1) = p(i, 1) * K(1, 1) / p(i, 2) + K(1, 2);
    }
    return p2d;
}
