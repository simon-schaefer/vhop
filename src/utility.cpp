#include "vhop/utility.h"

#include <cnpy.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <Eigen/Dense>
#include <vector>


std::vector<std::filesystem::path> vhop::utility::listFiles(const std::string& directory,
                                                            const std::string& suffix,
                                                            bool recursive) {
    std::vector<std::filesystem::path> outputs;
    const std::filesystem::path dir(directory);

    for(const auto& file : std::filesystem::recursive_directory_iterator(directory)) {
        const auto &filePath = file.path();
        if (filePath.extension() != suffix) continue;
        if (!recursive && !std::filesystem::equivalent(filePath.parent_path(), dir)) continue;
        outputs.emplace_back(filePath);
    }
    return outputs;
}

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
                                        const vhop::theta_t<double>& theta,
                                        const double& executionTime) {
    std::vector<double> outputs(vhop::SHAPE_BASIS_DIM + vhop::THETA_DIM + 1);
    for(int i = 0; i < vhop::SHAPE_BASIS_DIM; i++) {
        outputs[i] = beta(i);
    }
    for(int i = 0; i < vhop::THETA_DIM; i++) {
        outputs[i + SHAPE_BASIS_DIM] = theta(i);
    }
    outputs[vhop::SHAPE_BASIS_DIM + vhop::THETA_DIM] = executionTime;
    writeVector(filePath, outputs);
}

