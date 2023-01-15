#include "vhop/utility.h"

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


Eigen::VectorXd vhop::utility::loadVector(const std::string& filePath, int row){
    std::ifstream fin(filePath);

    if(!fin){
        std::cout << "Error: unable to read txt file: " << filePath << std::endl;
    }

    int r = 0; int c = 0;
    std::string line;

    // Read first line which contains dimensions of the matrix
    std::getline(fin, line);
    size_t pos;

    int vector_dim = stoi(line);
    // std::cout << "Reading a vector with " <<  vector_dim << " dimension" << std::endl;

    Eigen::MatrixXd out = Eigen::VectorXd::Zero(vector_dim);
    while (getline(fin, line)) {
        std::string token;
        while((pos = line.find('\t')) != std::string::npos){
            token = line.substr(0, pos);

            out(c) = stod(token);
            line.erase(0, pos + 1);
            c += 1;
            if(c== vector_dim) break;
        }
        if (!line.empty()){
            out(c) = stod(line);
        }
    }
    fin.close();

    if(out.rows() != row){
        throw std::invalid_argument( "loadded weight parameter dimension does not match with the weights" );
    }

    return out;
}

Eigen::MatrixXd vhop::utility::loadDoubleMatrix(const std::string& filePath, int row, int col) {

    std::ifstream fin(filePath);

    if(!fin){
        std::cout << "Error: unable to read txt file: " << filePath << std::endl;
    }

    int r = 0; int c = 0;
    std::string line;

    // Read first line which contains dimensions of the matrix
    std::getline(fin, line);
    size_t pos = line.find('\t');

    std::string str_dim1 = line.substr(0, pos);
    int dim1 = std::stoi(str_dim1);
    std::string str_dim2 = line.substr(pos+1);
    int dim2 = std::stoi(str_dim2);
    // std::cout << "Reading a vector with the dimension(" <<  dim1 << ", "<< dim2 << ")" << std::endl;

    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(dim1, dim2);
    while (getline(fin, line)) {
        std::string token;
        while((pos = line.find('\t')) != std::string::npos){
            token = line.substr(0, pos);

            out(r,c) = stod(token);
            line.erase(0, pos + 1);
            c++;
            if(c == dim2){
                break;
            }
        }
        if (!line.empty()){
            out(r,c) = stod(token);
        }
        r += 1;
        c = 0;
    }
    fin.close();

    if(out.rows() != row || out.cols() != col){
        throw std::invalid_argument( "loadded weight parameter dimension does not match with the weights" );
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
