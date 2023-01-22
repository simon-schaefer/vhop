#include <iostream>
#include "vposer/BatchNorm.h"
#include "vhop/utility.h"


BatchNorm::BatchNorm(int numFeatures) {
    mWeights = Eigen::VectorXd::Random(numFeatures);
    mBias = Eigen::VectorXd::Random(numFeatures);
    mMean = Eigen::VectorXd::Random(numFeatures);
    mVar = Eigen::VectorXd::Random(numFeatures);
}

BatchNorm::BatchNorm(int numFeatures, const cnpy::npz_t& raw, const std::string& name) {
    mWeights = vhop::utility::loadDoubleMatrix(raw.at(name + ".weight"), numFeatures, 1);
    mBias = vhop::utility::loadDoubleMatrix(raw.at(name + ".bias"), numFeatures, 1);
    mMean = vhop::utility::loadDoubleMatrix(raw.at(name + ".running_mean"), numFeatures, 1);
    mVar = vhop::utility::loadDoubleMatrix(raw.at(name + ".running_var"), numFeatures, 1);
}

Eigen::MatrixXd BatchNorm::forward(const Eigen::MatrixXd &x) {
    Eigen:: VectorXd denom = (mVar.array() + mEps).cwiseSqrt();
    Eigen::MatrixXd centered_x = x.rowwise() - mMean.transpose();
    Eigen::MatrixXd norm_x = centered_x.array().rowwise() / denom.transpose().array();
    Eigen::MatrixXd output = (norm_x.array().rowwise() * mWeights.transpose().array()).array().rowwise() + mBias.transpose().array();

    return output;
}

void BatchNorm::printDescription() {std::cout << "Batch Norm:("<< mBias.rows()<<" )" << std::endl;}
