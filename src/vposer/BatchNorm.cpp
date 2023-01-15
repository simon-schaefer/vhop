#include <iostream>
#include "vposer/BatchNorm.h"
#include "vposer/FileUtils.h"


BatchNorm::BatchNorm(int num_features) {
    mWeights = Eigen::VectorXd::Random(num_features);
    mBias = Eigen::VectorXd::Random(num_features);
    mMean = Eigen::VectorXd::Random(num_features);
    mVar = Eigen::VectorXd::Random(num_features);
}

Eigen::MatrixXd BatchNorm::forward(const Eigen::MatrixXd &x) {

    Eigen:: VectorXd denom = (mVar.array() + mEps).cwiseSqrt();
    Eigen::MatrixXd centered_x = x.rowwise() - mMean.transpose();
    Eigen::MatrixXd norm_x = centered_x.array().rowwise() / denom.transpose().array();
    Eigen::MatrixXd output = (norm_x.array().rowwise() * mWeights.transpose().array()).array().rowwise() + mBias.transpose().array();

    return output;
}

void BatchNorm::printDescription() {std::cout << "Batch Norm:("<< mBias.rows()<<" )" << std::endl;}


void BatchNorm::loadParams(std::string paramFilePath) {

    std::string weightFilePath = paramFilePath + ".weight.txt"; // gamma
    std::string biasFilePath = paramFilePath + ".bias.txt"; // beta
    std::string meanFilePath = paramFilePath + ".mean.txt"; // mean of train set
    std::string varFilePath = paramFilePath + ".var.txt"; // variance of train set

    //Pytorch stores weight matrices in the form of (output_shape, input_shape)
    //https://github.com/pytorch/pytorch/issues/2159
    //Before writing them into file take a transpose of the matrix
    mWeights = vhop::utility::loadVector(weightFilePath, mWeights.rows());
    mBias = vhop::utility::loadVector(biasFilePath, mBias.rows());
    mMean = vhop::utility::loadVector(meanFilePath, mMean.rows());
    mVar= vhop::utility::loadVector(varFilePath, mVar.rows());
}