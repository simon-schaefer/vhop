#include <iostream>
#include "vposer/Linear.h"
#include "vposer/FileUtils.h"


Linear::Linear(int inputFeatNum, int outFeatNum) {

    mWeights = Eigen::MatrixXd::Random(inputFeatNum, outFeatNum);
    mBias = Eigen::VectorXd::Random(outFeatNum);
}

Eigen::MatrixXd Linear::forward(const Eigen::MatrixXd &x) {
    Eigen::MatrixXd out = x.matrix() * mWeights.matrix();
    out = out.rowwise() + mBias.transpose();

    return out;
}

void Linear::printDescription() {
    std::cout << "Linear layer: (" << mWeights.rows() << ", " << mWeights.cols()<< ") " << std::endl;
}

void Linear::loadParams(std::string paramFilePath) {

    std::string weightFilePath = paramFilePath + ".weight.txt";
    std::string biasFilePath = paramFilePath + ".bias.txt";

    //Pytorch stores weight matrices in the form of (output_shape, input_shape)
    //https://github.com/pytorch/pytorch/issues/2159
    //Before writing them into file take a transpose of the matrix so that input text files are corrected
    Eigen::MatrixXd loadedParams = vhop::utility::loadDoubleMatrix(weightFilePath, mWeights.rows(),mWeights.cols());
    mWeights = loadedParams.matrix();

    Eigen::MatrixXd loadedBiasParams = vhop::utility::loadVector(biasFilePath, mBias.rows());
    mBias = loadedBiasParams.matrix();

}

Eigen::MatrixXd Linear::getWeights() { return mWeights; }

Eigen::MatrixXd Linear::getBias() { return mBias; }

