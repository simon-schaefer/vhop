#include <iostream>
#include "vposer/Linear.h"
#include "vhop/utility.h"


Linear::Linear(int inputFeatNum, int outFeatNum) {
    mWeights = Eigen::MatrixXd::Random(outFeatNum, inputFeatNum);
    mBias = Eigen::VectorXd::Random(outFeatNum);
}

Linear::Linear(int inputFeatNum, int outFeatNum, const cnpy::npz_t& raw, const std::string& name) {
    assert(raw.contains(name + ".weight"));
    assert(raw.contains(name + ".bias"));
    mWeights = vhop::utility::loadDoubleMatrix(raw.at(name + ".weight"), outFeatNum,inputFeatNum);
    mBias = vhop::utility::loadDoubleMatrix(raw.at(name + ".bias"), (int)mWeights.rows(), 1);
}

Eigen::MatrixXd Linear::forward(const Eigen::MatrixXd &x) {
    return mWeights * x + mBias;
}

void Linear::printDescription() {
    std::cout << "Linear layer: (" << mWeights.rows() << ", " << mWeights.cols()<< ") " << std::endl;
}
