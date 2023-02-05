#include <Eigen/Dense>
#include <iostream>

#include "vposer/LeakyRelu.h"


LeakyRelu::LeakyRelu(double alpha) {
    mAlpha = alpha;
}

Eigen::MatrixXd LeakyRelu::forward(const Eigen::MatrixXd &x) {
    return x.cwiseMax(mAlpha * x);
}

void LeakyRelu::printDescription() { std::cout << "Leaky ReLU activation with alpha: " << mAlpha <<std::endl; }
