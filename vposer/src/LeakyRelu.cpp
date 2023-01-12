//
// Created by Burak on 9.01.2023.
//

#include <Eigen/Dense>
#include <iostream>
#include "../include/LeakyRelu.h"

LeakyRelu::LeakyRelu() {
    mAlpha = 0.01;
}

LeakyRelu::LeakyRelu(float alpha) {
    mAlpha = alpha;
}

Eigen::MatrixXd LeakyRelu::forward(const Eigen::MatrixXd &x) {

    Eigen::MatrixXd out(x.rows(), x.cols());
    out = x.cwiseMax(mAlpha * x);
    return out;
}

void LeakyRelu::printDescription() { std::cout << "Leaky ReLU activation with alpha: " << mAlpha <<std::endl; }

void LeakyRelu::loadParams(std::string paramFilePath) {

}