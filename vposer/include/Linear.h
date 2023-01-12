//
// Created by Burak on 9.01.2023.
//

#ifndef VHOP_LINEAR_H
#define VHOP_LINEAR_H

#include "BaseLayer.h"

class Linear : public BaseLayer {

public:

    Linear(int inputFeatNum, int outFeatNum);
    ~Linear() override = default;

    Eigen::MatrixXd forward(const Eigen::MatrixXd &x) override;

    void printDescription() override;
    void loadParams(std::string paramFilePath) override;

    Eigen::MatrixXd getWeights();

    Eigen::MatrixXd getBias();


private:
    Eigen::MatrixXd mWeights;
    Eigen::VectorXd  mBias;
};

#endif //VHOP_LINEAR_H
