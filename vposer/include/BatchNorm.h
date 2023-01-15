//
// Created by Burak on 9.01.2023.
//

#ifndef VHOP_BATCHNORM_H
#define VHOP_BATCHNORM_H

#include "BaseLayer.h"

class BatchNorm : public BaseLayer{

public:

    BatchNorm(int num_features);

    ~BatchNorm() override = default;

    Eigen::MatrixXd forward(const Eigen::MatrixXd &x)override ;
    void printDescription() override;
    void loadParams(std::string paramFilePath)override ;
private:
    double mEps= 1e-5;
    Eigen::VectorXd mWeights;
    Eigen::VectorXd mBias;
    Eigen::VectorXd  mMean;
    Eigen::VectorXd  mVar;

};

#endif //VHOP_BATCHNORM_H
