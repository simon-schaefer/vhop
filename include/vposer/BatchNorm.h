#ifndef VHOP_BATCHNORM_H
#define VHOP_BATCHNORM_H

#include "BaseLayer.h"

class BatchNorm : public BaseLayer{

public:
    BatchNorm(int numFeatures);
    BatchNorm(int numFeatures, const cnpy::npz_t& raw, const std::string& name);
    ~BatchNorm() override = default;

    Eigen::MatrixXd forward(const Eigen::MatrixXd &x )override ;
    void printDescription() override;

private:
    double mEps= 1e-5;
    Eigen::VectorXd mWeights;
    Eigen::VectorXd mBias;
    Eigen::VectorXd  mMean;
    Eigen::VectorXd  mVar;

};

#endif //VHOP_BATCHNORM_H
