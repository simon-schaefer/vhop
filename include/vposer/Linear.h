#ifndef VHOP_LINEAR_H
#define VHOP_LINEAR_H

#include "BaseLayer.h"

class Linear : public BaseLayer {

public:
    Linear(int inputFeatNum, int outFeatNum);
    Linear(int inputFeatNum, int outFeatNum, const cnpy::npz_t& raw, const std::string& name);
    ~Linear() override = default;

    Eigen::MatrixXd forward(const Eigen::MatrixXd &x) override;
    void printDescription() override;

    inline Eigen::MatrixXd getWeights() { return mWeights; }
    inline Eigen::MatrixXd getBias() {return mBias; }

private:
    Eigen::MatrixXd mWeights;
    Eigen::VectorXd  mBias;
};

#endif //VHOP_LINEAR_H
