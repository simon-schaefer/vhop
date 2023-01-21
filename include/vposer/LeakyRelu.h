#ifndef VHOP_LEAKYRELU_H
#define VHOP_LEAKYRELU_H


#include "BaseLayer.h"

class LeakyRelu : public BaseLayer {

    public:
        LeakyRelu(double alpha = 0.1);
        ~LeakyRelu() override = default;

        Eigen::MatrixXd forward(const Eigen::MatrixXd &x) override;
        void printDescription() override;

    private:
        double mAlpha = 0.1;
        std::string mType = "Activation";
        std::string mName = "LeakyReLU";
        Eigen::MatrixXd mForwardInput;
};

#endif //VHOP_LEAKYRELU_H
