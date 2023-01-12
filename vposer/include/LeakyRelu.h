//
// Created by Burak on 9.01.2023.
//

#ifndef VHOP_LEAKYRELU_H
#define VHOP_LEAKYRELU_H


#include "BaseLayer.h"

class LeakyRelu : public BaseLayer {

    public:

        LeakyRelu();
        LeakyRelu(float alpha);
        ~LeakyRelu() override = default;

        Eigen::MatrixXd forward(const Eigen::MatrixXd &x) override;
        void printDescription() override;
        void loadParams(std::string paramFilePath) override;

    private:
        float mAlpha = 0.1f;
        std::string mType = "Activation";
        std::string mName = "LeakyReLU";
        Eigen::MatrixXd mForwardInput;
};

#endif //VHOP_LEAKYRELU_H
