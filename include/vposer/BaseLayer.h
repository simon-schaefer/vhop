#ifndef VHOP_BASELAYER_H
#define VHOP_BASELAYER_H
#include <Eigen/Dense>

class BaseLayer {
    public:
        virtual ~BaseLayer() = default;
        virtual Eigen::MatrixXd forward(const Eigen::MatrixXd &x) = 0;
        virtual void printDescription() = 0;
        virtual void loadParams(std::string paramFilePath) = 0;
};

#endif //VHOP_BASELAYER_H
