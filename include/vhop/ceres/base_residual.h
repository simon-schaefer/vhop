#ifndef VHOP_CERS_BASE_RESIDUAL
#define VHOP_CERS_BASE_RESIDUAL

#include <cnpy.h>
#include <Eigen/Dense>

#include "vhop/smpl_model.h"

namespace vhop {

class ResidualBase {

public:
    explicit ResidualBase(const vhop::SMPL &smpl_model) : smpl_model_(smpl_model) {};
    ResidualBase(const std::string& dataFilePath, const vhop::SMPL &smpl_model) : smpl_model_(smpl_model) {};
    virtual ~ResidualBase() = default;

    virtual bool operator()(const double *params, double *residuals) const = 0;
    virtual Eigen::VectorXd x0() = 0;
    virtual void convert2SMPL(const Eigen::VectorXd& z, beta_t<double>& beta, theta_t<double> theta) = 0;

protected:
    vhop::SMPL smpl_model_;
};

}

#endif //VHOP_CERS_BASE_RESIDUAL
