#ifndef VHOP_CERES_BASE_RESIDUAL_H
#define VHOP_CERES_BASE_RESIDUAL_H

#include <cnpy.h>
#include <Eigen/Dense>

#include "vhop/smpl_model.h"

namespace vhop {

class ResidualBase {

public:
    virtual ~ResidualBase() = default;

    virtual bool operator()(const double *params, double *residuals) const = 0;
};

}

#endif //VHOP_CERES_BASE_RESIDUAL_H
