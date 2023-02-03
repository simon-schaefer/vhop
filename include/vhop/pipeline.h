#ifndef VHOP_INCLUDE_VHOP_PIPELINE_H_
#define VHOP_INCLUDE_VHOP_PIPELINE_H_

#include "ceres/ceres.h"
#include "vhop/ceres/base_rpe_residual.h"

namespace vhop {

class Pipeline {

public:
    Pipeline(vhop::SMPL  smpl, ceres::Solver::Options solverOptions, bool verbose = false);

    [[nodiscard]] bool process(const std::string &filePath,
                 const std::string& outputPath,
                 const vhop::Methods& method,
                 const std::string& imagePath = "") const;

protected:
    bool addReProjectionCostFunction(const std::string &filePath,
                                     const vhop::Methods &method,
                                     vhop::RPEResidualBase **cost,
                                     ceres::LossFunction **lossFunction,
                                     Eigen::VectorXd &x0,
                                     ceres::Problem &problem) const;

    vhop::SMPL smpl_model_;
    ceres::Solver::Options ceres_options_;
    bool verbose_;
};

}

#endif //VHOP_INCLUDE_VHOP_PIPELINE_H_
