#ifndef VHOP_INCLUDE_VHOP_PIPELINE_H_
#define VHOP_INCLUDE_VHOP_PIPELINE_H_

#include "ceres/ceres.h"
#include "vhop/ceres/base_rpe_residual.h"

namespace vhop {

template<typename RPEResidualClass, size_t NUM_TIME_STEPS>
class Pipeline {

public:
    Pipeline(vhop::SMPL  smpl, ceres::Solver::Options solverOptions, bool verbose = false);

    [[nodiscard]] bool process(const std::vector<std::string> &filePaths,
                               const std::vector<std::string> &outputPaths,
                               const std::vector<std::string> &imagePaths = std::vector<std::string>()) const;

    [[nodiscard]] bool process(const std::string &filePath,
                               const std::string &outputPath,
                               const std::string &imagePath = "") const;

  [[nodiscard]] bool processDirectory(const std::string &directory,
                                      const std::string &outputDirectory) const;

protected:
    [[nodiscard]] bool addReProjectionCostFunction(const std::string &filePath,
                                                   size_t offset,
                                                   vhop::RPEResidualBase **cost,
                                                   double* x0,
                                                   ceres::Problem &problem) const;

    [[nodiscard]] bool addConstantVelocityCostFunction(double* x0,
                                                       ceres::Problem &problem) const;

    vhop::SMPL smpl_model_;
    ceres::Solver::Options ceres_options_;
    bool verbose_;
};

}

#include "vhop/implementation/pipeline_impl.hpp"

#endif //VHOP_INCLUDE_VHOP_PIPELINE_H_
