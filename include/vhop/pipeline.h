#ifndef VHOP_INCLUDE_VHOP_PIPELINE_H_
#define VHOP_INCLUDE_VHOP_PIPELINE_H_

#include "ceres/ceres.h"
#include "vhop/ceres/base_residual.h"

namespace vhop {

class Pipeline {

public:
  explicit Pipeline(const vhop::SMPL &smpl);

  bool process(const std::string &filePath,
               const std::string &outputPath,
               const vhop::Methods &method,
               bool doPrintSummary = false);

  void setSMPLCostFunction(const std::string &filePath,
                           vhop::ResidualBase **cost,
                           ceres::CostFunction **costFunction);
  void setVPoserCostFunction(const std::string &filePath,
                             vhop::ResidualBase **cost,
                             ceres::CostFunction **costFunction);

 protected:
  vhop::SMPL smpl_model_;

};

}

#endif //VHOP_INCLUDE_VHOP_PIPELINE_H_
