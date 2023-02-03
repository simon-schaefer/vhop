#ifndef VHOP_INCLUDE_VHOP_PIPELINE_IMPL_HPP_
#define VHOP_INCLUDE_VHOP_PIPELINE_IMPL_HPP_

#include <utility>

#include "vhop/pipeline.h"

#include "vhop/ceres/rpe_smpl.h"
#include "vhop/ceres/rpe_vposer.h"
#include "vhop/ceres/constant_motion.h"


template<typename RPEResidualClass, size_t numTimeSteps>
vhop::Pipeline<RPEResidualClass, numTimeSteps>::Pipeline(vhop::SMPL smpl,
                                                           ceres::Solver::Options solverOptions,
                                                           bool verbose)
    : smpl_model_(std::move(smpl)),
      ceres_options_(std::move(solverOptions)),
      verbose_(verbose) {}

template<typename RPEResidualClass, size_t numTimeSteps>
bool vhop::Pipeline<RPEResidualClass, numTimeSteps>::process(
    const std::vector<std::string> &filePaths,
    const std::vector<std::string> &outputPaths,
    const std::vector<std::string> &imagePaths) const {
  assert(filePath.size() == outputPath.size());
  assert(imagePath.size() == 0 || imagePath.size() == filePath.size());

  ceres::Problem::Options problemOptions;
  problemOptions.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problemOptions.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problemOptions.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problemOptions);

  constexpr int numParams = RPEResidualClass::getNumParams() * numTimeSteps;
  auto* x0 = new double[numParams];
  // Add re-projection error residuals for every image, i.e. once per timestamp.
  std::vector<vhop::RPEResidualBase*> costs;
  for(int t = 0; t < numTimeSteps; ++t) {
     vhop::RPEResidualBase* cost;
     const size_t offset = t * RPEResidualClass::getNumParams();
     if(!addReProjectionCostFunction(filePaths[t], offset, &cost, x0,problem)) {
      std::cout << "Failed to add re-projection cost function" << std::endl;
      return false;
    }
    costs.push_back(cost);
  }
  // Add constant velocity residual if there is more than one timestamp.
  if(numTimeSteps > 1) {
    if(!addConstantVelocityCostFunction(x0, problem)) {
      std::cout << "Failed to add constant velocity cost function" << std::endl;
      return false;
    }
  }

  // Solve the optimization problem.
  ceres::Solver::Summary summary;
  ceres::Solve(ceres_options_, &problem, &summary);
  if (verbose_) std::cout << summary.FullReport() << std::endl;

  // Write the results to several files.
  Eigen::Vector<double, numParams> x_sol(x0);
  for(int t = 0; t < numTimeSteps; ++t) {
    Eigen::VectorXd x_sol_t = x_sol.segment(t * numParams, numParams);
    vhop::beta_t<double> beta;
    vhop::theta_t<double> theta;
    costs[t]->convert2SMPL(x_sol_t, beta, theta);
    vhop::utility::writeSMPLParameters(outputPaths[t], beta, theta);
    if(!imagePaths.empty()) {
      costs[t]->drawReProjections(x_sol_t, imagePaths[t], outputPaths[t] + ".png");
    }
  }

  // Cleaning up the residuals, as we took ownership from ceres.
  for(auto cost : costs) {
    delete cost;
  }
  costs.clear();
  return true;
}

template<typename RPEResidualClass, size_t numTimeSteps>
bool vhop::Pipeline<RPEResidualClass, numTimeSteps>::process(
    const std::string &filePath,
    const std::string &outputPath,
    const std::string &imagePath) const {
  std::vector<std::string> filePaths = {filePath};
  std::vector<std::string> outputPaths = {outputPath};
  std::vector<std::string> imagePaths;
  if(!imagePath.empty()) {
    imagePaths.push_back(imagePath);
  }
  return process(filePaths, outputPaths, imagePaths);
}

template<typename RPEResidualClass, size_t numTimeSteps>
bool vhop::Pipeline<RPEResidualClass, numTimeSteps>::addReProjectionCostFunction(
    const std::string &filePath,
    const size_t offset,
    vhop::RPEResidualBase **cost,
    double* x0,
    ceres::Problem &problem) const {
  auto *costPtr = new RPEResidualClass(filePath, smpl_model_, offset);
  *cost = costPtr;
  constexpr int numParams = RPEResidualClass::getNumParams() * numTimeSteps;
  constexpr int numResiduals = RPEResidualClass::getNumResiduals();
  ceres::CostFunction *costFunction = new ceres::NumericDiffCostFunction<
      RPEResidualClass, ceres::CENTRAL, numResiduals, numParams>(costPtr);

  ceres::LossFunction *lossFunction = new ceres::CauchyLoss(1.0);
  Eigen::VectorXd x0_eigen = (*cost)->x0();
  for(int i = 0; i < numParams; i++)
    x0[i] = x0_eigen[i];
  problem.AddResidualBlock(costFunction, lossFunction, x0);
  return true;
}

template<typename RPEResidualClass, size_t numTimeSteps>
bool vhop::Pipeline<RPEResidualClass, numTimeSteps>::addConstantVelocityCostFunction(
    double* x0,
    ceres::Problem &problem) const {
  if(numTimeSteps < 2)
    throw std::runtime_error("Cannot add constant velocity cost function for less than 2 time steps");

  constexpr int numParamsPerT = RPEResidualClass::getNumParams();
  auto *cost = new vhop::ConstantMotionError<numParamsPerT, numTimeSteps>();

  // Ceres requires a cost function with residuals > 0 during compile time. Therefore, we hack
  // it by adding at least one residual, even if the function is not actually called in this case.
  constexpr int numResiduals = std::max(vhop::ConstantMotionError<numParamsPerT, numTimeSteps>::getNumResiduals(), 1);
  constexpr int numParams = RPEResidualClass::getNumParams() * numTimeSteps;
  static_assert(numParams == vhop::ConstantMotionError<numParamsPerT, numTimeSteps>::getNumParams(),
                "Non-Matching parameters of motion residual and re-projection residual blocks");
  ceres::CostFunction *costFunction = new ceres::NumericDiffCostFunction<
      vhop::ConstantMotionError<numParamsPerT, numTimeSteps>, ceres::CENTRAL, numResiduals, numParams>(cost);

  problem.AddResidualBlock(costFunction, nullptr, x0);
  return true;
}

#endif //VHOP_INCLUDE_VHOP_PIPELINE_IMPL_HPP_
