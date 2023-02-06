#ifndef VHOP_INCLUDE_VHOP_PIPELINE_IMPL_HPP_
#define VHOP_INCLUDE_VHOP_PIPELINE_IMPL_HPP_

#include <chrono>
#include <utility>

#include "vhop/pipeline.h"


template<typename RPEResidualClass>
vhop::Pipeline<RPEResidualClass>::Pipeline(vhop::SMPL smpl, ceres::Solver::Options solverOptions, bool verbose)
    : smpl_model_(std::move(smpl)),
      ceres_options_(std::move(solverOptions)),
      verbose_(verbose) {}

template<typename RPEResidualClass>
bool vhop::Pipeline<RPEResidualClass>::process(
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

  constexpr int numParams = RPEResidualClass::getNumParams();
  auto* x0 = new double[numParams];
  // Add residual to the problem. For the sake of simplicity and to avoid unnecessary computations,
  // we add the same residual for all time steps.
  vhop::ResidualBase* cost;
  if(!addReProjectionCostFunction(filePaths, &cost, x0,problem)) {
    std::cout << "Failed to add re-projection cost function" << std::endl;
    return false;
  }

  // Solve the optimization problem.
  ceres::Solver::Summary summary;
  auto start = std::chrono::steady_clock::now();
  ceres::Solve(ceres_options_, &problem, &summary);
  auto end = std::chrono::steady_clock::now();
  auto executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
  if (verbose_) {
    std::cout << summary.FullReport() << std::endl;
    std::cout << "... optimization finished in " << executionTime <<" milliseconds" << std::endl;
  }

  // Write the results to output files and draw the re-projections (if image files are given).
  Eigen::Vector<double, numParams> x_sol(x0);
  if(!cost->writeSMPLParameters(x_sol, outputPaths, (double)executionTime)) {
    std::cerr << "Failed to write SMPL parameters to output files" << std::endl;
    return false;
  }

  if(!imagePaths.empty()) {
    std::vector<std::string> outputImagePaths;
    for(const auto& outputPath : outputPaths)
      outputImagePaths.push_back(outputPath + ".png");
    if(!cost->drawReProjections(x_sol, imagePaths, outputImagePaths)) {
      std::cerr << "Failed to draw re-projections" << std::endl;
      return false;
    }
  }

  // Cleaning up the residuals, as we took ownership from ceres.
  delete cost;
  return true;
}

template<typename RPEResidualClass>
bool vhop::Pipeline<RPEResidualClass>::process(
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

template<typename RPEResidualClass>
bool vhop::Pipeline<RPEResidualClass>::processDirectory(
    const std::string &directory,
    const std::string &outputDirectory) const {
  const std::filesystem::path outputDir(outputDirectory);
  const std::filesystem::path dir(directory);
  std::filesystem::create_directories(outputDirectory);
  const auto files = vhop::utility::listFiles(directory, ".npz", false);
  if(files.empty()) {
    std::cout << "No files found in " << directory << std::endl;
    return false;
  }

  int i = 1;
  for(const auto& filePath : files) {
    std::cout << "Processing " << filePath << " [" << i << "/" << files.size() << "]" << std::endl;
    i++;

    const std::string fileName = filePath.stem().c_str();
    const size_t numTimeSteps = RPEResidualClass::getNumTimeSteps();
    if (numTimeSteps == 1) {
      const std::string imageFilePath = (dir / (fileName + ".jpg")).c_str();
      const std::string outputFilePath = (outputDir / (fileName +".bin")).c_str();
      bool success = process(filePath, outputFilePath, imageFilePath);
      if(!success) {
        std::cout << "... failed to process " << filePath << std::endl;
      }
    } else {
      std::vector<std::string> filePaths, outputPaths, imagePaths;
      const std::filesystem::path outputDir_i(outputDir / fileName);
      std::filesystem::create_directories(outputDir_i);

      // Find the remaining files, if they do not exist, then skip the whole sequence.
      // The current file is the last file in the sequence.
      const int fileIndex = std::stoi(fileName);
      bool loaded = true;
      for(int t = 0; t < numTimeSteps; t++) {
        // Get name of subsequent file, note that the numbers are zero padded to a length of 6.
        std::stringstream ss;
        ss << std::setw(6) << std::setfill('0') << fileIndex - numTimeSteps + 1 + t;
        std::string fileName_t = ss.str();
        // Get the file paths of the data, output and image files.
        const std::string filePath_t = (dir / (fileName_t + ".npz")).c_str();
        const std::string imageFilePath = (dir / (fileName_t + ".jpg")).c_str();
        // To simplify evaluation later, we store the current file in the results' directory, while
        // moving the other files to a subdirectory.
        std::string outputFilePath;
        if(t == numTimeSteps - 1)
          outputFilePath = (outputDir / (fileName + ".bin")).c_str();
        else
          outputFilePath = (outputDir_i / (fileName_t + ".bin")).c_str();

        // Check if the file exists. If not, we skip the rest of the files and continue with the next one.
        if(!std::filesystem::exists(filePath_t) || !std::filesystem::exists(imageFilePath)) {
          loaded = false;
          break;
        }

        // Otherwise push the remaining files to the list.
        filePaths.push_back(filePath_t);
        outputPaths.push_back(outputFilePath);
        imagePaths.push_back(imageFilePath);
      }
      if(!loaded) {
        std::cout << "... failed to load all files for " << filePath << std::endl;
        continue;
      }

      // Process the sequence.
      bool success = process(filePaths, outputPaths, imagePaths);
      if(!success) {
        std::cout << "... failed to process " << filePath << std::endl;
      }
    }
  }
  return true;
}

template<typename RPEResidualClass>
bool vhop::Pipeline<RPEResidualClass>::addReProjectionCostFunction(
    const std::vector<std::string>& filePaths,
    vhop::ResidualBase **cost,
    double* x0,
    ceres::Problem &problem) const {
  auto *costPtr = new RPEResidualClass(filePaths, smpl_model_);
  *cost = costPtr;
  constexpr int numParams = RPEResidualClass::getNumParams();
  constexpr int numResiduals = RPEResidualClass::getNumResiduals();
  ceres::CostFunction *costFunction = new ceres::AutoDiffCostFunction<
      RPEResidualClass, numResiduals, numParams>(costPtr);

  ceres::LossFunction *lossFunction = new ceres::CauchyLoss(1.0);
  Eigen::VectorXd x0_eigen = (*cost)->x0();
  for(int i = 0; i < numParams; i++) {
    x0[i] = x0_eigen[i];
  }
  problem.AddResidualBlock(costFunction, lossFunction, x0);
  return true;
}


#endif //VHOP_INCLUDE_VHOP_PIPELINE_IMPL_HPP_
