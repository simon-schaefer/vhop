#include <Eigen/Core>
#include <memory>

#include "vhop/smpl_model.h"
#include "vhop/utility.h"
#include "vhop/pipeline.h"


int main(int argc, char** argv) {
  if(argc != 4) {
      std::cout << "Usage: <dataset-directory> <method> <output-directory>" << std::endl;
      return 1;
  }
  const std::string smplPath = "../data/smpl_neutral.npz";
  const std::string directory = argv[1];
  const std::string methodName = argv[2];
  const std::string outputDirectory = argv[3];

  vhop::SMPL smplModel(smplPath);
  ceres::Solver::Options ceresOptions;
  ceresOptions.max_num_iterations = 60;
  ceresOptions.num_threads = 4;
  ceresOptions.minimizer_progress_to_stdout = false;

  bool success;
  if (methodName == "smpl") {
    vhop::Pipeline<vhop::ReProjectionErrorSMPL<1>> pipeline(smplModel, ceresOptions);
    success = pipeline.processDirectory(directory, outputDirectory);
  } else if (methodName == "vposer") {
    vhop::Pipeline<vhop::ReProjectionErrorVPoser<1>> pipeline(smplModel, ceresOptions);
    success = pipeline.processDirectory(directory, outputDirectory);
  } else if (methodName == "smpl+2") {
    vhop::Pipeline<vhop::ReProjectionErrorSMPL<2>> pipeline(smplModel, ceresOptions);
    success = pipeline.processDirectory(directory, outputDirectory);
  } else if (methodName == "vposer+2") {
    vhop::Pipeline<vhop::ReProjectionErrorVPoser<2>> pipeline(smplModel, ceresOptions);
    success = pipeline.processDirectory(directory, outputDirectory);
  } else if (methodName == "smpl+3") {
    vhop::Pipeline<vhop::ReProjectionErrorSMPL<3>> pipeline(smplModel, ceresOptions);
    success = pipeline.processDirectory(directory, outputDirectory);
  } else if (methodName == "vposer+3") {
    vhop::Pipeline<vhop::ReProjectionErrorVPoser<3>> pipeline(smplModel, ceresOptions);
    success = pipeline.processDirectory(directory, outputDirectory);
  } else {
    std::cout << "Unknown method: " << methodName << std::endl;
    return 1;
  }

  if (!success) {
    std::cout << "Failed to process dataset: " << directory << std::endl;
    return 1;
  }
  return 0;
}
