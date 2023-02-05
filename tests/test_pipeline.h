#include <gtest/gtest.h>

#include "vhop/smpl_model.h"
#include "vhop/pipeline.h"


TEST(TestPipeline, TestSingleImage) {
    ceres::Solver::Options ceres_options;
    ceres_options.max_num_iterations = 10;
    vhop::SMPL smpl_model("../data/smpl_neutral.npz");
    vhop::Pipeline<vhop::ReProjectionErrorSMPL, 1> pipeline(smpl_model, ceres_options);

    const std::string dataFilePath = "../data/test/sample.npz";
    const std::string outputFilePath = "../data/test/sample_output.bin";
    const std::string imageFilePath = "../data/test/sample.jpg";
    pipeline.process(dataFilePath, outputFilePath, imageFilePath);
}


TEST(TestPipeline, TestTwoImages) {
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = 40;
  vhop::SMPL smpl_model("../data/smpl_neutral.npz");
  vhop::Pipeline<vhop::ReProjectionErrorSMPL, 2> pipeline(smpl_model, ceres_options);

  std::vector<std::string> dataFilePaths = {"../data/test/sample.npz",
                                            "../data/test/sample.npz"};
  std::vector<std::string> outputFilePaths = {"../data/test/sample_output_21.bin",
                                              "../data/test/sample_output_22.bin"};
  std::vector<std::string> imageFilePaths = {"../data/test/sample.jpg",
                                             "../data/test/sample.jpg"};
  pipeline.process(dataFilePaths, outputFilePaths, imageFilePaths);
}
