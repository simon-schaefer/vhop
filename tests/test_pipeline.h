#include <gtest/gtest.h>

#include "vhop/smpl_model.h"
#include "vhop/pipeline.h"


TEST(TestPipeline, TestSingleImageSMPL) {
    ceres::Solver::Options ceres_options;
    ceres_options.max_num_iterations = 20;
    vhop::SMPL smpl_model("../data/smpl_neutral.npz");
    vhop::Pipeline<vhop::ReProjectionErrorSMPL<1>> pipeline(smpl_model, ceres_options);

    const std::string dataFilePath = "../data/test/samples/sample.npz";
    const std::string outputFilePath = "../data/test/sample_output_smpl.bin";
    const std::string imageFilePath = "../data/test/samples/sample.jpg";
    bool success = pipeline.process(dataFilePath,
                                    outputFilePath,
                                    imageFilePath,
                                    true);
    assert(success);
}

TEST(TestPipeline, TestSingleImageVPoser) {
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = 20;
  vhop::SMPL smpl_model("../data/smpl_neutral.npz");
  vhop::Pipeline<vhop::ReProjectionErrorVPoser<1>> pipeline(smpl_model, ceres_options);

  const std::string dataFilePath = "../data/test/samples/sample.npz";
  const std::string outputFilePath = "../data/test/sample_output_vposer.bin";
  const std::string imageFilePath = "../data/test/samples/sample.jpg";
  bool success = pipeline.process(dataFilePath,
                                  outputFilePath,
                                  imageFilePath,
                                  true);
  assert(success);
}


TEST(TestPipeline, TestTwoImages) {
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = 40;
  ceres_options.minimizer_progress_to_stdout = true;
  vhop::SMPL smpl_model("../data/smpl_neutral.npz");
  vhop::Pipeline<vhop::ReProjectionErrorSMPL<2>> pipeline(smpl_model, ceres_options);

  std::vector<std::string> dataFilePaths = {"../data/test/samples/sample_31.npz",
                                            "../data/test/samples/sample_32.npz"};
  std::vector<std::string> outputFilePaths = {"../data/test/sample_output_21.bin",
                                              "../data/test/sample_output_22.bin"};
  std::vector<std::string> imageFilePaths = {"../data/test/samples/sample_31.jpg",
                                             "../data/test/samples/sample_32.jpg"};
  bool success = pipeline.process(dataFilePaths,
                                  outputFilePaths,
                                  imageFilePaths,
                                  true);
  assert(success);
}

TEST(TestPipeline, TestThreeImages) {
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = 40;
  ceres_options.minimizer_progress_to_stdout = true;
  vhop::SMPL smpl_model("../data/smpl_neutral.npz");
  vhop::Pipeline<vhop::ReProjectionErrorSMPL<3>> pipeline(smpl_model, ceres_options);

  std::vector<std::string> dataFilePaths = {"../data/test/samples/sample_31.npz",
                                            "../data/test/samples/sample_32.npz",
                                            "../data/test/samples/sample_33.npz"};
  std::vector<std::string> outputFilePaths = {"../data/test/sample_output_31.bin",
                                              "../data/test/sample_output_32.bin",
                                              "../data/test/sample_output_33.bin"};
  std::vector<std::string> imageFilePaths = {"../data/test/samples/sample_31.jpg",
                                             "../data/test/samples/sample_32.jpg",
                                             "../data/test/samples/sample_33.jpg"};
  bool success = pipeline.process(dataFilePaths,
                                  outputFilePaths,
                                  imageFilePaths,
                                  true);
  assert(success);
}
