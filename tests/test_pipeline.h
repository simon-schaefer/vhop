#include <gtest/gtest.h>

#include "vhop/smpl_model.h"
#include "vhop/pipeline.h"


TEST(TestPipeline, TestSingleImage) {
    ceres::Solver::Options ceres_options;
    ceres_options.max_num_iterations = 40;

    vhop::SMPL smpl_model("../data/smpl_neutral.npz");
    vhop::Pipeline pipeline(smpl_model, ceres_options, false);

    const std::string dataFilePath = "../data/test/sample.npz";
    const std::string outputFilePath = "../data/test/sample_output.bin";
    const std::string imageFilePath = "../data/test/sample.jpg";
    pipeline.process(dataFilePath, outputFilePath, vhop::Methods::smplx, imageFilePath);
}
