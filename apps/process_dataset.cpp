#include <Eigen/Core>
#include <memory>

#include "vhop/utility.h"
#include "vhop/pipeline.h"


int main(int argc, char** argv) {
    const std::string directory = "../data/zju-mocap/";
    const std::string methodName = "smpl";
    ceres::Solver::Options ceres_options;
    ceres_options.max_num_iterations = 60;
    ceres_options.minimizer_progress_to_stdout = false;

    const std::filesystem::path outputDirectory("../data/results");
    const auto files = vhop::utility::listFilesRecursively(directory, ".npz");

    vhop::SMPL smpl_model("../data/smpl_neutral.npz");
    vhop::Pipeline<vhop::ReProjectionErrorSMPL, 1> pipeline(smpl_model, ceres_options);
    int i = 1;
    for(const auto& filePath : files) {
        std::cout << "Processing " << filePath << " [" << i << "/" << files.size() << "]" << std::endl;
        i++;

        const std::string cameraName = filePath.parent_path().stem();
        const std::string sequenceName = filePath.parent_path().parent_path().stem();
        const std::string fileName = filePath.stem().c_str();
        const std::filesystem::path outputDir = outputDirectory / methodName / sequenceName / cameraName;
        std::filesystem::create_directories(outputDir);

        std::filesystem::path filePathStem = filePath.parent_path() / fileName;
        const std::string imageFilePath = (filePath.parent_path() / (fileName + ".jpg")).c_str();
        const std::string dataFilePath = (filePath.parent_path() / (fileName + ".npz")).c_str();
        const std::string outputFilePath = (outputDir / (fileName +".bin")).c_str();

        bool success = pipeline.process(dataFilePath, outputFilePath, imageFilePath);
        if(!success) {
            std::cout << "... failed to process " << filePath << std::endl;
        }
    }
    return 0;
}
