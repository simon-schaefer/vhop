#include <Eigen/Core>
#include <memory>

#include "vhop/constants.h"
#include "vhop/utility.h"
#include "vhop/pipeline.h"


int main(int argc, char** argv) {
    const std::string directory = "../data/zju-mocap/";
    const std::vector<vhop::Methods> methods = {vhop::Methods::smplx, vhop::Methods::vposerx};
    ceres::Solver::Options ceres_options;
    ceres_options.max_num_iterations = 60;
    ceres_options.minimizer_progress_to_stdout = true;

    const std::filesystem::path outputDirectory("../data/results");
    const auto files = vhop::utility::listFilesRecursively(directory, ".npz");

    vhop::SMPL smpl_model("../data/smpl_neutral.npz");
    vhop::Pipeline pipeline(smpl_model, ceres_options, true);
    for(auto& method : methods) {
        const std::string methodName = vhop::utility::method2String(method);
        std::cout << "Processing method " << methodName << std::endl;
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

            pipeline.process(dataFilePath, outputFilePath, method, imageFilePath);
        }
    }
    return 0;
}
