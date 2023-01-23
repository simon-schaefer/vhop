#include "ceres/ceres.h"
#include <Eigen/Core>
#include <memory>

#include "vhop/constants.h"
#include "vhop/smpl_model.h"
#include "vhop/utility.h"
#include "vhop/visualization.h"

#include "vhop/ceres/rpe_smpl.h"


void setSMPLCostFunction(const std::string& filePath,
                         const vhop::SMPL& smplModel,
                         vhop::ResidualBase** cost,
                         ceres::CostFunction** costFunction) {
    auto* costPtr = new vhop::ReprojectionErrorSMPL(filePath, smplModel);
    *cost = costPtr;
    constexpr int numParams = vhop::ReprojectionErrorSMPL::getNumParams();
    constexpr int numResiduals = vhop::ReprojectionErrorSMPL::getNumResiduals();
    *costFunction = new ceres::NumericDiffCostFunction<
        vhop::ReprojectionErrorSMPL, ceres::CENTRAL, numResiduals, numParams>(costPtr);
}

int process(const std::string& filePath,
            const std::string& imagePath,
            const std::string& outputPath,
            const std::string& rpImagePath,
            bool doPrintSummary = false) {
    vhop::SMPL smpl_model("../data/smpl_neutral.npz");
    vhop::ResidualBase* cost;
    ceres::CostFunction* costFunction;
    setSMPLCostFunction(filePath, smpl_model, &cost, &costFunction);

    Eigen::VectorXd x0 = cost->x0();
    ceres::Problem::Options problemOptions;
    problemOptions.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problemOptions.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problemOptions.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem(problemOptions);
    ceres::LossFunction* loss_function = new ceres::CauchyLoss(1.0);
    problem.AddResidualBlock(costFunction, loss_function, x0.data());

    ceres::Solver::Options options;
    options.max_num_iterations = 3; //40;
    options.minimizer_progress_to_stdout = doPrintSummary;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if(doPrintSummary) std::cout << summary.FullReport() << std::endl;

    vhop::beta_t<double> beta;
    vhop::theta_t<double> theta;
    cost->convert2SMPL(x0, beta, theta);
    vhop::utility::writeSMPLParameters(outputPath, beta, theta);
//    vhop::joint_op_2d_t<double> joints_2d;
//    smpl_model.ComputeOpenPoseKP(beta, theta, T_C_B, K, &joints_2d);
//    vhop::visualization::drawKeypoints(imagePath,
//                                       joints_2d.cast<int>(),
//                                       joints_2d_gt.cast<int>(),
//                                       rpImagePath);

    cost = nullptr;  // TODO: why cannot delete?
    delete loss_function;
    delete costFunction;
    return 0;
}


int main(int argc, char** argv) {
    const std::string directory = "../data/zju-mocap/processed";
    const std::filesystem::path outputDirectory("../data/results");
    const auto files = vhop::utility::listFilesRecursively(directory, ".npz");

    int i = 1;
    for(const auto& filePath : files) {
        std::cout << "Processing " << filePath << " [" << i << "/" << files.size() << "]" << std::endl;
        i++;

        const std::string cameraName = filePath.parent_path().stem();
        const std::string sequenceName = filePath.parent_path().parent_path().stem();
        const std::string fileName = filePath.stem().c_str();
        const std::filesystem::path outputDir = outputDirectory / sequenceName / cameraName;
        std::filesystem::create_directories(outputDir);

        std::filesystem::path filePathStem = filePath.parent_path() / fileName;
        process((filePath.parent_path() / (fileName + ".npz")).c_str(),
                (filePath.parent_path() / (fileName + ".jpg")).c_str(),
                (outputDir / (fileName +".bin")).c_str(),
                (outputDir / (fileName + "_rp.jpg")).c_str(),
                false);
    }
    return 0;
}
