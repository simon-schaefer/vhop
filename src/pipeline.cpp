#include "vhop/pipeline.h"

#include "vhop/ceres/rpe_smpl.h"
#include "vhop/ceres/rpe_vposer.h"


vhop::Pipeline::Pipeline(const vhop::SMPL &smpl) : smpl_model_(smpl) {}

bool vhop::Pipeline::process(const std::string &filePath,
                             const std::string &outputPath,
                             const vhop::Methods &method,
                             bool doPrintSummary) {
    vhop::ResidualBase *cost;
    ceres::CostFunction *costFunction;
    if(method == Methods::smplx) {
        setSMPLCostFunction(filePath, &cost, &costFunction);
    } else if(method == Methods::vposerx) {
        setVPoserCostFunction(filePath, &cost, &costFunction);
    } else {
        std::cout << "Unknown method " << method << std::endl;
        return false;
    }

    Eigen::VectorXd x0 = cost->x0();
    ceres::Problem::Options problemOptions;
    problemOptions.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problemOptions.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problemOptions.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem(problemOptions);
    ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
    problem.AddResidualBlock(costFunction, loss_function, x0.data());

    ceres::Solver::Options options;
    options.max_num_iterations = 40;
    options.minimizer_progress_to_stdout = doPrintSummary;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (doPrintSummary) std::cout << summary.FullReport() << std::endl;

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
    return true;
}

void vhop::Pipeline::setSMPLCostFunction(const std::string &filePath,
                                         vhop::ResidualBase **cost,
                                         ceres::CostFunction **costFunction) {
    auto *costPtr = new vhop::ReprojectionErrorSMPL(filePath, smpl_model_);
    *cost = costPtr;
    constexpr int numParams = vhop::ReprojectionErrorSMPL::getNumParams();
    constexpr int numResiduals = vhop::ReprojectionErrorSMPL::getNumResiduals();
    *costFunction = new ceres::NumericDiffCostFunction<
        vhop::ReprojectionErrorSMPL, ceres::CENTRAL, numResiduals, numParams>(costPtr);
}


void vhop::Pipeline::setVPoserCostFunction(const std::string &filePath,
                                           vhop::ResidualBase **cost,
                                           ceres::CostFunction **costFunction) {
    auto *costPtr = new vhop::ReprojectionErrorVPoser(filePath, smpl_model_);
    *cost = costPtr;
    constexpr int numParams = vhop::ReprojectionErrorVPoser::getNumParams();
    constexpr int numResiduals = vhop::ReprojectionErrorVPoser::getNumResiduals();
    *costFunction = new ceres::NumericDiffCostFunction<
        vhop::ReprojectionErrorVPoser, ceres::CENTRAL, numResiduals, numParams>(costPtr);
}
