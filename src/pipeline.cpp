#include <utility>

#include "vhop/pipeline.h"

#include "vhop/ceres/rpe_smpl.h"
#include "vhop/ceres/rpe_vposer.h"


vhop::Pipeline::Pipeline(vhop::SMPL  smpl, ceres::Solver::Options solverOptions, bool verbose)
: smpl_model_(std::move(smpl)),
  ceres_options_(std::move(solverOptions)),
  verbose_(verbose) {}

bool vhop::Pipeline::process(const std::string &filePath,
                             const std::string &outputPath,
                             const vhop::Methods &method,
                             const std::string& imagePath) const {
    ceres::Problem::Options problemOptions;
    problemOptions.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problemOptions.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problemOptions.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem(problemOptions);

    vhop::RPEResidualBase *cost;
    Eigen::VectorXd x0;
    ceres::LossFunction *lossFunction;
    if (!addReProjectionCostFunction(filePath, method, &cost, &lossFunction, x0, problem)) {
        std::cout << "Failed to add reprojection cost function" << std::endl;
        return false;
    }

    ceres::Solver::Summary summary;
    ceres::Solve(ceres_options_, &problem, &summary);
    if (verbose_) std::cout << summary.FullReport() << std::endl;
    std::cout << "x0: " << x0.transpose() << std::endl;

    // vhop::beta_t<double> beta;
    // vhop::theta_t<double> theta;
    // cost->convert2SMPL(x0, beta, theta);
    // vhop::utility::writeSMPLParameters(outputPath, beta, theta);
    if(!imagePath.empty()) {
        cost->drawReProjections(x0, imagePath, outputPath + ".png");
    }

    cost = nullptr;
    return true;
}

bool vhop::Pipeline::addReProjectionCostFunction(const std::string &filePath,
                                                 const vhop::Methods &method,
                                                 vhop::RPEResidualBase **cost,
                                                 ceres::LossFunction **lossFunction,
                                                 Eigen::VectorXd &x0,
                                                 ceres::Problem &problem) const {
    ceres::CostFunction *costFunction;
    if(method == vhop::Methods::smplx) {
        auto *costPtr = new vhop::ReProjectionErrorSMPL(filePath, smpl_model_);
        *cost = costPtr;
        constexpr int numParams = vhop::ReProjectionErrorSMPL::getNumParams();
        constexpr int numResiduals = vhop::ReProjectionErrorSMPL::getNumResiduals();
        costFunction = new ceres::NumericDiffCostFunction<
            vhop::ReProjectionErrorSMPL, ceres::CENTRAL, numResiduals, numParams>(costPtr);
    } else if(method == vhop::Methods::vposerx) {
        auto *costPtr = new vhop::ReProjectionErrorVPoser(filePath, smpl_model_);
        *cost = costPtr;
        constexpr int numParams = vhop::ReProjectionErrorVPoser::getNumParams();
        constexpr int numResiduals = vhop::ReProjectionErrorVPoser::getNumResiduals();
        costFunction = new ceres::NumericDiffCostFunction<
            vhop::ReProjectionErrorVPoser, ceres::CENTRAL, numResiduals, numParams>(costPtr);
    } else {
        std::cout << "Unknown method " << method << std::endl;
        return false;
    }

    *lossFunction = new ceres::CauchyLoss(1.0);
    x0 = (*cost)->x0();
    problem.AddResidualBlock(costFunction, *lossFunction, x0.data());
    return true;
}
