#include "ceres/ceres.h"
#include <Eigen/Core>

#include "vhop/constants.h"
#include "vhop/reprojection_error.h"
#include "vhop/smpl_model.h"
#include "vhop/utility.h"


int main(int argc, char** argv) {
    cnpy::npz_t npz = cnpy::npz_load("../data/test/sample.npz");
    vhop::beta_t<double> beta = vhop::utility::loadDoubleMatrix(npz.at("betas"), vhop::SHAPE_BASIS_DIM, 1);
    vhop::translation_t<double> translation = vhop::utility::loadDoubleMatrix(npz.at("translation"), 3, 1);
    Eigen::Matrix3d K = vhop::utility::loadDoubleMatrix(npz.at("intrinsics"), 3, 3);
    vhop::joint_op_2d_t<double> joints_kp = vhop::utility::loadDoubleMatrix(npz.at("keypoints_2d"), 25, 2);

    vhop::SMPL smpl_model("../data/smpl_neutral.npz");
    vhop::theta_t<double> pose;
    ceres::Problem problem;
//    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectionError, 1, vhop::JOINT_NUM * 3>(
//        new ReprojectionError(beta, translation, K, joints_kp, smpl_model));
    ceres::CostFunction* cost_function = new ceres::NumericDiffCostFunction<vhop::ReprojectionError, ceres::CENTRAL, 1, vhop::JOINT_NUM * 3>(
        new vhop::ReprojectionError(beta, translation, K, joints_kp, smpl_model));
    problem.AddResidualBlock(cost_function, nullptr, pose.data());

    ceres::Solver::Options options;
    options.max_num_iterations = 50;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;

    return 0;
}
