#include "ceres/ceres.h"
#include <Eigen/Core>

#include "vhop/constants.h"
#include "vhop/reprojection_error.h"
#include "vhop/smpl_model.h"
#include "vhop/utility.h"
#include "vhop/visualization.h"


int main(int argc, char** argv) {
    cnpy::npz_t npz = cnpy::npz_load("../data/test/sample.npz");
    vhop::beta_t<double> beta = vhop::utility::loadDoubleMatrix(npz.at("betas"), vhop::SHAPE_BASIS_DIM, 1);
    Eigen::Matrix3d K = vhop::utility::loadDoubleMatrix(npz.at("intrinsics"), 3, 3);
    Eigen::Matrix4d T_C_B = vhop::utility::loadDoubleMatrix(npz.at("T_C_B"), 4, 4);
    vhop::joint_op_2d_t<double> joints_2d_gt = vhop::utility::loadDoubleMatrix(npz.at("keypoints_2d"), 25, 2);

    vhop::SMPL smpl_model("../data/smpl_neutral.npz");
    vhop::theta_t<double> pose;
    ceres::Problem problem;
    ceres::CostFunction* cost_function = new ceres::NumericDiffCostFunction<vhop::ReprojectionError, ceres::CENTRAL, 1, vhop::JOINT_NUM * 3>(
        new vhop::ReprojectionError(beta, K, T_C_B, joints_2d_gt, smpl_model));
    problem.AddResidualBlock(cost_function, nullptr, pose.data());

    ceres::Solver::Options options;
    options.max_num_iterations = 10;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 1;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    vhop::joint_op_2d_t<double> joints_2d;
    smpl_model.ComputeOpenPoseKP(beta, pose, T_C_B, K, &joints_2d);
    vhop::visualization::drawKeypoints("../data/test/sample.jpg",
                                       joints_2d.cast<int>(),
                                       joints_2d_gt.cast<int>(),
                                       "../data/test/main.png");

    return 0;
}
