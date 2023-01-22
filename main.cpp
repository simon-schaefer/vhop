#include "ceres/ceres.h"
#include <Eigen/Core>

#include "vhop/constants.h"
#include "vhop/smpl_model.h"
#include "vhop/utility.h"
#include "vhop/visualization.h"

#include "vposer/constants.h"
#include "vposer/VPoser.h"

#include "vhop/residuals/rp_smpl.h"
#include "vhop/residuals/rp_vposer.h"


int main(int argc, char** argv) {
    cnpy::npz_t npz = cnpy::npz_load("../data/zju-mocap/sample.npz");
    cnpy::npz_t npz_means = cnpy::npz_load("../data/vposer_mean.npz");
    vhop::beta_t<double> beta = vhop::utility::loadDoubleMatrix(npz.at("betas"), vhop::SHAPE_BASIS_DIM, 1);
    Eigen::Matrix3d K = vhop::utility::loadDoubleMatrix(npz.at("intrinsics"), 3, 3);
    Eigen::Matrix4d T_C_B = vhop::utility::loadDoubleMatrix(npz.at("T_C_B"), 4, 4);
    vhop::joint_op_2d_t<double> joints_2d_gt = vhop::utility::loadDoubleMatrix(npz.at("keypoints_2d"), vhop::JOINT_NUM_OP, 2);
    vhop::joint_op_scores_t joints_2d_scores = vhop::utility::loadDoubleMatrix(npz.at("keypoints_2d_scores"), vhop::JOINT_NUM_OP, 1);

    vhop::SMPL smpl_model("../data/smpl_neutral.npz");
    VPoser vposer_model("../data/vposer_weights.npz", 512);
    vposer::latent_t<double> z = vposer::latent_t<double>::Zero();
    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::CauchyLoss(1.0);
    ceres::CostFunction* cost_function = new ceres::NumericDiffCostFunction<
        vhop::ReprojectionErrorVPoser, ceres::CENTRAL, vhop::JOINT_NUM_OP * 2, vposer::LATENT_DIM>(
        new vhop::ReprojectionErrorVPoser(beta, vposer_model, K, T_C_B, joints_2d_gt, joints_2d_scores, smpl_model));
    problem.AddResidualBlock(cost_function, loss_function, z.data());

//    vhop::theta_t<double> pose = vhop::theta_t<double>::Zero();
//    ceres::Problem problem;
//    ceres::LossFunction* loss_function = new ceres::CauchyLoss(1.0);
//    ceres::CostFunction* cost_function = new ceres::NumericDiffCostFunction<
//        vhop::ReprojectionErrorSMPL, ceres::CENTRAL, vhop::JOINT_NUM_OP * 2, vhop::JOINT_NUM * 3>(
//        new vhop::ReprojectionErrorSMPL(beta, K, T_C_B, joints_2d_gt, joints_2d_scores, smpl_model));
//    problem.AddResidualBlock(cost_function, loss_function, pose.data());

    ceres::Solver::Options options;
    options.max_num_iterations = 40;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    vhop::joint_op_2d_t<double> joints_2d;
    vhop::rotMats_t<double> rotMats = vposer_model.decode(z);
    smpl_model.ComputeOpenPoseKP(beta, rotMats, T_C_B, K, &joints_2d);
    // smpl_model.ComputeOpenPoseKP(beta, pose, T_C_B, K, &joints_2d);
    vhop::visualization::drawKeypoints("../data/zju-mocap/sample.jpg",
                                       joints_2d.cast<int>(),
                                       joints_2d_gt.cast<int>(),
                                       "../data/results/main.png");
//    vhop::utility::writeSMPLParameters("../data/results/smpl_params.bin", beta, pose);

    return 0;
}
