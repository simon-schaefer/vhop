#ifndef VHOP_CERES_BASE_RPE_RESIDUAL
#define VHOP_CERES_BASE_RPE_RESIDUAL

#include <cnpy.h>
#include <Eigen/Dense>
#include <utility>

#include "vhop/smpl_model.h"
#include "vhop/visualization.h"
#include "vhop/ceres/base_residual.h"

namespace vhop {

class RPEResidualBase : public ResidualBase {

public:
    RPEResidualBase(
        vhop::SMPL smpl_model,
        Eigen::Matrix3d K,
        Eigen::Matrix4d T_C_B,
        vhop::joint_op_2d_t<double> joint_kps,
        vhop::joint_op_scores_t kp_scores)
        : smpl_model_(std::move(smpl_model)),
          K_(std::move(K)),
          T_C_B_(std::move(T_C_B)),
          joint_kps_(std::move(joint_kps)),
          joint_kps_scores_(std::move(kp_scores)){};

    RPEResidualBase(const std::string& dataFilePath, vhop::SMPL smpl_model) : smpl_model_(std::move(smpl_model)) {
      cnpy::npz_t npz = cnpy::npz_load(dataFilePath);
      K_ = vhop::utility::loadDoubleMatrix(npz.at("intrinsics"), 3, 3);
      T_C_B_ = vhop::utility::loadDoubleMatrix(npz.at("T_C_B"), 4, 4);
      joint_kps_ = vhop::utility::loadDoubleMatrix(npz.at("keypoints_2d"), vhop::JOINT_NUM_OP, 2);
      joint_kps_scores_ = vhop::utility::loadDoubleMatrix(npz.at("keypoints_2d_scores"), vhop::JOINT_NUM_OP, 1);
    };

    virtual void convert2SMPL(const Eigen::VectorXd& z, beta_t<double>& beta, theta_t<double>& theta) = 0;

    bool drawReProjections(const Eigen::VectorXd& z, const std::string& imagePath, const std::string& outputImagePath) {
        beta_t<double> beta;
        theta_t<double> theta;
        convert2SMPL(z, beta, theta);

        vhop::joint_op_2d_t<double> joints_2d;
        smpl_model_.ComputeOpenPoseKP(beta, theta, T_C_B_, K_, &joints_2d);

        vhop::visualization::drawKeypoints(imagePath,
                                           joints_2d.cast<int>(),
                                           joint_kps_.cast<int>(),
                                           outputImagePath);
        return true;
    }

protected:
    vhop::SMPL smpl_model_;
    Eigen::Matrix3d K_;
    Eigen::Matrix4d T_C_B_;
    vhop::joint_op_2d_t<double> joint_kps_;
    vhop::joint_op_scores_t joint_kps_scores_;
};

}

#endif //VHOP_CERES_BASE_RPE_RESIDUAL
