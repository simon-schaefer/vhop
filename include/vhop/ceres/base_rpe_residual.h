#ifndef VHOP_CERES_BASE_RPE_RESIDUAL
#define VHOP_CERES_BASE_RPE_RESIDUAL

#include <cnpy.h>
#include <Eigen/Dense>
#include <utility>

#include "vhop/smpl_model.h"
#include "vhop/visualization.h"
#include "vhop/ceres/base_residual.h"

namespace vhop {

template<size_t N_TIME_STEPS>
class RPEResidualBase : public ResidualBase {

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  RPEResidualBase(const std::vector<std::string>& dataFilePaths, vhop::SMPL smpl_model)
  : smpl_model_(std::move(smpl_model)) {
    static_assert(N_TIME_STEPS > 0, "N_TIME_STEPS must be greater than 0");
    if(dataFilePaths.size() != N_TIME_STEPS) {
      std::cerr << "dataFilePaths.size() must be equal to N_TIME_STEPS" << std::endl;
      exit(-1);
    }

    for(const auto& dataFilePath : dataFilePaths) {
      cnpy::npz_t npz = cnpy::npz_load(dataFilePath);
      K_.emplace_back(vhop::utility::loadDoubleMatrix(npz.at("intrinsics"), 3, 3));
      T_C_B_.emplace_back(vhop::utility::loadDoubleMatrix(npz.at("T_C_B"), 4, 4));
      joint_kps_.emplace_back(vhop::utility::loadDoubleMatrix(npz.at("keypoints_2d"), vhop::JOINT_NUM_OP, 2));
      joint_kps_scores_.emplace_back(vhop::utility::loadDoubleMatrix(npz.at("keypoints_2d_scores"), vhop::JOINT_NUM_OP, 1));
    }
  };

  // @brief Compute the residual for the given parameters.
  // The residuals consist of two parts:
  // 1) For each image, the re-projection error to the OpenPose 2D key-points is computed.
  // 2) For multiple time-steps, the estimated parameters are constraint to be in zero/constant motion.
  //
  // @param params The optimization parameters.
  // @param residuals The residual values to be computed and overwritten.
  // @return true if the computation was successful, false otherwise.
  bool operator()(const double *params, double *re_projection_error) const override {
    Eigen::VectorXd params_eigen;
    convert2Eigen(params, params_eigen);

    // Compute the OpenPose re-projection based on the optimization parameters.
    AlignedVector<vhop::joint_op_2d_t<double>> joints2d;
    bool success = computeReProjection(params_eigen, joints2d);
    if(!success) {
      std::cerr << "Failed to compute re-projection from optimization parameters" << std::endl;
      return false;
    }

    // For each joint and time-step, compute the (weighted) re-projection error.
    for(int t = 0; t < N_TIME_STEPS; t++) {
      const size_t offset_t = t * vhop::JOINT_NUM_OP * 2;
      for (int i = 0; i < vhop::JOINT_NUM_OP; ++i) {
        double score = RPEResidualBase<N_TIME_STEPS>::joint_kps_scores_[t](i);
        Eigen::Vector2d joint2d_gt = RPEResidualBase<N_TIME_STEPS>::joint_kps_[t].row(i);
        re_projection_error[offset_t + i * 2] = score * (joints2d[t](i, 0) - joint2d_gt(0));
        re_projection_error[offset_t + i * 2 + 1] = score * (joints2d[t](i, 1) - joint2d_gt(1));
      }
    }

    // In case of two timestamps, add a zero motion cost, i.e. the difference between the two re-projected
    // key-points should be zero.
    const size_t offset = N_TIME_STEPS * vhop::JOINT_NUM_OP * 2;
    if(N_TIME_STEPS == 2) {
      for (int i = 0; i < vhop::JOINT_NUM_OP; ++i) {
        double score = RPEResidualBase<N_TIME_STEPS>::joint_kps_scores_[0](i);
        re_projection_error[offset + i * 2] = score * (joints2d[0](i, 0) - joints2d[1](i, 0));
        re_projection_error[offset + i * 2 + 1] = score * (joints2d[0](i, 1) - joints2d[1](i, 1));
      }
    } else if (N_TIME_STEPS > 2) {
      // TODO: implement for more than two timestamps
      throw std::runtime_error("Not implemented yet");
    }

    return true;
  }

  // @brief Convert the optimization parameters to SMPL and write to .npz files.
  // The output files contain the following arrays:
  // - betas: The shape parameters [0:10].
  // - thetas: The pose parameters [10:72].
  // - execution_time: The execution time of the optimization [72].
  //
  // @param z The optimization parameters.
  // @param outputPaths The paths to the output files.
  // @param executionTime The execution time of the optimization.
  // @return true if the computation was successful, false otherwise.
  [[nodiscard]] bool writeSMPLParameters(const Eigen::VectorXd& z,
                                         const std::vector<std::string>& outputPaths,
                                         double executionTime) const override {
    if(outputPaths.size() != N_TIME_STEPS) {
      std::cerr << "outputPaths.size() must be equal to N_TIME_STEPS" << std::endl;
      return false;
    }

    // Convert optimization parameters z to SMPL parameters.
    AlignedVector<beta_t<double>> betas;
    AlignedVector<theta_t<double>> thetas;
    convert2SMPL(z, betas, thetas);

    // For each time step, write the SMPL parameters to the outputPaths.
    for(size_t t = 0; t < N_TIME_STEPS; ++t) {
      vhop::utility::writeSMPLParameters(outputPaths[t],betas[t], thetas[t], executionTime);
    }
    return true;
  }

  // @brief Compute the OpenPose re-projections based on the optimization parameters and visualize
  // them on the images. The re-projections are drawn in red, the ground-truth in green.
  //
  // @param params The optimization parameters.
  // @param imagePath The path to the images.
  // @param outputImagePath The path to the output images.
  // @return true if the computation was successful, false otherwise.
  [[nodiscard]] bool drawReProjections(const Eigen::VectorXd& params,
                                       const std::vector<std::string>& imagePath,
                                       const std::vector<std::string>& outputImagePath) const override {
    if(imagePath.size() != N_TIME_STEPS || outputImagePath.size() != N_TIME_STEPS) {
      std::cerr << "imagePath.size() and outputImagePath.size() must be equal to N_TIME_STEPS" << std::endl;
      return false;
    }

    // Convert optimization parameters z to SMPL parameters.
    AlignedVector<vhop::joint_op_2d_t<double>> joints2d;
    bool success = computeReProjection(params, joints2d);
    if(!success) {
      std::cerr << "Failed to compute re-projection for drawing images" << std::endl;
      return false;
    }

    // For each time step, draw the re-projections and save them to the outputImagePath.
    for(size_t t = 0; t < N_TIME_STEPS; ++t) {
      vhop::visualization::drawKeypoints(imagePath[t],
                                         joints2d[t].cast<int>(),
                                         joint_kps_[t].template cast<int>(),
                                         outputImagePath[t]);
    }
    return true;
  }

  // N_TIME_STEPS * vhop::JOINT_NUM_OP * 2 + (N_TIME_STEPS - 1) * vhop::JOINT_NUM_OP * 2
  // = vhop::JOINT_NUM_OP * 2 * (N_TIME_STEPS + N_TIME_STEPS - 1)
  // = vhop::JOINT_NUM_OP * 2 * (2 * N_TIME_STEPS - 1)
  static constexpr int getNumResiduals() { return vhop::JOINT_NUM_OP * 2 * (2 * N_TIME_STEPS - 1); }
  // return the number of time steps
  static constexpr int getNumTimeSteps() { return N_TIME_STEPS; }

 protected:
  [[nodiscard]] bool computeReProjection(const Eigen::VectorXd& params,
                                         AlignedVector<vhop::joint_op_2d_t<double>>& joints2d) const {
    AlignedVector<beta_t<double>> betas;
    AlignedVector<theta_t<double>> thetas;
    convert2SMPL(params, betas, thetas);

    joints2d = AlignedVector<vhop::joint_op_2d_t<double>>(N_TIME_STEPS);
    for(int t = 0; t < N_TIME_STEPS; t++) {
      smpl_model_.ComputeOpenPoseKP<double>(betas[t],
                                            thetas[t],
                                            T_C_B_[t],
                                            K_[t],
                                            &joints2d[t]);
    }
    return true;
  }


  vhop::SMPL smpl_model_;

  AlignedVector<Eigen::Matrix3d> K_;
  AlignedVector<Eigen::Matrix4d> T_C_B_;
  AlignedVector<vhop::joint_op_2d_t<double>> joint_kps_;
  AlignedVector<vhop::joint_op_scores_t> joint_kps_scores_;
};

}

#endif //VHOP_CERES_BASE_RPE_RESIDUAL
