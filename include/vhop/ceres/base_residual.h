#ifndef VHOP_CERES_BASE_RESIDUAL_H
#define VHOP_CERES_BASE_RESIDUAL_H

#include <cnpy.h>
#include <Eigen/Dense>

#include "vhop/smpl_model.h"

namespace vhop {

class ResidualBase {

public:
  virtual ~ResidualBase() = default;

  // @brief Compute the residual for the given parameters.
  // @param params The optimization parameters.
  // @param residuals The residual values to be computed and overwritten.
  // @return true if the computation was successful, false otherwise.
  virtual bool operator()(const double *params, double *residuals) const = 0;

  // @brief Get the initial parameters for the optimization.
  [[nodiscard]] virtual Eigen::VectorXd x0() const = 0;

  // @brief Compute the OpenPose re-projections based on the optimization parameters and visualize
  // them on the images. The re-projections are drawn in red, the ground-truth in green.
  //
  // @param params The optimization parameters.
  // @param imagePath The path to the images.
  // @param outputImagePath The path to the output images.
  // @return true if the computation was successful, false otherwise.
  [[nodiscard]] virtual bool drawReProjections(const Eigen::VectorXd& params,
                                               const std::vector<std::string>& imagePath,
                                               const std::vector<std::string>& outputImagePath) const = 0;

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
  [[nodiscard]] virtual bool writeSMPLParameters(const Eigen::VectorXd& z,
                                                 const std::vector<std::string>& outputPaths,
                                                 double executionTime) const = 0;

  // @brief Convert the optimization parameters to SMPL parameters.
  // @param z The optimization parameters.
  // @param betas The SMPL shape parameters to overwrite.
  // @param thetas The SMPL pose parameters to overwrite.
  virtual void convert2SMPL(const Eigen::VectorXd& z,
                            AlignedVector<beta_t<double>>& betas,
                            AlignedVector<theta_t<double>>& thetas) const = 0;
  // @brief Convert the optimization parameters to an Eigen vector.
  // @param z The optimization parameters.
  // @param z_eigen The Eigen vector.
  virtual void convert2Eigen(const double* z, Eigen::VectorXd& z_eigen) const = 0;

};

}

#endif //VHOP_CERES_BASE_RESIDUAL_H
