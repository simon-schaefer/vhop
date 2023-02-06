#ifndef VHOP_INCLUDE_VHOP_CERES_RPE_SMPL_H_
#define VHOP_INCLUDE_VHOP_CERES_RPE_SMPL_H_

#include <utility>

#include "vhop/constants.h"
#include "vhop/utility.h"
#include "base_rpe_residual.h"

namespace vhop {

template<size_t N_TIME_STEPS>
class ReProjectionErrorSMPL : public vhop::RPEResidualBase<N_TIME_STEPS> {

 public:

  ReProjectionErrorSMPL(const std::vector<std::string>& dataFilePaths, const vhop::SMPL &smpl_model)
      : RPEResidualBase<N_TIME_STEPS>(dataFilePaths, smpl_model) {
    for(const auto& dataFilePath : dataFilePaths) {
      cnpy::npz_t npz = cnpy::npz_load(dataFilePath);
      beta_.emplace_back(vhop::utility::loadDoubleMatrix(npz.at("betas"), vhop::SHAPE_BASIS_DIM, 1));
    }
  }

  // Get the initial parameters for the optimization.
  [[nodiscard]] Eigen::VectorXd x0() const override {
    // cnpy::npz_t npz_means = cnpy::npz_load("../data/vposer_mean.npz");
    int num_params = getNumParams();
    return Eigen::VectorXd::Zero(num_params);
  }

  // @brief Convert the optimization parameters to SMPL parameters.
  // @param z The optimization parameters.
  // @param betas The SMPL shape parameters to overwrite.
  // @param thetas The SMPL pose parameters to overwrite.
  void convert2SMPL(const Eigen::VectorXd& z,
                    AlignedVector<beta_t<double>>& betas,
                    AlignedVector<theta_t<double>>& thetas) const override {
    betas.clear();
    thetas.clear();
    for (size_t t = 0; t < N_TIME_STEPS; ++t) {
      betas.emplace_back(beta_[t]);
      thetas.emplace_back(z.segment((long)(t * vhop::THETA_DIM), vhop::THETA_DIM));
    }
  }

  // @brief Convert the optimization parameters to an Eigen vector.
  // @param z The optimization parameters.
  // @param z_eigen The Eigen vector.
  void convert2Eigen(const double* z, Eigen::VectorXd& z_eigen) const override {
    z_eigen = Eigen::Vector<double, getNumParams()>(z);
  }

  // THETA_DIM per time step, so N_TIME_STEPS * THETA_DIM
  static constexpr int getNumParams() { return vhop::THETA_DIM * N_TIME_STEPS; }

 private:
    AlignedVector<beta_t<double>> beta_;
};

}

#endif //VHOP_INCLUDE_VHOP_CERES_RPE_SMPL_H_
