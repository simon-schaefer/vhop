#ifndef VHOP_INCLUDE_VHOP_CERES_RPE_VPOSER_H_
#define VHOP_INCLUDE_VHOP_CERES_RPE_VPOSER_H_

#include "vhop/constants.h"
#include "vhop/utility.h"
#include "vposer/VPoser.h"
#include "vposer/constants.h"
#include "base_rpe_residual.h"

namespace vhop {

template<size_t N_TIME_STEPS>
class ReProjectionErrorVPoser : public vhop::RPEResidualBase<N_TIME_STEPS> {

 public:

  ReProjectionErrorVPoser(const std::vector<std::string>& dataFilePaths, const vhop::SMPL &smpl_model)
      : RPEResidualBase<N_TIME_STEPS>(dataFilePaths, smpl_model),
      vposer_("../data/vposer_weights.npz", 512) {
    for(const auto& dataFilePath : dataFilePaths) {
      cnpy::npz_t npz = cnpy::npz_load(dataFilePath);
      beta_.emplace_back(vhop::utility::loadDoubleMatrix(npz.at("betas"), vhop::SHAPE_BASIS_DIM, 1));
    }
  }

  // Get the initial parameters for the optimization.
  [[nodiscard]] Eigen::VectorXd x0() const override {
    const size_t numParams = getNumParams();
    return Eigen::VectorXd::Zero(numParams);
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
      vposer::latent_t<double> z_t = z.segment((long)(t * vposer::LATENT_DIM), vposer::LATENT_DIM);
      vhop::rotMats_t<double> rotMats = vposer_.decode(z_t, true);

      vhop::theta_t<double> theta_t;
      for(int i = 0; i < rotMats.size(); i++) {
        theta_t.segment<3>(i * 3) = vhop::utility::rodriguesVector(rotMats[i]);
      }

      betas.emplace_back(beta_[t]);
      thetas.emplace_back(theta_t);
    }
  }

  // @brief Convert the optimization parameters to an Eigen vector.
  // @param z The optimization parameters.
  // @param z_eigen The Eigen vector.
  void convert2Eigen(const double* z, Eigen::VectorXd& z_eigen) const override {
    z_eigen = Eigen::Vector<double, getNumParams()>(z);
  }

  // LATENT_DIM for each time step, so N_TIME_STEPS * LATENT_DIM.
  static constexpr int getNumParams() { return vposer::LATENT_DIM * N_TIME_STEPS; }

 private:
  VPoser vposer_;
  AlignedVector<beta_t<double>> beta_;
};

}

#endif //VHOP_INCLUDE_VHOP_CERES_RPE_VPOSER_H_
