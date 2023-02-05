#ifndef VHOP_VPOSER_H
#define VHOP_VPOSER_H

#include "Linear.h"
#include "LatentDist.h"

#include "vhop/constants.h"
#include "constants.h"

class VPoser {

public:
    VPoser(const std::string& weightsPath, int nn, int z_dim = vposer::LATENT_DIM);
    ~VPoser() = default;
    void printModel();
    [[nodiscard]] vhop::rotMats_t<double> forward(const Eigen::MatrixXd& input) const;
    [[nodiscard]] LatentDist encode(const Eigen::MatrixXd& input) const;
    [[nodiscard]] vhop::rotMats_t<double> decode(const vposer::latent_t<double>& z,
                                                 bool returnFullRotMats = true) const;
    [[nodiscard]] vhop::rotMats_t<double> continuousRotReprDecoder(const Eigen::MatrixXd& decoderOut,
                                                                   bool returnFullRotMats = true) const;

private:
    int num_joints = 21;
    int num_features = num_joints * 3;
    std::vector<BaseLayer*> encoder_layers;
    Linear* mu_layer;
    Linear* logvar_layer;
    std::vector<BaseLayer*> decoder_layers;
};


#endif //VHOP_VPOSER_H
