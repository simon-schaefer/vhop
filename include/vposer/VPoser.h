#ifndef VHOP_VPOSER_H
#define VHOP_VPOSER_H

#include "Linear.h"
#include "LatentDist.h"

#include "vhop/constants.h"

class VPoser {

public:
    VPoser(const std::string& weightsPath, int nn, int z_dim);
    ~VPoser() = default;
    void printModel();
    vhop::AlignedVector<Eigen::Matrix3d> forward(const Eigen::MatrixXd& input);
    LatentDist encode(const Eigen::MatrixXd& input);
    vhop::AlignedVector<Eigen::Matrix3d> decode(const Eigen::MatrixXd& input);
    vhop::AlignedVector<Eigen::Matrix3d> continuousRotReprDecoder(const Eigen::MatrixXd& decoderOut) const;

private:
    int num_joints = 21;
    int num_features = num_joints * 3;
    std::vector<BaseLayer*> encoder_layers;
    Linear* mu_layer;
    Linear* logvar_layer;
    std::vector<BaseLayer*> decoder_layers;
};


#endif //VHOP_VPOSER_H
