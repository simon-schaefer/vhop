#ifndef VHOP_VPOSER_H
#define VHOP_VPOSER_H

#include "Linear.h"
#include "LatentDist.h"
#include "VposerOutputs.h"

class VPoser {

public:
    VPoser(const std::string weightsPath, int num_neurons, int latent_dim);
    ~VPoser() = default;
    void printModel();
    void loadParams();
    VposerOut forward(Eigen::MatrixXd input);
    LatentDist encode(Eigen::MatrixXd input);
    DecoderOut decode(Eigen::MatrixXd input);
    std::vector<Eigen::MatrixXd> postProcess(Eigen::MatrixXd decoderOut);
    Eigen::Vector3d matrot2aa(Eigen::MatrixXd decoderOut);


private:

    int num_joints = 21;
    int num_features = num_joints * 3;
    std::string mainWeightsPath = "";
    std::vector<BaseLayer*> encoder_layers;
    Linear* mu_layer;
    Linear* logvar_layer;
    std::vector<BaseLayer*> decoder_layers;
};


#endif //VHOP_VPOSER_H
