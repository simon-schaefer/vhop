//
// Created by Burak on 9.01.2023.
//


#include <iostream>
#include "../include/Linear.h"
#include "../include/VPoser.h"
#include "../include/LeakyRelu.h"
#include "../include/BatchNorm.h"
#include "../include/Transformations.h"
#include "../include/VposerOutputs.h"
#include <Eigen/Core>

class SoftplusFunctor {
public:
    double operator()(const double& x) const { return std::log(1.0 + std::exp(x)); }
};

Eigen::MatrixXd softplus(const Eigen::MatrixXd& X) {
    return X.unaryExpr(SoftplusFunctor());
}

Eigen::MatrixXd softplusCustom(const Eigen::MatrixXd& X) {
    return (X.array().exp() + 1).log();
}


VPoser::VPoser(const std::string  weightsPath, int num_neurons, int latent_dim) {

    mainWeightsPath = weightsPath;
    // VPoser - Encoder
    encoder_layers.push_back(new BatchNorm(num_features));
    encoder_layers.push_back(new Linear(num_features, num_neurons));
    encoder_layers.push_back(new LeakyRelu());
    encoder_layers.push_back(new BatchNorm(num_neurons));
    encoder_layers.push_back(new Linear(num_neurons, num_neurons));
    encoder_layers.push_back(new Linear(num_neurons, num_neurons));

    // Vposer - NormalDistDecoder

    mu_layer = new Linear(num_neurons, latent_dim);
    logvar_layer = new Linear(num_neurons, latent_dim);

    decoder_layers.push_back(new Linear(latent_dim, num_neurons));
    decoder_layers.push_back(new LeakyRelu());
    decoder_layers.push_back(new Linear(num_neurons, num_neurons));
    decoder_layers.push_back(new LeakyRelu());
    decoder_layers.push_back(new Linear(num_neurons, this->num_joints*6));

}

void VPoser::printModel() {
    for(int i =0; i < encoder_layers.size(); i++){
        encoder_layers[i]->printDescription();
    }
    mu_layer->printDescription();
    logvar_layer->printDescription();

    for(int i =0; i < decoder_layers.size(); i++){
        decoder_layers[i]->printDescription();
    }
}

void VPoser::loadParams(){

    // First batch norm
    encoder_layers[0]->loadParams(mainWeightsPath + "encoder_net.1");
    // First linear
    encoder_layers[1]->loadParams(mainWeightsPath + "encoder_net.2");
    // Second batch norm
    encoder_layers[3]->loadParams(mainWeightsPath + "encoder_net.4");
    // Second linear
    encoder_layers[4]->loadParams(mainWeightsPath + "encoder_net.6");
    // Final linear
    encoder_layers[5]->loadParams(mainWeightsPath + "encoder_net.7");

    mu_layer->loadParams(mainWeightsPath + "encoder_net.8.mu");
    logvar_layer->loadParams(mainWeightsPath + "encoder_net.8.logvar");

    // First linear
    decoder_layers[0]->loadParams(mainWeightsPath + "decoder_net.0");
    // Second linear
    decoder_layers[2]->loadParams(mainWeightsPath + "decoder_net.3");
    // Final linear
    decoder_layers[4]->loadParams(mainWeightsPath + "decoder_net.5");

}

VposerOut VPoser::forward(Eigen::MatrixXd input) {

    LatentDist latentDist = encode(input);

    Eigen::MatrixXd latentSample = latentDist.sample();

    DecoderOut decoderOut = decode(latentSample);

    VposerOut modelOut {decoderOut, latentSample, latentDist.getMean(), latentDist.getScale()};

    return modelOut;
}

LatentDist VPoser::encode(Eigen::MatrixXd input) {

    Eigen::MatrixXd x = input;

    for(int i = 0; i < encoder_layers.size(); i++){
        x = encoder_layers[i]->forward(x);
    }

    Eigen::MatrixXd mu_out = mu_layer->forward(x);
    Eigen::MatrixXd logvar_out = logvar_layer->forward(x);
    Eigen::MatrixXd sigma = softplusCustom(logvar_out);

    return LatentDist(mu_out, sigma);
}

DecoderOut VPoser::decode(Eigen::MatrixXd input) {

    Eigen::MatrixXd x = input;
    int n_samples = input.rows();

    for(int i = 0; i < decoder_layers.size(); i++){
        x = decoder_layers[i]->forward(x);
    }

    std::vector<Eigen::MatrixXd> sampleOuts =  postProcess(x);


    int remaining_dim = sampleOuts.size() / n_samples;
    std::vector<Eigen::MatrixXd> sampleMatrots;

    std::vector<Eigen::MatrixXd> poseBodyMatrot;
    std::vector<Eigen::MatrixXd> poseBody;

    for(int i = 0; i < n_samples; i++){

        Eigen::MatrixXd curPoseBody = Eigen::MatrixXd::Zero(remaining_dim, 3);
        Eigen::MatrixXd curPoseBodyMatrot = Eigen::MatrixXd::Zero(remaining_dim, 9);

        for(int j = 0; j < remaining_dim; j++){

            Eigen::Vector3d angleRes = matrot2aa(sampleOuts[i*remaining_dim+j]);
            curPoseBody.row(j) = angleRes;

            for (int k =0 ; k < 3 ; k++){
                for(int kk =0; kk < 3; kk++){
                    curPoseBodyMatrot(j,k*3+kk) = sampleOuts[i*remaining_dim+j](k,kk);
                }
            }
        }
        poseBody.push_back(curPoseBody);
        poseBodyMatrot.push_back(curPoseBodyMatrot);
    }

    DecoderOut out {poseBody, poseBodyMatrot};

    return out;
}

// Code for ContinousRotReprDecoder
std::vector<Eigen::MatrixXd> VPoser::postProcess(Eigen::MatrixXd decoderOut) {

    // Calculate the dimensions that we will have after view
    int n_samples  = decoderOut.rows();
    int n_features = decoderOut.cols();
    int remaining_dim = n_samples*21;

    // Original code rearranges the matrices in the form of (-1, 3, 2) then uses two arrays by changing the value of
    // last dimension as [-1, 3, 0] and [-1, 3, 1]
    // Below resized1 is used for [-1, 3, 0] and resized2 is used for [-1, 3, 1]
    Eigen::MatrixXd resized1(remaining_dim, 3);
    Eigen::MatrixXd resized2(remaining_dim, 3);

    // reshaped_input = module_input.view(-1, 3, 2)
    Eigen::VectorXd flatten_out = Eigen::VectorXd {decoderOut.transpose().reshaped()};
    for (int i = 0; i < remaining_dim; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 2; ++k) {
                if(k == 0){
                    resized1(i,j) = flatten_out(i*(3*2)+j*2+k);
                }
                else{
                    resized2(i,j) =  flatten_out(i*(3*2)+j*2+k);
                }
            }
        }
    }
    // b1 = F.normalize(reshaped_input[:, :, 0], dim=1)
    Eigen::Matrix<double , Eigen::Dynamic, 3> b1 = resized1.rowwise().normalized();
    // dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
    Eigen::MatrixXd dp = b1.cwiseProduct(resized2);
    Eigen::VectorXd dot_product = dp.rowwise().sum();
    // b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
    Eigen::Matrix<double , Eigen::Dynamic, 3> b2 = resized2.array() - (b1.array().colwise() * dot_product.array());
    b2 = b2.rowwise().normalized();

    // b3 = torch.cross(b1, b2, dim=1)
    Eigen::Matrix<double , Eigen::Dynamic, 3> b3 = Eigen::Matrix<double, Eigen::Dynamic, 3>(b1.rows(), 3);
    for (int i = 0; i < b1.rows(); ++i) {
        b3.row(i) = b1.row(i).cross(b2.row(i));
    }

    // torch.stack([b1, b2, b3], dim=-1)
    // Not implemented
    std::vector<Eigen::MatrixXd> out;
    // Create N, 3x3 matrices
    for(int i =0; i < b1.rows(); i++){
        // The original code only stacks 3x3 matrices but in post process stage adds a column of zereos
        // In this implementation this padding is also handled here
        Eigen::MatrixXd current = Eigen::MatrixXd::Zero(3, 4);
        current(0, 0) = b1(i, 0);
        current(1, 0) = b1(i, 1);
        current(2, 0) = b1(i, 2);

        current(0, 1) = b2(i, 0);
        current(1, 1) = b2(i, 1);
        current(2, 1) = b2(i, 2);

        current(0, 2) = b3(i, 0);
        current(1, 2) = b3(i, 1);
        current(2, 2) = b3(i, 2);
        out.push_back(current);
    }
    return out;
}

Eigen::Vector3d VPoser::matrot2aa(Eigen::MatrixXd decoderOut) {
    return vhop::transformations::rotationMatrix2Angle(decoderOut);
}

