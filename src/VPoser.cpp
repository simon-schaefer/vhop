#include <iostream>

#include "vposer/Linear.h"
#include "vposer/LeakyRelu.h"
#include "vposer/BatchNorm.h"
#include "vposer/VPoser.h"


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


VPoser::VPoser(const std::string& weightsPath, int nn, int z_dim) {
    cnpy::npz_t npz = cnpy::npz_load(weightsPath);

    // VPoser - Encoder
    encoder_layers.clear();
    encoder_layers.push_back(new BatchNorm(num_features, npz, "bodyprior_enc_bn1"));
    encoder_layers.push_back(new Linear(num_features, nn, npz, "bodyprior_enc_fc1"));
    encoder_layers.push_back(new LeakyRelu(0.2));
    encoder_layers.push_back(new BatchNorm(nn, npz, "bodyprior_enc_bn2"));
    encoder_layers.push_back(new Linear(nn, nn, npz, "bodyprior_enc_fc2"));

    // Vposer - NormalDistDecoder
    mu_layer = new Linear(nn, z_dim, npz, "bodyprior_enc_mu");
    logvar_layer = new Linear(nn, z_dim, npz, "bodyprior_enc_logvar");

    // VPoser - Decoder
    decoder_layers.clear();
    decoder_layers.push_back(new Linear(z_dim, nn, npz, "bodyprior_dec_fc1"));
    decoder_layers.push_back(new LeakyRelu(0.2));
    decoder_layers.push_back(new Linear(nn, nn, npz, "bodyprior_dec_fc2"));
    decoder_layers.push_back(new LeakyRelu(0.2));
    decoder_layers.push_back(new Linear(nn, num_joints*6, npz, "bodyprior_dec_out"));
}

void VPoser::printModel() {
    for(auto & encoder_layer : encoder_layers){
        encoder_layer->printDescription();
    }
    mu_layer->printDescription();
    logvar_layer->printDescription();
    for(auto & decoder_layer : decoder_layers){
        decoder_layer->printDescription();
    }
}

vhop::AlignedVector<Eigen::Matrix3d> VPoser::forward(const Eigen::MatrixXd& input) const {
    LatentDist latentDist = encode(input);
    Eigen::MatrixXd z = latentDist.sample();
    return decode(z);
}

LatentDist VPoser::encode(const Eigen::MatrixXd& input) const {
    Eigen::MatrixXd x = input;
    for(auto & encoder_layer : encoder_layers){
        x = encoder_layer->forward(x);
    }
    Eigen::MatrixXd mu_out = mu_layer->forward(x);
    Eigen::MatrixXd logvar_out = logvar_layer->forward(x);
    Eigen::MatrixXd sigma = softplusCustom(logvar_out);
    return {mu_out, sigma};  // LatentDist w/ brace initializer
}

vhop::AlignedVector<Eigen::Matrix3d> VPoser::decode(const vposer::latent_t<double>& input,
                                                    bool returnFullRotMats) const {
    Eigen::MatrixXd x = input;
    for(auto & decoder_layer : decoder_layers){
        x = decoder_layer->forward(x);
    }
    return continuousRotReprDecoder(x, returnFullRotMats);
}

vhop::AlignedVector<Eigen::Matrix3d> VPoser::continuousRotReprDecoder(const Eigen::MatrixXd& decoderOut,
                                                                      bool returnFullRotMats) const {
    assert(decoderOut.rows() == decoderOut.size());  // currently no batching supported
    assert(decoderOut.rows() == num_joints * 6);
    size_t encoding_size = num_joints * 6 * 21;

    // Original code rearranges the matrices in the form of (-1, 3, 2) then uses two arrays by changing the value of
    // last dimension as [-1, 3, 0] and [-1, 3, 1]
    // Below resized1 is used for [-1, 3, 0] and resized2 is used for [-1, 3, 1]
    Eigen::MatrixXd resized1(encoding_size, 3);
    Eigen::MatrixXd resized2(encoding_size, 3);

    // reshaped_input = module_input.view(-1, 3, 2)
    Eigen::VectorXd flatten_out = Eigen::VectorXd {decoderOut.transpose().reshaped()};
    for (int i = 0; i < encoding_size; ++i) {
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
    vhop::AlignedVector<Eigen::Matrix3d> rotMats(num_joints);
    for(int i = 0; i < 21; i++) {
        Eigen::Matrix3d rotMat;
        rotMat.block<3, 1>(0, 0) = b1.block<1, 3>(i, 0);
        rotMat.block<3, 1>(0, 1) = b2.block<1, 3>(i, 0);
        rotMat.block<3, 1>(0, 2) = b3.block<1, 3>(i, 0);
        rotMats.at(i) = rotMat;
    }

    // If returnFullRotMats is true, then the output is of size 24, otherwise it is of size 21.
    if(returnFullRotMats) {
        vhop::AlignedVector<Eigen::Matrix3d> fullRotMats(24);
        for(int i = 0; i < rotMats.size(); i++) {
            fullRotMats.at(i + 1) = rotMats.at(i);
        }
        fullRotMats.at(0) = Eigen::Matrix3d::Identity();  // root joint
        fullRotMats.at(22) = Eigen::Matrix3d::Identity();  // left hand joint
        fullRotMats.at(23) = Eigen::Matrix3d::Identity();  // right hand joint

        return fullRotMats;
    }
    return rotMats;
}
