//
// Created by Burak on 17.01.2023.
//

#ifndef VHOP_VPOSEROUTPUTS_H
#define VHOP_VPOSEROUTPUTS_H

#include <Eigen/Core>
#include "LatentDist.h"

struct DecoderOut{
    std::vector<Eigen::MatrixXd> poseBodyMatrot;
    std::vector<Eigen::MatrixXd> poseBody;
};

struct VposerOut{
    DecoderOut decoderOut;
    Eigen::MatrixXd sampledLatent;
    Eigen::MatrixXd latentMean;
    Eigen::MatrixXd latentScale;

};

#endif //VHOP_VPOSEROUTPUTS_H
