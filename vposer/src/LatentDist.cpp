//
// Created by Burak on 10.01.2023.
//


#include <Eigen/Dense>
#include "../include/LatentDist.h"

LatentDist::LatentDist(Eigen::MatrixXd mu, Eigen::MatrixXd sigma){

    mMu = mu;
    mSigma = sigma;

    for(int i = 0 ; i < mu.rows(); i++){
        for(int j = 0 ; j < mu.cols(); j++){
            std::normal_distribution<> d{mu(i,j), sigma(i,j)};
            dists.push_back(d);
        }
    }
}

Eigen::MatrixXd LatentDist::sample(){

    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(mMu.rows(), mMu.cols());

    for(int i = 0; i < mMu.rows(); i++){
        for(int j = 0; j < mMu.cols(); j++){
            out(i,j) = dists[i*mMu.cols() + j](gen);
        }
    }
    return out;
}

Eigen::MatrixXd LatentDist::getMean() {return mMu;}
Eigen::MatrixXd LatentDist::getScale() {return mSigma;}