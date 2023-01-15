#ifndef VHOP_LATENTDIST_H
#define VHOP_LATENTDIST_H

#include <Eigen/Dense>
#include <iomanip>
#include <map>
#include <random>


class LatentDist {

    public:
        LatentDist(Eigen::MatrixXd mu, Eigen::MatrixXd sigma);
        Eigen::MatrixXd sample();

    private:
        Eigen::MatrixXd mMu;
        Eigen::MatrixXd mSigma;
        // Will we need rd or gen for each dist
        std::vector<std::normal_distribution<>> dists;
        std::random_device rd{};
        std::mt19937 gen{rd()};
};


#endif //VHOP_LATENTDIST_H
