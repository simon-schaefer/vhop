//
// Created by Burak on 15.01.2023.
//

#ifndef VHOP_TRANSFORMATIONS_H
#define VHOP_TRANSFORMATIONS_H

#include <Eigen/Dense>

namespace vhop::transformations {
    Eigen::Vector3d rotationMatrix2Angle(Eigen::MatrixXd rotationMatrix);
    Eigen::Vector4d rotationMatrix2Quartenion(Eigen::MatrixXd rotationMatrix);
    Eigen::Vector3d quartenion2Axis(Eigen::Vector4d quaternion);

};
#endif //VHOP_TRANSFORMATIONS_H
