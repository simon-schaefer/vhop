
#include <Eigen/Core>
#include <iostream>
#include "../include/Transformations.h"

/**
 * @brief Convert 3x4 rotation matrix to Rodrigues vector
 * @param 3x4 rotataionMatrix
 * @return Rodrigues vector transformation. Vector of size (3,)
 */
Eigen::Vector3d vhop::transformations::rotationMatrix2Angle(Eigen::MatrixXd rotataionMatrix) {
    Eigen::Vector4d transformationOut = vhop::transformations::rotationMatrix2Quartenion(rotataionMatrix);
    Eigen::Vector3d angleRes = vhop::transformations::quartenion2Axis(transformationOut);
    return angleRes;
}

/**
 * @brief Convert 3x4 rotation matrix to 4D quartenion vector
 * @param 3x4 rotataionMatrix
 * @return The rotation in quartenion. A vector of size (4,)
 */
Eigen::Vector4d vhop::transformations::rotationMatrix2Quartenion(Eigen::MatrixXd rotationMatrix) {
    double eps=1e-6;
    int n_row = rotationMatrix.rows();
    int n_col = rotationMatrix.cols();

    if(n_row != 3 && n_col != 4){
        throw std::invalid_argument( "The shape of the rotation matrix must be (3,4)" );
    }

    Eigen::MatrixXd trans_rotation = rotationMatrix.transpose();

    // Conversion of the line: mask_d2 = rmat_t[:, 2, 2] < eps
    // Since we have only one sample a bool will be enough
    // The same idea follows in other mask operations
    bool mask_d2 = trans_rotation(2, 2) < eps;
    bool mask_d0_d1  = trans_rotation(0, 0) > trans_rotation(1,1);
    bool mask_d0_nd1 = trans_rotation(0, 0) < -trans_rotation(1,1);

    double t0 = 1 + trans_rotation(0,0) - trans_rotation(1,1) - trans_rotation(2,2);
    Eigen::Vector4d q0{trans_rotation(1,2) - trans_rotation(2,1),
                       t0,
                       trans_rotation(0,1) + trans_rotation(1,0),
                       trans_rotation(2,0) + trans_rotation(0,2)
    };
    Eigen::Vector4d t0_rep{t0, t0, t0, t0};


    double t1 = 1 - trans_rotation(0,0) + trans_rotation(1,1) - trans_rotation(2,2);
    Eigen::Vector4d q1{trans_rotation(2,0) - trans_rotation(0,2),
                       trans_rotation(0,1) + trans_rotation(1,0),
                       t1,
                       trans_rotation(1,2) + trans_rotation(2,1)
    };
    Eigen::Vector4d t1_rep{t1, t1, t1, t1};


    double t2 = 1 - trans_rotation(0,0) - trans_rotation(1,1) + trans_rotation(2,2);
    Eigen::Vector4d q2{trans_rotation(0,1) - trans_rotation(1,0),
                       trans_rotation(2,0) + trans_rotation(0,2),
                       trans_rotation(1,2) + trans_rotation(2,1),
                       t2
    };
    Eigen::Vector4d t2_rep{t2, t2, t2, t2};


    double t3 = 1 + trans_rotation(0,0) + trans_rotation(1,1) + trans_rotation(2,2);
    Eigen::Vector4d q3{t3,
                       trans_rotation(1,2) - trans_rotation(2,1),
                       trans_rotation(2,0) - trans_rotation(0,2),
                       trans_rotation(0,1) - trans_rotation(1,0)
    };
    Eigen::Vector4d t3_rep{t3, t3, t3, t3};

    double mask_c0 = (double)(mask_d2 * mask_d0_d1);
    double mask_c1 = (double)(mask_d2 * (!mask_d0_d1));
    double mask_c2 = (double)((!mask_d2) * mask_d0_nd1);
    double mask_c3 = (double)((!mask_d2) * (!mask_d0_nd1));

    Eigen::Vector4d q = q0*mask_c0 + q1*mask_c1 + q2*mask_c2 + q3*mask_c3;
    Eigen::Vector4d divisor = (t0_rep * mask_c0 + t1_rep * mask_c1 + t2_rep * mask_c2 + t3_rep * mask_c3).cwiseSqrt();
    q = q.array() / divisor.array();

    return q * 0.5;

}

/**
 * @brief Convert quaternion vector to angle axis of rotation.
 * @param quartenion 4D vector with quartenion values
 * @return Tensor with angle axis of rotation a vector of size 3
 *
 */
Eigen::Vector3d vhop::transformations::quartenion2Axis(Eigen::Vector4d quaternion) {
    // TODO: Original code adapts a ceres code:  ceres-solver/include/ceres/rotation.h
    // After adding ceres, this function can be replaced with library call

    double q1 = quaternion(1);
    double q2 = quaternion(2);
    double q3 = quaternion(3);
    double sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3;
    double sin_theta = sqrt(sin_squared_theta);

    double cos_theta = quaternion(0);

    double two_theta = 0.0;
    if(cos_theta < 0.0){
        two_theta = 2.0 * atan2(-sin_theta, -cos_theta);
    }
    else{
        two_theta = 2.0 * atan2(sin_theta, cos_theta);
    }

    double k_pos = two_theta / sin_theta;
    double k_neg = 2.0;
    double k = k_pos;
    if(sin_squared_theta < 0.0){
        k = k_neg;
    }

    Eigen::Vector3d angle_axis = Eigen::Vector3d::Zero(3);
    angle_axis(0) += q1 * k;
    angle_axis(1) += q2 * k;
    angle_axis(2) += q3 * k;
    return angle_axis;
}
