#ifndef VHOP_INCLUDE_VHOP_SMPL_MODEL_IMPL_H_
#define VHOP_INCLUDE_VHOP_SMPL_MODEL_IMPL_H_

#include <cnpy.h>
#include <vector>

#include "vhop/smpl_model.h"
#include "vhop/utility.h"

namespace vhop {

template<typename T>
bool SMPL::Forward(const beta_t<double> &beta,
                   const theta_t<T> &theta,
                   joint_t<T> *joints,
                   vertex_t<T> *vertices) const {
    ///> compute rotation matrices
    vhop::rotMats_t<T> rotMats;
    rotMats.reserve(JOINT_NUM);
    for (size_t i = 0; i < JOINT_NUM; i++) {
        const Eigen::Matrix<T, 3, 1> theta_i = theta.template segment<3>(i * 3);
        const Eigen::Matrix<T, 3, 3> rotMat_i = utility::rodriguesMatrix(theta_i);
        rotMats.push_back(rotMat_i);
    }

    return Forward(beta, rotMats, joints, vertices);
}

template<typename T>
bool SMPL::Forward(const beta_t<double> & beta,
                   const rotMats_t<T>& rotMats,
                   joint_t<T>* joints,
                   vertex_t<T>* vertices) const {
    assert(rotMats.size() == vhop::JOINT_NUM);

    Eigen::Matrix<double, POSE_BASIS_DIM, 1> poseFeatures;
    for (int i = 0; i < JOINT_NUM; i++) {
        ///> compute pose features from rotation matrices (by subtracting identity)
        if (i > 0) {
            const Eigen::Matrix3d poseFeature_i = rotMats[i] - Eigen::Matrix3d::Identity();
            const int j = i - 1;
            poseFeatures.block(j * 9 + 0, 0, 3, 1) = poseFeature_i.row(0).transpose();
            poseFeatures.block(j * 9 + 3, 0, 3, 1) = poseFeature_i.row(1).transpose();
            poseFeatures.block(j * 9 + 6, 0, 3, 1) = poseFeature_i.row(2).transpose();
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    ///> shaped rest shape
    Eigen::MatrixXd restShape(VERTEX_NUM, 3);
    restShape = restShape_;
    restShape.col(0) += shapeBlendBasis_0 * beta;
    restShape.col(1) += shapeBlendBasis_1 * beta;
    restShape.col(2) += shapeBlendBasis_2 * beta;

    ///> rest joints from joint regressor: (JOINT_NUM x VERTEX_NUM) x (VERTEX_NUM x 3)
    const Eigen::Matrix<double, JOINT_NUM, 3> jointsRest = jointRegressor_ * restShape;

    ///> relative rest joints
    Eigen::MatrixXd jointsRelative = jointsRest;

    ///> posed joints
    joint_t<double> jointsPosed;

    ///> relative transforms
    AlignedVector<Eigen::Matrix4d> jointTransformMats;
    AlignedVector<Eigen::Matrix4d> jointTransformsChain;
    AlignedVector<Eigen::Matrix4d> relativeJointTransforms;
    Eigen::Matrix<double, JOINT_NUM, 16> relativeJointTransforms_A;

    ///> initialize first joints transformation matrix
    Eigen::Matrix4d jointTransformMat_0 = Eigen::Matrix4d::Identity();
    jointTransformMat_0.topLeftCorner<3, 3>() = rotMats[0];
    jointTransformMat_0.topRightCorner<3, 1>() = jointsRelative.row(0);

    jointTransformMats.push_back(jointTransformMat_0);
    jointTransformsChain.push_back(jointTransformMat_0);

    jointsPosed.head<3>() = jointsRelative.row(0).transpose();

    ///> initialise first relative joint transformation (A in original SMPL implementation)
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> relativeJointTransform_0 = jointTransformMat_0;
    relativeJointTransform_0.topRightCorner<3, 1>() -=
        relativeJointTransform_0.topLeftCorner<3, 3>() * jointsRest.row(0).transpose();
    relativeJointTransforms.emplace_back(relativeJointTransform_0);
    relativeJointTransforms_A.row(0) = Eigen::Map<Eigen::RowVectorXd>(
        relativeJointTransform_0.data(), relativeJointTransform_0.size());

    for (int j = 1; j < JOINT_NUM; j++) {
        const uint32_t parent = kinematicTree_[j];
        jointsRelative.row(j) -= jointsRest.row(parent);

        Eigen::Matrix4d jointTransformMat_j = Eigen::Matrix4d::Identity();
        jointTransformMat_j.topLeftCorner<3, 3>() = rotMats[j];
        jointTransformMat_j.topRightCorner<3, 1>() = jointsRelative.row(j);

        jointTransformMats.push_back(jointTransformMat_j);

        const Eigen::Matrix4d jointTransformChain_j = jointTransformsChain[parent] * jointTransformMat_j;
        jointTransformsChain.push_back(jointTransformChain_j);

        jointsPosed.segment<3>(j * 3) = jointTransformChain_j.topRightCorner<3, 1>().transpose();

        const Eigen::Vector3d currRestJoint = jointsRest.row(j);
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> relativeJointTransform_j(jointTransformChain_j);
        const Eigen::Vector3d
            currRestJoint_transformed = relativeJointTransform_j.topLeftCorner<3, 3>() * currRestJoint;
        relativeJointTransform_j.topRightCorner<3, 1>() -= currRestJoint_transformed;
        relativeJointTransforms.emplace_back(relativeJointTransform_j);

        const Eigen::Map<Eigen::RowVectorXd> relativeJointTransforms_row(
            relativeJointTransform_j.data(), relativeJointTransform_j.size());
        relativeJointTransforms_A.row(j) = relativeJointTransforms_row;
    }

    if (joints != nullptr) {
        *joints = jointsPosed;
    }

    restShape.col(0) += poseBlendBasis_0 * poseFeatures;
    restShape.col(1) += poseBlendBasis_1 * poseFeatures;
    restShape.col(2) += poseBlendBasis_2 * poseFeatures;

    ///> JOINT_NUM_EXTRA x 16 (4 x 4 transformation)
    const Eigen::Matrix<double, JOINT_NUM_EXTRA, 16>
        weightedVertexJointTransformations = weightsVertexJoints_ * relativeJointTransforms_A; ///> aka: T
    vertex_t<T> transformedVertexJoints;
    transformedVertexJoints.setZero();
    AlignedVector<Eigen::Matrix4d> transformationMatricesVertexJoints;//(JOINT_NUM_EXTRA);
    transformationMatricesVertexJoints.resize(JOINT_NUM_EXTRA);
    for (int j = 0; j < JOINT_NUM_EXTRA; j++) {
        const int vertex_id = VERTEX_JOINT_IDXS[j];
        Eigen::MatrixXd tf_transpose = weightedVertexJointTransformations.row(j);
        tf_transpose.resize(4, 4);
        const Eigen::Matrix4d tf = tf_transpose.transpose(); ///> necessary cause stored in column major order
        transformationMatricesVertexJoints[j] = tf;

        const Eigen::Vector4d unposedVertex = restShape.row(vertex_id).homogeneous();
        transformedVertexJoints.template segment<3>(j * 3) = (tf * unposedVertex).head<3>();
    }

    if (vertices != nullptr) {
        *vertices = transformedVertexJoints;
    }
    return true;
}

template<typename T>
bool ComputeOpenPoseKPFromJoints(const joint_t<T>& joints,
                                 const vertex_t<T>& vertices,
                                 const Eigen::Matrix4d& T_C_B,
                                 const Eigen::Matrix3d& K,
                                 joint_op_2d_t<T>* keypointsOpenPose) {
    // Combine the base and the additional joints to get the full joint set.
    Eigen::Matrix<T, JOINT_NUM_TOTAL * 3, 1> fullJoints;
    fullJoints.template segment<JOINT_NUM * 3>(0) = joints;
    fullJoints.template segment<JOINT_NUM_EXTRA * 3>(JOINT_NUM * 3) = vertices;

    // Convert the joint set to the OpenPose format.
    joint_op_3d_t<T> jointsOpenPose3DFlat;
    for (size_t j = 0; j < JOINT_NUM_OP; j++) {
        int op_idx = OPENPOSE_JOINT_INDEXES[j];
        jointsOpenPose3DFlat.template segment<3>(j * 3) = fullJoints.template segment<3>(op_idx * 3);
    }
    Eigen::Matrix<T, vhop::JOINT_NUM_OP, 3> joints3d_B = jointsOpenPose3DFlat.reshaped(3, vhop::JOINT_NUM_OP).transpose();

    // Convert 3D joints to camera frame.
    Eigen::Matrix<T, vhop::JOINT_NUM_OP, 3> joints3d_C;
    for (int i = 0; i < vhop::JOINT_NUM_OP; i++) {
        Eigen::Vector<T, 4> joints3d_Bh_i(joints3d_B(i, 0), joints3d_B(i, 1), joints3d_B(i, 2), 1);
        Eigen::Vector4d joints3d_Ch_i = T_C_B * joints3d_Bh_i;
        joints3d_C.row(i) = joints3d_Ch_i.head(3).cast<T>();
    }

    // Pinhole camera projection.
    *keypointsOpenPose = vhop::utility::project(joints3d_C, K);
    return true;
}

template<typename T>
bool SMPL::ComputeOpenPoseKP(const beta_t<double> & beta,
                             const theta_t<T>& theta,
                             const Eigen::Matrix4d& T_C_B,
                             const Eigen::Matrix3d& K,
                             joint_op_2d_t<T>* keypointsOpenPose) const {
  joint_t<T> joints;
  vertex_t<T> vertices;
  Forward(beta, theta, &joints, &vertices);
  return ComputeOpenPoseKPFromJoints(joints, vertices, T_C_B, K, keypointsOpenPose);
}

template<typename T>
bool SMPL::ComputeOpenPoseKP(const beta_t<double> & beta,
                             const rotMats_t<T>& rotMats,
                             const Eigen::Matrix4d& T_C_B,
                             const Eigen::Matrix3d& K,
                             joint_op_2d_t<T>* keypointsOpenPose) const {
    joint_t<T> joints;
    vertex_t<T> vertices;
    Forward(beta, rotMats, &joints, &vertices);
    return ComputeOpenPoseKPFromJoints(joints, vertices, T_C_B, K, keypointsOpenPose);
}
}

#endif //VHOP_INCLUDE_VHOP_SMPL_MODEL_IMPL_H_
