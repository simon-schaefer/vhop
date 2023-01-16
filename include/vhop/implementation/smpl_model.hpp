#include <cnpy.h>
#include <vector>

#include "vhop/smpl_model.h"
#include "vhop/utility.h"

namespace vhop {

SMPL::SMPL(const std::string &path)
    : poseBlendBasis_0(VERTEX_NUM, POSE_BASIS_DIM),
      poseBlendBasis_1(VERTEX_NUM, POSE_BASIS_DIM),
      poseBlendBasis_2(VERTEX_NUM, POSE_BASIS_DIM),
      restShape_(VERTEX_NUM, 3),
      jointRegressor_(JOINT_NUM, VERTEX_NUM) {

    ///> load the entire npz file
    cnpy::npz_t model_npz = cnpy::npz_load(path);
    shapeBlendBasis_0 = utility::loadDoubleMatrix3D(model_npz["shape_blend_shapes"], VERTEX_NUM, SHAPE_BASIS_DIM, 0);
    shapeBlendBasis_1 = utility::loadDoubleMatrix3D(model_npz["shape_blend_shapes"], VERTEX_NUM, SHAPE_BASIS_DIM, 1);
    shapeBlendBasis_2 = utility::loadDoubleMatrix3D(model_npz["shape_blend_shapes"], VERTEX_NUM, SHAPE_BASIS_DIM, 2);
    poseBlendBasis_0 = utility::loadDoubleMatrix3D(model_npz["pose_blend_shapes"], VERTEX_NUM, POSE_BASIS_DIM, 0);
    poseBlendBasis_1 = utility::loadDoubleMatrix3D(model_npz["pose_blend_shapes"], VERTEX_NUM, POSE_BASIS_DIM, 1);
    poseBlendBasis_2 = utility::loadDoubleMatrix3D(model_npz["pose_blend_shapes"], VERTEX_NUM, POSE_BASIS_DIM, 2);

    jointRegressor_ = utility::loadDoubleMatrix(model_npz["joint_regressor"], JOINT_NUM, VERTEX_NUM);
    restShape_ = utility::loadDoubleMatrix(model_npz["vertices_template"], VERTEX_NUM, 3);

    auto *kt = model_npz["kinematic_tree"].data<uint32_t>();
    kinematicTree_.assign(kt, kt + JOINT_NUM * 2);

    std::vector<double> weights;  ///< linear blend skinning weights
    auto *w = model_npz["weights"].data<double>();
    weights.assign(w, w + VERTEX_NUM * JOINT_NUM);

    Eigen::MatrixXd weightsMat;
    weightsMat.resize(VERTEX_NUM, JOINT_NUM);
    for (int j = 0; j < JOINT_NUM; j++) {
        for (int v = 0; v < VERTEX_NUM; v++) {
            weightsMat(v, j) = weights[v * JOINT_NUM + j];
        }
    }
    for (int j = 0; j < JOINT_NUM_EXTRA; j++) {
        int vertex_id = VERTEX_JOINT_IDXS[j];
        weightsVertexJoints_.row(j) = weightsMat.row(vertex_id);
    }
}

SMPL::~SMPL() = default;

template<typename T>
bool SMPL::Forward(const beta_t<double> &beta,
                   const theta_t<T> &theta,
                   joint_t<T> *joints,
                   vertex_t<T> *vertices) const {
  ///> compute rotation matrices
  AlignedVector<Eigen::Matrix3d> rotMats;
  rotMats.reserve(JOINT_NUM);
  Eigen::Matrix<double, POSE_BASIS_DIM, 1> poseFeatures;
  for (size_t i = 0; i < JOINT_NUM; i++) {
    const Eigen::Vector3d theta_i = theta.template segment<3>(i * 3);

    ///> compute actual rotation matrix from rotation vector
    const Eigen::Matrix3d rotMat_i = utility::rodriguesMatrix(theta_i);
    rotMats.push_back(rotMat_i);

    ///> compute pose features from rotation matrices (by subtracting identity)
    if (i > 0) {
      const Eigen::Matrix3d poseFeature_i = rotMat_i - Eigen::Matrix3d::Identity();
      const size_t j = i - 1;
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
  relativeJointTransforms.push_back(relativeJointTransform_0);
  relativeJointTransforms_A.row(0) = Eigen::Map<Eigen::RowVectorXd>(
          relativeJointTransform_0.data(), relativeJointTransform_0.size());

  for (size_t j = 1; j < JOINT_NUM; j++) {
    const size_t parent = kinematicTree_[j];
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
    relativeJointTransforms.push_back(relativeJointTransform_j);

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
  for (size_t j = 0; j < JOINT_NUM_EXTRA; j++) {
    const size_t vertex_id = VERTEX_JOINT_IDXS[j];
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
bool SMPL::Forward(const beta_t<double> &beta,
                   const joint_t<T> &theta,
                   const translation_t<double> &translation,
                   joint_t<T> *joints,
                   vertex_t<T> *vertices) const {
  Forward(beta, theta, joints, vertices);
  if (joints != nullptr) {
    joint_t<T> jointsShift = translation.replicate<JOINT_NUM, 1>();
    *joints += jointsShift;
  }
  if (vertices != nullptr) {
    vertex_t<T> vertexShift = translation.replicate<JOINT_NUM_EXTRA, 1>();
    *vertices += vertexShift;
  }
  return true;
}

}