#ifndef VHOP_INCLUDE_VHOP_SMPL_MODEL_H_
#define VHOP_INCLUDE_VHOP_SMPL_MODEL_H_

#include <string>
#include <memory>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cnpy.h>

#include "vhop/constants.h"


namespace vhop {

/**
 * @brief SMPL model class (SMPL: A Skinned Multi-Person Linear Model).
 * Inspired by https://github.com/sxyu/smplxpp
 */
class SMPL {
 protected:
  Eigen::MatrixXd shapeBlendBasis_0;
  Eigen::MatrixXd shapeBlendBasis_1;
  Eigen::MatrixXd shapeBlendBasis_2;

  Eigen::MatrixXd poseBlendBasis_0;
  Eigen::MatrixXd poseBlendBasis_1;
  Eigen::MatrixXd poseBlendBasis_2;

  Eigen::MatrixXd restShape_;
  Eigen::MatrixXd jointRegressor_;
  Eigen::Matrix<double, JOINT_NUM_EXTRA, JOINT_NUM> weightsVertexJoints_;

  // Hierarchy relation between joints, the root is at the belly button, (2, 24).
  std::vector<uint32_t> kinematicTree_;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Load model data stored into current application.
  explicit SMPL(const std::string &path);
  ~SMPL();

  /**
   * @brief Forward kinematics of SMPL model.
   * @param betas: shape parameters, (10, 1).
   * @param theta: pose parameters, (72, 1).
   * @param joints: output joints, (24 * 3, 1).
   * @param vertices: output extra joints, (21 * 3, 3).
   */
  template<typename T>
  bool Forward(const beta_t<double> & beta,
               const theta_t<T>& theta,
               joint_t<T>* joints,
               vertex_t<T>* vertices) const;

  /**
   * @brief Forward kinematics of SMPL model.
   * @param betas: shape parameters, (10, 1).
   * @param theta: pose parameters, (72, 1).
   * @param translation: root translation parameters, (3, 1).
   * @param joints: output joints, (24 * 3, 1).
   * @param vertices: output extra joints, (21 * 3, 3).
   */
  template<typename T>
  bool Forward(const beta_t<double> & beta,
               const theta_t<T>& theta,
               const translation_t<double>& translation,
               joint_t<T>* joints,
               vertex_t<T>* vertices) const;

  /**
   * @brief Forward kinematics of SMPL model outputting OpenPose joints.
   * @param betas: shape parameters, (10, 1).
   * @param theta: pose parameters, (72, 1).
   * @param translation: root translation parameters, (3, 1).
   * @param jointsOpenPose: output joints, (25 * 3, 1).
   */
  template<typename T>
  bool ForwardOpenPose(const beta_t<double> & beta,
                       const theta_t<T>& theta,
                       const translation_t<double>& translation,
                       joint_op_3d_t<T>* jointsOpenPose) const;

};

} // namespace vhop

#include "vhop/implementation/smpl_model.hpp"

#endif //VHOP_INCLUDE_VHOP_SMPL_MODEL_H_
