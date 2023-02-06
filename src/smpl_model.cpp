#include "vhop/smpl_model.h"

using namespace vhop;

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
    weightsVertexJoints_.resize(JOINT_NUM_EXTRA, JOINT_NUM);
    for (int j = 0; j < JOINT_NUM_EXTRA; j++) {
        int vertex_id = VERTEX_JOINT_IDXS[j];
        weightsVertexJoints_.row(j) = weightsMat.row(vertex_id);
    }
}
