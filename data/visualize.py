import balanna
import numpy as np
import smplx
import torch
import torram


smpl_model_file = "../../models/smpl/smpl_pkl/SMPL_NEUTRAL.pkl"
data_file = "zju-mocap/sample.npz"
prediction_file = "results/smpl_params.bin"


def compute_smpl_vertices(smpl_model: smplx.SMPL, betas: np.ndarray, thetas: np.ndarray, T: np.ndarray) -> np.ndarray:
    betas_t = torch.tensor(betas, dtype=torch.float32).view(1, 10)
    thetas_t = torch.tensor(thetas, dtype=torch.float32).view(1, 72)
    smpl_output = smpl_model.forward(betas_t, thetas_t[:, 3:], global_orient=thetas_t[:, :3])
    vertices = smpl_output.vertices

    T_t = torch.tensor(T, dtype=torch.float32).view(1, 4, 4)
    vertices = torram.geometry.transform_points(T_t, vertices)
    return vertices[0].detach().numpy()


def main():
    smpl_model = smplx.SMPL(smpl_model_file)
    data = np.load(data_file)
    data_hat = np.fromfile(prediction_file)

    vertices_gt = compute_smpl_vertices(smpl_model, data["betas"], data["thetas"], data["T_C_B"])
    vertices_hat = compute_smpl_vertices(smpl_model, data_hat[:10], data_hat[10:], data["T_C_B"])

    scene = balanna.trimesh.show_axis(np.eye(4))  # camera frame
    scene = balanna.trimesh.show_point_cloud(vertices_gt, colors=(0, 255, 0), scene=scene)
    scene = balanna.trimesh.show_point_cloud(vertices_hat, colors=(255, 0, 0), scene=scene)
    for _ in range(10):
        yield {"scene": scene, "image": data["image"]}


if __name__ == "__main__":
    balanna.display_scenes(main())
