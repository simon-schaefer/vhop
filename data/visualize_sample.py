import balanna
import numpy as np
import pathlib
import smplx
import torch
import torram


smpl_model_file = pathlib.Path("/home/wiss/sas/models/smpl/smpl_pkl/SMPL_NEUTRAL.pkl")
data_file = pathlib.Path("/home/wiss/sas/vhop/data/test/sample.npz")


def main():
    smpl_model = smplx.SMPL(smpl_model_file)
    data = np.load(data_file.as_posix())

    betas = torch.tensor(data["betas"], dtype=torch.float32).view(1, 10)
    thetas = torch.tensor(data["thetas"], dtype=torch.float32).view(1, 72)
    T = torch.tensor(data["T_C_B"], dtype=torch.float32).view(1, 4, 4)

    smpl_output = smpl_model.forward(betas, thetas[:, 3:], global_orient=thetas[:, :3])
    vertices = smpl_output.vertices
    vertices = torram.geometry.transform_points(T, vertices)

    scene = balanna.trimesh.show_axis(np.eye(4))
    scene = balanna.trimesh.show_point_cloud(vertices[0].detach().cpu().numpy(), scene=scene)
    yield {"scene": scene, "image": data["image"]}


if __name__ == "__main__":
    balanna.display_scenes(main())
