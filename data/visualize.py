import argparse
import pathlib

import balanna
import cv2
import numpy as np
import smplx
import torch
import torram

from typing import Tuple


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=pathlib.Path, help="Path starting from zju-mocap dataset directory.")
    parser.add_argument("--smpl-model-file", type=pathlib.Path, default="../../models/smpl/smpl_pkl/SMPL_NEUTRAL.pkl")
    return parser.parse_args()


def compute_smpl_vertices(smpl_model: smplx.SMPL, betas: np.ndarray, thetas: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    betas_t = torch.tensor(betas, dtype=torch.float32).view(1, 10)
    thetas_t = torch.tensor(thetas, dtype=torch.float32).view(1, 72)
    smpl_output = smpl_model.forward(betas_t, thetas_t[:, 3:], global_orient=thetas_t[:, :3])

    T_t = torch.tensor(T, dtype=torch.float32).view(1, 4, 4)
    vertices = torram.geometry.transform_points(T_t, smpl_output.vertices)
    vertices = vertices[0].detach().numpy()
    joints = torram.geometry.transform_points(T_t, smpl_output.joints)
    joints = joints[0].detach().numpy()
    return joints, vertices


def load_data_and_results(data_file: pathlib.Path):
    data = np.load(data_file.as_posix(), allow_pickle=True)
    results_file = pathlib.Path("results") / data_file.relative_to("zju-mocap")
    results_file = results_file.with_suffix(".bin")
    if results_file.exists():
        data_hat = np.fromfile(results_file)
    else:
        data_hat = None
    return data, data_hat


def visualize_sample(data_file: pathlib.Path, smpl_model: smplx.SMPL):
    data, data_hat = load_data_and_results(data_file)

    scene = balanna.trimesh.show_axis(np.eye(4))  # camera frame
    _, vertices_gt = compute_smpl_vertices(smpl_model, data["betas"], data["thetas"], data["T_C_B"])
    scene = balanna.trimesh.show_point_cloud(vertices_gt, colors=(0, 255, 0), scene=scene)
    
    image = cv2.imread(data_file.with_suffix(".jpg").as_posix())
    width = int(image.shape[1] / 4)
    height = int(image.shape[0] / 4)
    image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    image = image[:, :, [2, 1, 0]]  # BGR to RGB
    image = np.swapaxes(np.swapaxes(image, 0, 2), 1, 2)  # HWC to CHW

    if data_hat is not None:
        _, vertices_hat = compute_smpl_vertices(smpl_model, data_hat[:10], data_hat[10:], data["T_C_B"])
        scene = balanna.trimesh.show_point_cloud(vertices_hat, colors=(255, 0, 0), scene=scene)

    return {"scene": scene, "image": image}


def main():
    args = parse_args()
    smpl_model = smplx.SMPL(args.smpl_model_file)
    
    data_path = pathlib.Path("zju-mocap") / args.data
    if data_path.is_dir():
        files = sorted(list(data_path.glob("**/*.npz")))
        for data_file in files:
            yield visualize_sample(data_file, smpl_model)
    else:
        for _ in range(10):
            yield visualize_sample(data_path, smpl_model)


if __name__ == "__main__":
    balanna.display_scenes(main())
