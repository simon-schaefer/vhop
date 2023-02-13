import argparse
import pathlib
import numpy as np
import pandas as pd
import smplx
import torch
import torram

from typing import Tuple


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smpl-model-file", type=pathlib.Path, default="../../../models/smpl/smpl_pkl/SMPL_NEUTRAL.pkl")
    return parser.parse_args()


def compute_smpl_vertices(smpl_model: smplx.SMPL, betas: np.ndarray, thetas: np.ndarray, T: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
    betas_t = torch.tensor(betas, dtype=torch.float32).view(-1, 10)
    thetas_t = torch.tensor(thetas, dtype=torch.float32).view(-1, 72)
    smpl_output = smpl_model.forward(betas_t, thetas_t[:, 3:], global_orient=thetas_t[:, :3])

    T_t = torch.tensor(T, dtype=torch.float32).view(-1, 4, 4)
    vertices = torram.geometry.transform_points(T_t, smpl_output.vertices)
    vertices = vertices.detach().numpy()
    joints = torram.geometry.transform_points(T_t, smpl_output.joints)
    joints = joints.detach().numpy()
    return joints, vertices


def evaluate_sequence(data_file: pathlib.Path, smpl_model: smplx.SMPL):
    data = np.load(data_file.as_posix())
    method, smoothness, camera, sequence, _ = data_file.name.split(".")

    num_frames = data["betas"].shape[0]
    assert data["thetas"].shape == data["thetas_gt"].shape == (num_frames, 72)
    assert data["betas"].shape == data["betas_gt"].shape == (num_frames, 10)
    assert data["T"].shape == data["T_gt"].shape == (num_frames, 4, 4)

    joints_gt, vertices_gt = compute_smpl_vertices(smpl_model, data["betas_gt"], data["thetas_gt"], data["T_gt"])
    joints_hat, vertices_hat = compute_smpl_vertices(smpl_model, data["betas"], data["thetas"], data["T"])
    execution_time = data["execution_time"]

    eval_dicts = []
    import pdb; pdb.set_trace()
    for i in range(num_frames):
        eval_dict = {
            "camera": camera,
            "sequence": sequence,
            "method": method,
            "smoothness": smoothness,
            "execution_time": execution_time[i],
            "mpjpe": np.linalg.norm(joints_gt[i] - joints_hat[i], axis=-1).mean(),
            "mve": np.linalg.norm(vertices_gt[i] - vertices_hat[i], axis=-1).mean(),
        }
        eval_dicts.append(eval_dict)
    return eval_dicts


def main():
    args = parse_args()
    smpl_model = smplx.SMPL(args.smpl_model_file)

    results = []
    for f in pathlib.Path("post_processed").glob("*.npz"):
        results += evaluate_sequence(f, smpl_model=smpl_model)
    results = pd.DataFrame(results)

    # Print the mean results, by camera and sequence.
    output_df = results.groupby(["camera", "sequence", "method", "smoothness"])
    print(output_df.mean())
    results.to_csv("results.csv")


if __name__ == "__main__":
    main()
