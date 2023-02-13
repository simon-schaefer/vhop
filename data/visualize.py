import argparse
import balanna
import cv2
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import pathlib
import smplx
import trimesh

from evaluation.evaluate import compute_smpl_vertices


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=pathlib.Path, help="Path starting from zju-mocap dataset directory.")
    parser.add_argument("--smpl-model-file", type=pathlib.Path, default="../../models/smpl/smpl_pkl/SMPL_NEUTRAL.pkl")
    return parser.parse_args()


def main():
    args = parse_args()
    data = np.load(args.data.as_posix(), allow_pickle=True)
    smpl_model = smplx.SMPL(args.smpl_model_file)

    num_frames = data["betas"].shape[0]
    normalize = mcolors.Normalize(vmin=0, vmax=6890 // 2)
    s_map = cm.ScalarMappable(norm=normalize, cmap=cm.viridis)
    v_idx = np.concatenate([np.arange(3445), np.arange(3445)])
    vertex_colors = s_map.to_rgba(v_idx)[..., :3]

    for i in range(num_frames):
        # scene = balanna.trimesh.show_axis(np.eye(4))  # camera frame
        scene = trimesh.Scene()

        image = cv2.imread(data["image_files"][i])
        width = int(image.shape[1] / 2)
        height = int(image.shape[0] / 2)
        image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        image = image[:, :, [2, 1, 0]]  # BGR to RGB
        image = np.swapaxes(np.swapaxes(image, 0, 2), 1, 2)  # HWC to CHW

        betas = data["betas"][i]
        thetas = data["thetas"][i]
        T = data["T"][i]
        if not (np.isnan(thetas).any() or np.isnan(betas).any() or np.isnan(T).any()):
            _, vertices_hat = compute_smpl_vertices(smpl_model, betas, thetas, T=T)
            vertices_hat = vertices_hat[0]
            scene = balanna.trimesh.show_point_cloud(vertices_hat, colors=vertex_colors, scene=scene)

        yield {"scene": scene, "image": image}


if __name__ == "__main__":
    balanna.display_scenes(main())
