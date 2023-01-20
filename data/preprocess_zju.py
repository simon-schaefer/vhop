import argparse
import json
import numpy as np
import pathlib
import shutil

import cv2
import smplx
import torch
import torram


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def process_sample(data_directory: pathlib.Path, camera_name: str, output_file: pathlib.Path, smpl_model: smplx.SMPL, debug: bool = False):
    smpl_data = np.load((data_directory / "new_params" / "0.npy").as_posix(), allow_pickle=True).item()
    with open(data_directory / "keypoints2d" / camera_name / "000000_keypoints.json", 'r') as f:
        keypoints_2d_data = json.load(f)
    keypoints_2d = np.array(keypoints_2d_data['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
    keypoints_2d = keypoints_2d[:, :2].astype(int)
    keypoints_scores = keypoints_2d[:, 2]

    intrinsic_data = cv2.FileStorage((data_directory / "intri.yml").as_posix(), cv2.FILE_STORAGE_READ)
    K = intrinsic_data.getNode(f"K_{camera_name}").mat()
    D = intrinsic_data.getNode(f"dist_{camera_name}").mat()
    extrinsic_data = cv2.FileStorage((data_directory / "extri.yml").as_posix(), cv2.FILE_STORAGE_READ)
    T_C_W = np.eye(4)
    T_C_W[:3, :3] = extrinsic_data.getNode(f"Rot_{camera_name}").mat()
    T_C_W[:3, 3] = extrinsic_data.getNode(f"T_{camera_name}").mat().flatten()
    T_C_W = torch.tensor(T_C_W, dtype=torch.float32)

    betas = smpl_data['shapes']
    thetas = smpl_data['poses']
    orientation = smpl_data['Rh']
    translation = smpl_data['Th']
    T_W_B = torch.eye(4)
    T_W_B[:3, :3] = torram.geometry.angle_axis_to_rotation_matrix(torch.tensor(orientation))
    T_W_B[:3, 3] = torch.tensor(translation)

    # Compute 3D joints for debugging and testing.
    smpl_output = smpl_model.forward(
        betas=torch.tensor(betas, dtype=torch.float32),
        body_pose=torch.tensor(thetas[:, 3:], dtype=torch.float32),
        global_orient=torch.tensor(thetas[:, :3], dtype=torch.float32),
    )
    joints_3d_wo_translation = smpl_output.joints.detach().cpu().numpy()
    vertices_w = smpl_output.vertices.detach()
    T_C_B = (T_C_W @ T_W_B).view(1, 4, 4)
    vertices_c = torram.geometry.transform_points(T_C_B, vertices_w)
    vertices_2d = torram.geometry.project_points(vertices_c, torch.tensor(K)).detach().numpy().reshape(-1, 2).astype(int)

    # Check validity of data.
    assert len(keypoints_2d_data['people']) == 1
    image_file_name = (data_directory / camera_name / "000000.jpg").as_posix()
    image = cv2.imread(image_file_name)
    if debug:
        for x, y in keypoints_2d:
            assert 0 <= x < image.shape[1]
            assert 0 <= y < image.shape[0]
            image = cv2.circle(image, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
        for x, y in vertices_2d:
            image = cv2.circle(image, (x, y), radius=4, color=(255, 0, 0), thickness=-1)
        cv2.imshow("image", image)
        cv2.waitKey()
    image = cv2.resize(image, (int(image.shape[1] * 0.4), int(image.shape[0] * 0.4)), interpolation=cv2.INTER_AREA)
    image = np.moveaxis(image, -1, 0)  # (H, W, 3) -> (3, H, W)
    image = image[[2, 1, 0], :, :]  # BGR -> RGB

    # Save data in output file as npz file.
    np.savez(
        output_file,
        betas=betas[0].astype(np.float64),
        thetas=thetas[0].astype(np.float64),
        intrinsics=K.astype(np.float64),
        distortion=D.astype(np.float64),
        joints_3d_wo_translation=joints_3d_wo_translation[0].astype(np.float64),
        T_C_B=T_C_B[0].numpy().astype(np.float64),
        keypoints_2d=keypoints_2d.astype(np.float64),
        keypoints_2d_scores=keypoints_scores.astype(np.float64),
        image=image
    )
    shutil.copy(image_file_name, output_file.with_suffix(".jpg").as_posix())


if __name__ == '__main__':
    args = parse_args()
    smpl_model_file = pathlib.Path("/home/wiss/sas/models/smpl/smpl_pkl/SMPL_NEUTRAL.pkl")
    smpl_model = smplx.SMPL(smpl_model_file)

    data_dir = pathlib.Path("/storage/user/sas/sas/zju-mocap/CoreView_387")
    output_file = pathlib.Path("/usr/wiss/sas/vhop/data/test/sample.npz")
    process_sample(data_dir, "Camera_B1", output_file, smpl_model=smpl_model, debug=args.debug)
