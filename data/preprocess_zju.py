import cv2
import json
import numpy as np
import pathlib


def process_sample(data_directory: pathlib.Path, camera_name: str, output_file: pathlib.Path, debug: bool = False):
    smpl_data = np.load((data_directory / "new_params" / "0.npy").as_posix(), allow_pickle=True).item()
    with open(data_directory / "keypoints2d" / camera_name / "000000_keypoints.json", 'r') as f:
        keypoints_2d_data = json.load(f)
    keypoints_2d = np.array(keypoints_2d_data['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
    keypoints_2d = keypoints_2d[:, :2].astype(int)

    intrinsic_data = cv2.FileStorage((data_directory / "intri.yml").as_posix(), cv2.FILE_STORAGE_READ)
    K = intrinsic_data.getNode("K_Camera_B1").mat()
    D = intrinsic_data.getNode("dist_Camera_B1").mat()

    # Check validity of data.
    assert len(keypoints_2d_data['people']) == 1
    if debug:
        image = cv2.imread((data_directory / camera_name / "000000.jpg").as_posix())
        for x, y in keypoints_2d:
            assert 0 <= x < image.shape[1]
            assert 0 <= y < image.shape[0]
            image = cv2.circle(image, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
        cv2.imshow("image", image)
        cv2.waitKey()

    # Save data in output file as npz file.
    np.savez(
        output_file,
        betas=smpl_data['shapes'][0].astype(np.float64),
        thetas=smpl_data['poses'][0].astype(np.float64),
        orientation=smpl_data['Rh'][0].astype(np.float64),
        translation=smpl_data['Th'][0].astype(np.float64),
        intrinsics=K.astype(np.float64),
        distortion=D.astype(np.float64),
        keypoints_2d=keypoints_2d.astype(np.float64),
    )


if __name__ == '__main__':
    data_dir = pathlib.Path("/storage/user/sas/sas/zju-mocap/CoreView_387")
    output_file = pathlib.Path("/usr/wiss/sas/vhop/data/test/sample.npz")

    process_sample(data_dir, "Camera_B1", output_file, debug=True)
