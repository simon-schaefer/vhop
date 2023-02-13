import numpy as np
import pathlib

from pykalman import KalmanFilter


CMU_TO_SMPL = np.array([
    8,  # pelvis -> hips 0
    12,  # left_hip -> leftUpLeg 1
    9,  # right_hip -> rightUpLeg 2
    8,  # thorax -> spine1 3
    13,  # lknee -> leftLeg 4
    10,  # rknee -> rightLeg 5
    8,  # thorax -> spine2 6
    14,  # lankle -> left_ankle 7
    11,  # rankle -> right_ankle 8
    1,  # neck -> spine3 9
    21,  # left_big_toe -> left_foot 10
    12,  # right_big_toe -> right_foot 11
    1,  # neck -> neck 12
    5,  # left_shoulder -> left_collar 13
    2,  # right_shoulder -> right_collar 14
    0,  # nose -> head 15
    5,  # left_shoulder -> left_shoulder 16
    2,  # right_shoulder -> right_shoulder 17
    6,  # left_elbow -> left_elbow 18
    3,  # right_elbow -> right_elbow 19
    7,  # left_wrist -> left_wrist 20
    4,  # right_wrist -> right_wrist 21
    7,  # left_wrist -> leftHandIndex1 22
    4,  # right_wrist -> rightHandIndex1 23
])


def load_sequence(directory: pathlib.Path):
    files = sorted(list(directory.glob("*.bin")))
    num_files = len(files)

    betas_gt = np.zeros((num_files, 10))
    thetas_gt = np.zeros((num_files, 72))
    T_gt = np.zeros((num_files, 4, 4))

    betas = np.zeros((num_files, 10))
    thetas = np.zeros((num_files, 72))
    T = np.zeros((num_files, 4, 4))
    certainties = np.zeros((num_files, 72))
    execution_time = np.zeros(num_files)

    image_files = []
    for i, data_file in enumerate(files):
        method = data_file.relative_to("results").parts[0]
        method_dir = pathlib.Path("results") / method
        data_file = (pathlib.Path("zju-mocap") / data_file.relative_to(method_dir)).with_suffix(".npz")
        data = np.load(data_file.as_posix(), allow_pickle=True)

        if data_file.exists():
            data_hat = np.fromfile(data_file)
        else:
            data_hat = None

        betas_gt[i] = data["betas"]
        thetas_gt[i] = data["thetas"]
        T_gt[i] = data["T_C_B"]

        if data_hat is not None:
            data_hat[10+20*3:82] = 0  # hands not optimized
            betas[i] = data_hat[:10]
            thetas[i] = data_hat[10:82]
            execution_time[i] = data_hat[-1]

            # Remap OpenPose uncertainties to SMPL parameters
            certainties_op = data["keypoints_2d_scores"][CMU_TO_SMPL]
            for j in range(24):
                certainties[i, 3*j:3*(j+1)] = max(certainties_op[j], 1e-6)
        else:
            betas[i] = data["betas"]
            thetas[i] = np.nan
            certainties[i] = 1e-6
            execution_time[i] = np.nan
        T[i] = data["T_C_B"]

        # Get image files from data files.
        image_files.append(data_file.with_suffix(".jpg"))

    # Assume that the number of ground truth frames is the same as the number of frames with predictions.
    assert betas_gt.shape == betas.shape
    assert thetas_gt.shape == thetas.shape
    assert T_gt.shape == T.shape
    return image_files, (betas_gt, thetas_gt, T_gt), (betas, thetas, T, certainties, execution_time)


def kalman_filtering(thetas, certainties):
    # Interpolate missing values.
    if np.any(np.isnan(thetas)):
        print("Interpolating missing values.")
        ok = ~np.isnan(thetas)
        xp = ok.ravel().nonzero()[0]
        fp = thetas[~np.isnan(thetas)]
        x = np.isnan(thetas).ravel().nonzero()[0]
        thetas[np.isnan(thetas)] = np.interp(x, xp, fp)

    # Kalman filtering.
    kf = KalmanFilter(transition_matrices=np.eye(72),
                      observation_matrices=np.eye(72),
                      observation_covariance=[np.diag(1.0 / certainties[i]) for i in range(len(certainties))],
                      initial_state_mean=thetas[0],
                      initial_state_covariance=np.eye(72),
                      em_vars=['transition_covariance', 'initial_state_covariance'])
    thetas_smoothed, _ = kf.smooth(thetas)
    return thetas_smoothed


def main():
    sequences = set([d.parent for d in pathlib.Path("results").glob("**/*.bin") if d.parent.name.startswith("Camera")])
    for sequence_dir in sequences:
        print("Processing sequence in", sequence_dir.as_posix())
        method = sequence_dir.relative_to("results").parts[0]
        method_dir = pathlib.Path("results") / method
        sequence_name = sequence_dir.relative_to(method_dir).as_posix().replace("/", ".")

        image_files, (betas_gt, thetas_gt, T_gt), (betas, thetas, T, certainties, execution_time) = load_sequence(sequence_dir)
        thetas_smooth = kalman_filtering(thetas, certainties)

        output_file = pathlib.Path("evaluation") / "post_processed" / (method + ".raw." + sequence_name + ".npz")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        store_dict = dict(
            image_files=image_files,
            betas_gt=betas_gt,
            thetas_gt=thetas_gt,
            T_gt=T_gt,
            betas=betas,
            T=T,
            certainties=certainties,
            execution_time=execution_time,
        )
        np.savez(output_file, thetas=thetas, **store_dict)
        output_file = pathlib.Path("evaluation") / "post_processed" / (method + ".smoothed." + sequence_name + ".npz")
        np.savez(output_file.as_posix(), thetas=thetas_smooth, **store_dict)


if __name__ == "__main__":
    main()
