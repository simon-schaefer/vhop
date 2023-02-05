import argparse
import pathlib
import numpy as np
import pandas as pd
import smplx
import tqdm

from visualize import compute_smpl_vertices, load_data_and_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smpl-model-file", type=pathlib.Path, default="../../models/smpl/smpl_pkl/SMPL_NEUTRAL.pkl")
    return parser.parse_args()


def evaluate_sample(data_file: pathlib.Path, method: str, smpl_model: smplx.SMPL):
    data, data_hat = load_data_and_results(data_file, method=method)
    if data_hat is None:
        return {}
    joints_gt, vertices_gt = compute_smpl_vertices(smpl_model, data["betas"], data["thetas"], data["T_C_B"])
    joints_hat, vertices_hat = compute_smpl_vertices(smpl_model, data_hat[:10], data_hat[10:82], data["T_C_B"])
    execution_time = data_hat[-1]

    return {"mpjpe": np.linalg.norm(joints_gt - joints_hat, axis=-1).mean(), 
            "mve": np.linalg.norm(vertices_gt - vertices_hat, axis=-1).mean(),
            "camera": data_file.parent.name, 
            "sequence": data_file.parent.parent.name,
            "execution_time": execution_time,
            "method": method}


def main():
    args = parse_args()
    data_path = pathlib.Path("zju-mocap")
    results_path = pathlib.Path("results")
    smpl_model = smplx.SMPL(args.smpl_model_file)

    methods = [m.name for m in results_path.iterdir() if m.is_dir()]
    print("Available methods: ", [m for m in methods])

    files = sorted(list(data_path.glob("CoreView_*/**/*.npz")))
    results = []
    for method in methods:
        print("Evaluating method: ", method)
        for data_file in tqdm.tqdm(files):
            results_dict = evaluate_sample(data_file, method=method, smpl_model=smpl_model)
            if len(results_dict) == 0:
                continue
            results.append(results_dict)

    # Concatenate results to a pandas database. If there aren't any results, return.
    if len(results) == 0:
        print("No results found. Process the dataset first.")
        return
    results = pd.DataFrame(results)

    # Print the mean results, by camera and sequence.
    print(results.groupby(["camera", "sequence", "method"]).mean())


if __name__ == "__main__":
    main()
