# VHOP - Virtual Humans from Optimization with Priors

Final project of 3D Scanning & Motion Capture course at TUM.

The goal of this project is to estimate 3D human pose from monocular images. To achieve that, we used extract 2D joint locations with OpenPose and 
optimize SMPL parameters with respect to re-projection and constant motion errors. We also provided pose priors with 
[Vposer](https://smpl-x.is.tue.mpg.de/) to make generated poses more realistic. 

## Data Preparation
This project uses [ZJU-MoCap Dataset](https://chingswy.github.io/Dataset-Demo/). The access for the dataset must be requested
from its website. Once you have the access, run [preprocess_zju.py script](/data/zju-mocap/preprocess_zju.py) as follows:

```
python preprocess_zju.py --smpl-model-file <path-to-smpl-model-pkl> --dataset-dir <path-to-zju-mocap> --sample-name <sample-name-zju> --camera-name <camera-name-zju>
```

The SMPL and VPoser model weights are already provided in the [data](/data) folder, as they have been preprocessed.


## Build & Run

This project in depend on Eigen, Ceres, Cnpy, Zlib, Gtest, OpenCV.
After installing them you can build project with the below commands: 

```
cmake --build -DCMAKE_BUILD_TYPE=Release .
make
```

Then run the code with the following arguments:
```
./process_dataset <dataset-directory> <method> <output-directory>

# Example
./process_dataset ../data/zju-mocap/CoreView_377/Camera_B2/ smpl ../data/results/smpl/CoreView_377/Camera_B2
```

## Evaluation
After running the code, you can post-process, evaluate and visualize the results with the following command. It assumes that the results
are stored in `data/results` directory.
```
cd data
python post_processing.py

# Evaluation
cd data/evaluation
python evaluate.py --smpl-model <smpl-model-path>

# Visualization
python visualize.py <seq-file> --smpl-model <smpl-model-path>
```