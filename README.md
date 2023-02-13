# VHOP - Virtual Humans from Optimization with Priors

Final project of 3D Scanning & Motion Capture course at TUM.

The goal of this project is to estimate 3D human pose from monocular images. To achieve that, we used extract 2D joint locations with OpenPose and optimize SMPL parameters with respect to re-projection and constant motion errors. We also provided pose priors
with Vposer to make generated poses more realistic. 

## Dataset

This project uses [ZJU-MoCap Dataset](https://chingswy.github.io/Dataset-Demo/). The access for the dataset must be requested from its website. Once you have the acess, run [preprocess_zju.py script](/data/zju-mocap/preprocess_zju.py) as follows:

```
python preprocess_zju.py --smpl-model-file <path-to-smpl-model-pkl> --dataset-dir <path-to-zju-mocap> --sample-name <sample-name-zju> --camera-name <camera-name-zju>
```

## Build & Run

This project in depend on Eigen, Ceres, Cnpy, Zlib, Gtest, OpenCV.
After installing them you can build project with the below commands: 

```
cmake --build .
make
```

Then run the code with the following arguments:

```
./process_dataset <dataset-directory> <method> <output-directory>
```

Ex:
```
./process_dataset ../data/zju-mocap/CoreView_377/Camera_B2/ smpl ../data/results/smpl/CoreView_377/Camera_B2
```

