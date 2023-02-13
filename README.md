# VHOP - Virtual Humans from Optimization with Priors

Final project of 3D Scanning & Motion Capture course at TUM.

The goal of this project is to estimate 3D human pose from monocular images. To achieve that, we used extract 2D joint locations with OpenPose and optimize SMPL parameters with respect to re-projection and constant motion errors. We also provided pose priors
with Vposer to make generated poses more realistic. 

## Build

This project in depend on Eigen, Ceres, Cnpy, Zlib, Gtest, OpenCV.
After installing them you can build project with the below commands: 

```
cmake --build .
make
./process_dataset ../data/zju-mocap/CoreView_377/Camera_B2/ smpl ../data/results/smpl/CoreView_377/Camera_B2
```