# Flexipod simulation
The simulation engine for the flexipod project, build upon the original [Titan library](https://github.com/jacobaustin123/Titan)[[1]](#ref-1) from Jacob Austin.


## Hardware Requirement:
**Windows/Linux** machine with Nvidia Graphics card that has **Cuda** support, preferably with compute Capability>6.0. See a list of CUDA enabled graphics card [here](https://developer.nvidia.com/cuda-gpus).


## Setup

### 0. (windows) Install visual studio 2019:[link](https://visualstudio.microsoft.com/downloads/)
and in the installation manager, install with **"Desktop development with c++"**

### 1. Install CUDA (preferably 10.0+):[link](https://developer.nvidia.com/cuda-downloads)
### 2. Install vcpkg by floowing the [quick start guid](https://github.com/Microsoft/vcpkg#quick-start).

For example in window, install vcpkg in ```C:\vcpkg``` using git bash:
```bash
cd C:/
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.bat
./vcpkg integrate install
```
To use vcpkg in CMAKE, you can either:

+ add it onto your CMake command line as -DCMAKE_TOOLCHAIN_FILE=[vcpkg root]\scripts\buildsystems\vcpkg.cmake.
+ or setup an invironment variable as: ```VCPKG_ROOT = C:\vcpkg```. This method is included in the [CMakeLists.txt](./CMakeLists.txt) file.

next install some packages with vcpkg
For windows:
```bash
vcpkg install --triplet x64-windows glew glad glm glfw3 msgpack
```

### 3. Clone the repo
either to clone (ssh):
```bash
git clone git@github.com:boxiXia/FlexipodFast.git
```
or to clone (HTTPS):
```bash
git clone https://github.com/boxiXia/FlexipodFast.git
```
### 4. Modify the [CMakeLists.txt](./CMakeLists.txt) file
find these lines, and modify occording to your GPU architecture
```cmake
###### here you must modify to fit your GPU architecture #####
# check https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_75,code=sm_75")
```

## Troubleshoot
### (Windows) Simulation window is showing, but I see no robot
In Nvidia control panel, go to "Manage 3D Settings" -> "program setting". Add the "flexipod.exe" and make sure the setting: "OpenGL rendering GPU" is set to Nvidia GPU


## Reference:
[1](#ref-1) J. Austin, R. Corrales-Fatou, S. Wyetzner, and H. Lipson, “Titan: A Parallel Asynchronous Library for Multi-Agent and Soft-Body Robotics using NVIDIA CUDA,” ICRA 2020, May 2020.

