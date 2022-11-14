# SDA on CUDA


Implementation of Statisctical Dominance Algorythm and Flying Histogram on CUDA runtime.

Program accepts volumetric TIFF images and outputs in same format and dimensions.


## Build instructions

Build and install TinyTIFF library:
https://github.com/jkriege2/TinyTIFF#documentation

Download and install CUDA Toolkit:
https://developer.nvidia.com/cuda-downloads

Sweet.hpp should be downloaded automatically as submodule from fork.

If used with different CUDA architecture than Ampere, appropriately named Cmake property should be changed.
