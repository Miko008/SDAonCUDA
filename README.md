# SDA on CUDA


Implementation of Statisctical Dominance Algorythm and Flying Histogram on CUDA runtime.

Program accepts volumetric TIFF images and outputs in same format and dimensions.

## References
### SDA 
Adam Piórkowski 
A statistical dominance algorithm for edge detection and segmentation of medical images. 
In Information Technologies in Medicine, volume 471 of Advances in Intelligent Systems and Computing, pages 3–14. Springer, 2016

https://home.agh.edu.pl/~pioro/sda/

### FH
Piotr Wiśniewski, Krzysztof Stencel, Michał Chlebiej, Emilia Wiśniewska
Flying Histogram Optimization of Statistical Dominance Algorithm
Proceedings of the 26th International Workshop on Concurrency, Specification and Programming, Warsaw, Poland, pages
25-27. 2017



## Build instructions

Build and install TinyTIFF library:
https://github.com/jkriege2/TinyTIFF#documentation

Download and install CUDA Toolkit:
https://developer.nvidia.com/cuda-downloads

Sweet.hpp should be downloaded automatically as submodule from fork.

If used with different CUDA architecture than Ampere, appropriately named Cmake property should be changed.
