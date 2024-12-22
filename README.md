# Mars-drone-SLAM-algorithm-zynqMPSOC-accelerator
## CNN-SLAM3,源码来自ORB-SLAM3源码修改
### 环境
zynqMP AX15UEG 开发板(armv8)
aarch64-linux-gnu-gcc: v9.2.0
Ptorch: v1.12.0
OpenCV: v3.4
Boost: v1.8.5
Openssl: v3.0.15

### 使用注意
编译cmakelist之前，需要将Thirdparty/pytorch.zip解压
如果不是相同架构环境，建议自己对应编译Thirdparty
若修改CMAKELSIT，需要注意保证CXX_STANDARD = 14 (因为pytorch新版本采用c++17标准，会导致与boost库发生冲突，建议使用c++14编译的老pytorch版本，除非能解决D_GLIBCXX_USE_CXX11_ABI的新旧问题)

### 环境移步
ORB-SLAM3(Main): https://github.com/UZ-SLAMLab/ORB_SLAM3 

Pytorch: https://github.com/pytorch/pytorch

OpenCV: https://github.com/opencv/opencv