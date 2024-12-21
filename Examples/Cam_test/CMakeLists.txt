cmake_minimum_required(VERSION 3.5.0)
project(usb_cam)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3")

# opencv
set(OpenCV_DIR  ../../Thirdparty/opencv/lib/cmake/opencv4) 
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_PREFIX_PATH ../../Thirdparty/Eigen3)
find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIRS})

# pytorch
set(Torch_DIR ../../Thirdparty/pytorch/share/cmake/Torch)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

# openssl
set(OPENSSL_ROOT_DIR ../../Thirdparty/openssl)
find_package(OpenSSL REQUIRED)
include_directories(${OPENSSL_INCLUDE_DIR})

include_directories(
${PROJECT_SOURCE_DIR}/../../
${PROJECT_SOURCE_DIR}/../../include
${PROJECT_SOURCE_DIR}/../../include/CameraModels
${PROJECT_SOURCE_DIR}/../../Thirdparty
${PROJECT_SOURCE_DIR}/../../Thirdparty/Sophus
${PROJECT_SOURCE_DIR}/../../Thirdparty/boost/include
)

set(LIBS 
${PROJECT_SOURCE_DIR}/../../lib/libcnn-slam3.so
${OpenCV_LIBS}
#${PROJECT_SOURCE_DIR}/../../Thirdparty/opencv/lib/libopencv_highgui.so
#${EIGEN3_LIBS}
#${PROJECT_SOURCE_DIR}/../../Thirdparty/pytorch/lib/libtorch_cpu.so
#${Torch_LIBS}
#${PROJECT_SOURCE_DIR}/../../Thirdparty/DBoW2/lib/libDBoW2.so
#${PROJECT_SOURCE_DIR}/../../Thirdparty/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/../../Thirdparty/boost/lib/libboost_serialization.so
#${PROJECT_SOURCE_DIR}/../../Thirdparty/openssl/lib64/libcrypto.so
)

# Node for monocular camera
add_executable(Mono
${PROJECT_SOURCE_DIR}/src/mono.cc
)

target_link_libraries(Mono 
${LIBS}
)
