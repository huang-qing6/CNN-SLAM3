cmake_minimum_required(VERSION 3.5.0)
project(zynq-slam3 VERSION 0.1.0 LANGUAGES C CXX)
##----------------add settings-------------------##
set(CMAKE_BUILD_TYPE "Realse") #Debug
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -ggdb -fPIC -rdynamic") # 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -ggdb -fPIC -rdynamic") # 
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

##--------------------link-library---------------------##
# opencv
set(OpenCV_DIR  Thirdparty/opencv/lib/cmake/opencv4) 
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# Eigen

#add_definitions(-DEIGEN_DONT_ALIGN)
set(CMAKE_PREFIX_PATH Thirdparty/Eigen3)
find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIRS})

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/CameraModels
#${PROJECT_SOURCE_DIR}/include/FPGA
${PROJECT_SOURCE_DIR}/Thirdparty
${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
${PROJECT_SOURCE_DIR}/Thirdparty/boost/include
${PROJECT_SOURCE_DIR}/Thirdparty/Eigen3/include/eigen3
#${PROJECT_SOURCE_DIR}/Thirdparty/DMA/include
${PROJECT_SOURCE_DIR}/Thirdparty/pytorch/include
${PROJECT_SOURCE_DIR}/Thirdparty/pytorch/include/torch/csrc/api/include
)

file(GLOB SRC_LIST 
"src/*.cc" 
"src/*.cpp" 
"src/CameraModels/*.cpp" 
#"Thirdparty/cma/*.h"
)

# libSLAM3
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_library(${PROJECT_NAME} SHARED ${SRC_LIST})

# g2o
add_subdirectory(Thirdparty/g2o)
add_subdirectory(Thirdparty/DBoW2)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/Thirdparty/boost/lib/libboost_serialization.so
${PROJECT_SOURCE_DIR}/Thirdparty/openssl/lib/libcrypto.so
# 加入libtorch,下面这些是一系列torch相关的共享库
${PROJECT_SOURCE_DIR}/Thirdparty/pytorch/lib/libc10.so
${PROJECT_SOURCE_DIR}/Thirdparty/pytorch/lib/libclog.a
${PROJECT_SOURCE_DIR}/Thirdparty/pytorch/lib/libcpuinfo.a
${PROJECT_SOURCE_DIR}/Thirdparty/pytorch/lib/libfmt.a
${PROJECT_SOURCE_DIR}/Thirdparty/pytorch/lib/libprotobuf-lite.a
${PROJECT_SOURCE_DIR}/Thirdparty/pytorch/lib/libprotobuf.a
${PROJECT_SOURCE_DIR}/Thirdparty/pytorch/lib/libprotoc.a
${PROJECT_SOURCE_DIR}/Thirdparty/pytorch/lib/libsleef.a
${PROJECT_SOURCE_DIR}/Thirdparty/pytorch/lib/libtensorpipe_uv.a
${PROJECT_SOURCE_DIR}/Thirdparty/pytorch/lib/libtensorpipe.a
${PROJECT_SOURCE_DIR}/Thirdparty/pytorch/lib/libtorch_cpu.so
${PROJECT_SOURCE_DIR}/Thirdparty/pytorch/lib/libtorch_global_deps.so
${PROJECT_SOURCE_DIR}/Thirdparty/pytorch/lib/libtorch.so
# 取消DMA,如有需要再用
#${PROJECT_SOURCE_DIR}/Thirdparty/cma/lib/libcma.so
#${PROJECT_SOURCE_DIR}/Thirdparty/DMA/lib/libaxidma.so
)

#------Build Examples-----#
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

add_executable(mono_euroc ${PROJECT_SOURCE_DIR}/Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME} ${OpenCV_LIBS})

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/cma_test)
#add_executable(cma_test ${PROJECT_SOURCE_DIR}/Examples/cma_test/cma_test.c)
#target_link_libraries(cma_test 
#-lpthread
#${PROJECT_SOURCE_DIR}/Thirdparty/cma/lib/libcma.so)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/cma_test)
#add_executable(cma_test ${PROJECT_SOURCE_DIR}/Examples/cma_test/cma_test.c)
#target_link_libraries(cma_test 
#-lpthread
#${PROJECT_SOURCE_DIR}/Thirdparty/cma/lib/libcma.so)
