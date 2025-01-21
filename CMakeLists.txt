cmake_minimum_required(VERSION 3.5.0)
project(cnn-slam3 VERSION 0.1.0 LANGUAGES C CXX)
##----------------add settings-------------------##
set(CMAKE_BUILD_TYPE "Debug") #Debug
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -ggdb -fPIC -rdynamic") # 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -ggdb -fPIC -rdynamic") # 
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
##--------------------link-library---------------------##
# opencv
#set(OpenCV_DIR  Thirdparty/opencv/lib/cmake/opencv4) 
#set(OpenCV_DIR  Thirdparty/opencv/share/OpenCV) 
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# Eigen

#add_definitions(-DEIGEN_DONT_ALIGN)
set(CMAKE_PREFIX_PATH Thirdparty/Eigen3)
find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIRS})

# pytorch
set(Torch_DIR Thirdparty/pytorch/share/cmake/Torch)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1) #新旧版本c++11导致libtorch的错误

# BOOST
set(BOOST_ROOT Thirdparty/boost)
find_package(Boost COMPONENTS serialization system filesystem REQUIRED)
if(Boost_FOUND)
  message(STATUS "Boost found: ${Boost_VERSION}")
  include_directories(${BOOST_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "Boost not found")
endif()

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/CameraModels
${PROJECT_SOURCE_DIR}/Thirdparty
${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
${PROJECT_SOURCE_DIR}/Thirdparty/Eigen3/include/eigen3
${PROJECT_SOURCE_DIR}/Thirdparty/boost/include
${PROJECT_SOURCE_DIR}/Thirdparty/openssl/include
${PROJECT_SOURCE_DIR}/Thirdparty/pytorch/include
${PROJECT_SOURCE_DIR}/Thirdparty/pytorch/include/torch/csrc/api/include
)

file(GLOB SRC_LIST 
"src/*.cc" 
"src/*.cpp" 
"src/CameraModels/*.cpp" 
)

# libSLAM3
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_library(${PROJECT_NAME} SHARED ${SRC_LIST})

# g2o
add_subdirectory(Thirdparty/g2o)

# DBoW3
add_subdirectory(Thirdparty/DBoW3)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Boost_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW3/lib/libDBoW3.so
${PROJECT_SOURCE_DIR}/Thirdparty/openssl/lib64/libcrypto.so
)

# link libtorch
target_link_libraries(${PROJECT_NAME}
${PROJECT_SOURCE_DIR}/Thirdparty/pytorch/lib/libc10.so
${PROJECT_SOURCE_DIR}/Thirdparty/pytorch/lib/libtorch_cpu.so
${PROJECT_SOURCE_DIR}/Thirdparty/pytorch/lib/libtorch_global_deps.so
${TORCH_LIBS}
)

#------Build Examples-----#
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/CNN_Module)
add_executable(mono_cam Examples/Cam_test/src/mono.cc)
target_link_libraries(mono_cam ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/CNN_Module)
add_executable(mono_euroc Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME} ${OpenCV_LIBS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/CNN_Module)
add_executable(rgbd Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd ${PROJECT_NAME} ${OpenCV_LIBS})