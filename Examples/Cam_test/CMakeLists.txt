cmake_minimum_required(VERSION 3.5.0)
project(usb_cam)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../cmake_modules)

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4 not found.")
endif()

find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
${OpenCV_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/../../
${PROJECT_SOURCE_DIR}/../../include
${PROJECT_SOURCE_DIR}/../../include/CameraModels
${PROJECT_SOURCE_DIR}/../../Thirdparty/Sophus
${Pangolin_INCLUDE_DIRS}
)

set(LIBS 
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/../../Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/../../lib/libORB_SLAM3.so
-lboost_system
)

# Node for monocular camera
add_executable(Mono
src/mono.cc
)

target_link_libraries(Mono
${LIBS}
boost_system
boost_filesystem
)
