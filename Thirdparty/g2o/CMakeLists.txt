cmake_minimum_required(VERSION 3.5.0)
project(g2o)

# find_eigen3 necessary
set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/../Eigen3 )
find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIRS})
#build g2o.so
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -W -O3 ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -W -O3")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
configure_file(config.h.in ${g2o_SOURCE_DIR}/config.h)
include_directories(g2o/core g2o/solves g2o/stuff g2o/types)
file(GLOB SRC_LIST "g2o/types/*.cpp" "g2o/core/*.cpp" "g2o/stuff/*.cpp" "g2o/stuff/*.c")
add_library(g2o SHARED ${SRC_LIST})
