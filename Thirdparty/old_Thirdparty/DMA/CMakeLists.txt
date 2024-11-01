cmake_minimum_required(VERSION 3.5.0)
project(axidma VERSION 0.1.0 LANGUAGES C CXX)
##----------------add settings-------------------##
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -ggdb -fPIC -rdynamic") # 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -ggdb -fPIC -rdynamic") # 
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ")

include_directories(
${PROJECT_SOURCE_DIR}/include
)

file(GLOB SRC_LIST 
"src/*.c" 
)

# libSLAM3
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_library(${PROJECT_NAME} SHARED ${SRC_LIST})