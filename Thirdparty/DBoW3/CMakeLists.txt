cmake_minimum_required(VERSION 3.5)
project(DBoW3)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O3")

set(HDRS_DBOW3
  src/BowVector.h
  src/Database.h
  src/DBoW3.h
  src/DescManip.h
  src/exports.h
  src/QueryResults.h
  src/quicklz.h
  src/timers.h
  src/FeatureVector.h
  src/ScoringObject.h   
  src/Vocabulary.h)

set(SRCS_DBOW3
  src/BowVector.cpp
  src/Database.cpp
  src/DescManip.cpp
  src/QueryResults.cpp
  src/quicklz.c
  src/FeatureVector.cpp
  src/ScoringObject.cpp   
  src/Vocabulary.cpp)

set(HDRS_DUTILS
  DUtils/Random.h
  DUtils/Timestamp.h)

set(SRCS_DUTILS
  DUtils/Random.cpp
  DUtils/Timestamp.cpp)

# 要保证整个工程的opencv版本一致，包括dbow，源码以及ros相关的
# 3 4 都可以正常运行
#set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../opencv/lib/cmake/opencv4)
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../opencv/share/OpenCV)
find_package(OpenCV)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../Thirdparty/boost/include)

include_directories(${OpenCV_INCLUDE_DIRS})
add_library(DBoW3 SHARED ${SRCS_DBOW3} ${SRCS_DUTILS})
target_link_libraries(DBoW3 ${OpenCV_LIBS})
