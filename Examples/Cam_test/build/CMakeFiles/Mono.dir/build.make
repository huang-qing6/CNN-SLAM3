# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mario/gitee/CNN-SLAM3/Examples/Cam_test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mario/gitee/CNN-SLAM3/Examples/Cam_test/build

# Include any dependencies generated for this target.
include CMakeFiles/Mono.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Mono.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Mono.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Mono.dir/flags.make

CMakeFiles/Mono.dir/src/mono.cc.o: CMakeFiles/Mono.dir/flags.make
CMakeFiles/Mono.dir/src/mono.cc.o: /home/mario/gitee/CNN-SLAM3/Examples/Cam_test/src/mono.cc
CMakeFiles/Mono.dir/src/mono.cc.o: CMakeFiles/Mono.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/mario/gitee/CNN-SLAM3/Examples/Cam_test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Mono.dir/src/mono.cc.o"
	/tools/Xilinx/Vitis/2020.1/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Mono.dir/src/mono.cc.o -MF CMakeFiles/Mono.dir/src/mono.cc.o.d -o CMakeFiles/Mono.dir/src/mono.cc.o -c /home/mario/gitee/CNN-SLAM3/Examples/Cam_test/src/mono.cc

CMakeFiles/Mono.dir/src/mono.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Mono.dir/src/mono.cc.i"
	/tools/Xilinx/Vitis/2020.1/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mario/gitee/CNN-SLAM3/Examples/Cam_test/src/mono.cc > CMakeFiles/Mono.dir/src/mono.cc.i

CMakeFiles/Mono.dir/src/mono.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Mono.dir/src/mono.cc.s"
	/tools/Xilinx/Vitis/2020.1/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mario/gitee/CNN-SLAM3/Examples/Cam_test/src/mono.cc -o CMakeFiles/Mono.dir/src/mono.cc.s

# Object files for target Mono
Mono_OBJECTS = \
"CMakeFiles/Mono.dir/src/mono.cc.o"

# External object files for target Mono
Mono_EXTERNAL_OBJECTS =

Mono: CMakeFiles/Mono.dir/src/mono.cc.o
Mono: CMakeFiles/Mono.dir/build.make
Mono: /home/mario/gitee/CNN-SLAM3/Examples/Cam_test/../../lib/libcnn-slam3.so
Mono: /home/mario/gitee/CNN-SLAM3/Thirdparty/opencv/lib/libopencv_gapi.so.4.9.0
Mono: /home/mario/gitee/CNN-SLAM3/Thirdparty/opencv/lib/libopencv_highgui.so.4.9.0
Mono: /home/mario/gitee/CNN-SLAM3/Thirdparty/opencv/lib/libopencv_ml.so.4.9.0
Mono: /home/mario/gitee/CNN-SLAM3/Thirdparty/opencv/lib/libopencv_objdetect.so.4.9.0
Mono: /home/mario/gitee/CNN-SLAM3/Thirdparty/opencv/lib/libopencv_photo.so.4.9.0
Mono: /home/mario/gitee/CNN-SLAM3/Thirdparty/opencv/lib/libopencv_stitching.so.4.9.0
Mono: /home/mario/gitee/CNN-SLAM3/Thirdparty/opencv/lib/libopencv_video.so.4.9.0
Mono: /home/mario/gitee/CNN-SLAM3/Thirdparty/opencv/lib/libopencv_videoio.so.4.9.0
Mono: /home/mario/gitee/CNN-SLAM3/Examples/Cam_test/../../Thirdparty/boost/lib/libboost_serialization.so
Mono: /home/mario/gitee/CNN-SLAM3/Thirdparty/opencv/lib/libopencv_imgcodecs.so.4.9.0
Mono: /home/mario/gitee/CNN-SLAM3/Thirdparty/opencv/lib/libopencv_dnn.so.4.9.0
Mono: /home/mario/gitee/CNN-SLAM3/Thirdparty/opencv/lib/libopencv_calib3d.so.4.9.0
Mono: /home/mario/gitee/CNN-SLAM3/Thirdparty/opencv/lib/libopencv_features2d.so.4.9.0
Mono: /home/mario/gitee/CNN-SLAM3/Thirdparty/opencv/lib/libopencv_flann.so.4.9.0
Mono: /home/mario/gitee/CNN-SLAM3/Thirdparty/opencv/lib/libopencv_imgproc.so.4.9.0
Mono: /home/mario/gitee/CNN-SLAM3/Thirdparty/opencv/lib/libopencv_core.so.4.9.0
Mono: CMakeFiles/Mono.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/mario/gitee/CNN-SLAM3/Examples/Cam_test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Mono"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Mono.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Mono.dir/build: Mono
.PHONY : CMakeFiles/Mono.dir/build

CMakeFiles/Mono.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Mono.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Mono.dir/clean

CMakeFiles/Mono.dir/depend:
	cd /home/mario/gitee/CNN-SLAM3/Examples/Cam_test/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mario/gitee/CNN-SLAM3/Examples/Cam_test /home/mario/gitee/CNN-SLAM3/Examples/Cam_test /home/mario/gitee/CNN-SLAM3/Examples/Cam_test/build /home/mario/gitee/CNN-SLAM3/Examples/Cam_test/build /home/mario/gitee/CNN-SLAM3/Examples/Cam_test/build/CMakeFiles/Mono.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/Mono.dir/depend

