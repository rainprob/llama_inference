# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/autodl-tmp/llama_inference

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/autodl-tmp/llama_inference/build

# Include any dependencies generated for this target.
include src/kernels/CMakeFiles/causalMaskFunctor.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/kernels/CMakeFiles/causalMaskFunctor.dir/compiler_depend.make

# Include the progress variables for this target.
include src/kernels/CMakeFiles/causalMaskFunctor.dir/progress.make

# Include the compile flags for this target's objects.
include src/kernels/CMakeFiles/causalMaskFunctor.dir/flags.make

src/kernels/CMakeFiles/causalMaskFunctor.dir/causal_mask.cu.o: src/kernels/CMakeFiles/causalMaskFunctor.dir/flags.make
src/kernels/CMakeFiles/causalMaskFunctor.dir/causal_mask.cu.o: ../src/kernels/causal_mask.cu
src/kernels/CMakeFiles/causalMaskFunctor.dir/causal_mask.cu.o: src/kernels/CMakeFiles/causalMaskFunctor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/autodl-tmp/llama_inference/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object src/kernels/CMakeFiles/causalMaskFunctor.dir/causal_mask.cu.o"
	cd /root/autodl-tmp/llama_inference/build/src/kernels && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT src/kernels/CMakeFiles/causalMaskFunctor.dir/causal_mask.cu.o -MF CMakeFiles/causalMaskFunctor.dir/causal_mask.cu.o.d -x cu -dc /root/autodl-tmp/llama_inference/src/kernels/causal_mask.cu -o CMakeFiles/causalMaskFunctor.dir/causal_mask.cu.o

src/kernels/CMakeFiles/causalMaskFunctor.dir/causal_mask.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/causalMaskFunctor.dir/causal_mask.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

src/kernels/CMakeFiles/causalMaskFunctor.dir/causal_mask.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/causalMaskFunctor.dir/causal_mask.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target causalMaskFunctor
causalMaskFunctor_OBJECTS = \
"CMakeFiles/causalMaskFunctor.dir/causal_mask.cu.o"

# External object files for target causalMaskFunctor
causalMaskFunctor_EXTERNAL_OBJECTS =

src/kernels/CMakeFiles/causalMaskFunctor.dir/cmake_device_link.o: src/kernels/CMakeFiles/causalMaskFunctor.dir/causal_mask.cu.o
src/kernels/CMakeFiles/causalMaskFunctor.dir/cmake_device_link.o: src/kernels/CMakeFiles/causalMaskFunctor.dir/build.make
src/kernels/CMakeFiles/causalMaskFunctor.dir/cmake_device_link.o: src/kernels/CMakeFiles/causalMaskFunctor.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/autodl-tmp/llama_inference/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/causalMaskFunctor.dir/cmake_device_link.o"
	cd /root/autodl-tmp/llama_inference/build/src/kernels && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/causalMaskFunctor.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/kernels/CMakeFiles/causalMaskFunctor.dir/build: src/kernels/CMakeFiles/causalMaskFunctor.dir/cmake_device_link.o
.PHONY : src/kernels/CMakeFiles/causalMaskFunctor.dir/build

# Object files for target causalMaskFunctor
causalMaskFunctor_OBJECTS = \
"CMakeFiles/causalMaskFunctor.dir/causal_mask.cu.o"

# External object files for target causalMaskFunctor
causalMaskFunctor_EXTERNAL_OBJECTS =

lib/libcausalMaskFunctor.a: src/kernels/CMakeFiles/causalMaskFunctor.dir/causal_mask.cu.o
lib/libcausalMaskFunctor.a: src/kernels/CMakeFiles/causalMaskFunctor.dir/build.make
lib/libcausalMaskFunctor.a: src/kernels/CMakeFiles/causalMaskFunctor.dir/cmake_device_link.o
lib/libcausalMaskFunctor.a: src/kernels/CMakeFiles/causalMaskFunctor.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/autodl-tmp/llama_inference/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA static library ../../lib/libcausalMaskFunctor.a"
	cd /root/autodl-tmp/llama_inference/build/src/kernels && $(CMAKE_COMMAND) -P CMakeFiles/causalMaskFunctor.dir/cmake_clean_target.cmake
	cd /root/autodl-tmp/llama_inference/build/src/kernels && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/causalMaskFunctor.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/kernels/CMakeFiles/causalMaskFunctor.dir/build: lib/libcausalMaskFunctor.a
.PHONY : src/kernels/CMakeFiles/causalMaskFunctor.dir/build

src/kernels/CMakeFiles/causalMaskFunctor.dir/clean:
	cd /root/autodl-tmp/llama_inference/build/src/kernels && $(CMAKE_COMMAND) -P CMakeFiles/causalMaskFunctor.dir/cmake_clean.cmake
.PHONY : src/kernels/CMakeFiles/causalMaskFunctor.dir/clean

src/kernels/CMakeFiles/causalMaskFunctor.dir/depend:
	cd /root/autodl-tmp/llama_inference/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/autodl-tmp/llama_inference /root/autodl-tmp/llama_inference/src/kernels /root/autodl-tmp/llama_inference/build /root/autodl-tmp/llama_inference/build/src/kernels /root/autodl-tmp/llama_inference/build/src/kernels/CMakeFiles/causalMaskFunctor.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/kernels/CMakeFiles/causalMaskFunctor.dir/depend

