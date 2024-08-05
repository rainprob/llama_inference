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
include tests/unittests/CMakeFiles/test_fused_addresidual_norm.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/unittests/CMakeFiles/test_fused_addresidual_norm.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/unittests/CMakeFiles/test_fused_addresidual_norm.dir/progress.make

# Include the compile flags for this target's objects.
include tests/unittests/CMakeFiles/test_fused_addresidual_norm.dir/flags.make

tests/unittests/CMakeFiles/test_fused_addresidual_norm.dir/test_fused_addresidual_norm.cu.o: tests/unittests/CMakeFiles/test_fused_addresidual_norm.dir/flags.make
tests/unittests/CMakeFiles/test_fused_addresidual_norm.dir/test_fused_addresidual_norm.cu.o: ../tests/unittests/test_fused_addresidual_norm.cu
tests/unittests/CMakeFiles/test_fused_addresidual_norm.dir/test_fused_addresidual_norm.cu.o: tests/unittests/CMakeFiles/test_fused_addresidual_norm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/autodl-tmp/llama_inference/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object tests/unittests/CMakeFiles/test_fused_addresidual_norm.dir/test_fused_addresidual_norm.cu.o"
	cd /root/autodl-tmp/llama_inference/build/tests/unittests && /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT tests/unittests/CMakeFiles/test_fused_addresidual_norm.dir/test_fused_addresidual_norm.cu.o -MF CMakeFiles/test_fused_addresidual_norm.dir/test_fused_addresidual_norm.cu.o.d -x cu -c /root/autodl-tmp/llama_inference/tests/unittests/test_fused_addresidual_norm.cu -o CMakeFiles/test_fused_addresidual_norm.dir/test_fused_addresidual_norm.cu.o

tests/unittests/CMakeFiles/test_fused_addresidual_norm.dir/test_fused_addresidual_norm.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/test_fused_addresidual_norm.dir/test_fused_addresidual_norm.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

tests/unittests/CMakeFiles/test_fused_addresidual_norm.dir/test_fused_addresidual_norm.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/test_fused_addresidual_norm.dir/test_fused_addresidual_norm.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target test_fused_addresidual_norm
test_fused_addresidual_norm_OBJECTS = \
"CMakeFiles/test_fused_addresidual_norm.dir/test_fused_addresidual_norm.cu.o"

# External object files for target test_fused_addresidual_norm
test_fused_addresidual_norm_EXTERNAL_OBJECTS =

bin/test_fused_addresidual_norm: tests/unittests/CMakeFiles/test_fused_addresidual_norm.dir/test_fused_addresidual_norm.cu.o
bin/test_fused_addresidual_norm: tests/unittests/CMakeFiles/test_fused_addresidual_norm.dir/build.make
bin/test_fused_addresidual_norm: lib/libfused_addresidual_norm.a
bin/test_fused_addresidual_norm: tests/unittests/CMakeFiles/test_fused_addresidual_norm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/autodl-tmp/llama_inference/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable ../../bin/test_fused_addresidual_norm"
	cd /root/autodl-tmp/llama_inference/build/tests/unittests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_fused_addresidual_norm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/unittests/CMakeFiles/test_fused_addresidual_norm.dir/build: bin/test_fused_addresidual_norm
.PHONY : tests/unittests/CMakeFiles/test_fused_addresidual_norm.dir/build

tests/unittests/CMakeFiles/test_fused_addresidual_norm.dir/clean:
	cd /root/autodl-tmp/llama_inference/build/tests/unittests && $(CMAKE_COMMAND) -P CMakeFiles/test_fused_addresidual_norm.dir/cmake_clean.cmake
.PHONY : tests/unittests/CMakeFiles/test_fused_addresidual_norm.dir/clean

tests/unittests/CMakeFiles/test_fused_addresidual_norm.dir/depend:
	cd /root/autodl-tmp/llama_inference/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/autodl-tmp/llama_inference /root/autodl-tmp/llama_inference/tests/unittests /root/autodl-tmp/llama_inference/build /root/autodl-tmp/llama_inference/build/tests/unittests /root/autodl-tmp/llama_inference/build/tests/unittests/CMakeFiles/test_fused_addresidual_norm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/unittests/CMakeFiles/test_fused_addresidual_norm.dir/depend
