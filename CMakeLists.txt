cmake_minimum_required(VERSION 3.28)
project(pca_sca)

# --- Global Settings ---
set(CMAKE_CXX_STANDARD 26)

# 1. FIND EIGEN (Essential for M2 Mac)
# This tells the compiler where to look for <Eigen/Dense>
include_directories("/opt/homebrew/include/eigen3")

# Compiler Options
if(MSVC)
    add_compile_options(/std:c++latest /O2 /W4)
else()
    # -march=native is crucial on M2 to enable NEON (SIMD) instructions
    add_compile_options(-std=c++2b -O3 -Wall -Wextra -march=native)
endif()

# --- Target 1: The Data Generation Tool ---
add_executable(gen_data
        generator.cpp
        DataGenerator.cpp
)

# --- Target 2: The Main Research Application ---
#add_executable(pca_sca
#        main.cpp
#        PolytopeStructs.h
#        UnitCube.h
#        PolytopeOps.h
#        ITree.h       # <--- Updated
#)

# --- Target 3: The Test Playground ---
# Create a file named 'test_polytope.cpp' to use this!
add_executable(test_runner
        test_polytope.cpp    # <--- Your new test file
        PolytopeStructs.h
        PolytopeOps.h
)

# --- Target 4: The Test New Main ---
# Create a file named 'new_main.cpp' to use this!
add_executable(new_main_runner
        NewMain.cpp    # <--- Your new test file
        PolytopeStructs.h
        PolytopeOps.h
        ITree.h
        CompactIO.h
)