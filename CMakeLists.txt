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
        LPSolver.h
        LPTree.h
        main_lp.cpp
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

# --- Target 4: LP-Based Solver (Using HiGHS) ---

# 1. Find HiGHS in Homebrew paths
# --- HiGHS Sanity Check ---

# 1. Find HiGHS (Homebrew Paths)
find_path(HIGHS_INCLUDE_DIR Highs.h
        PATHS /opt/homebrew/include/highs /usr/local/include/highs
)
find_library(HIGHS_LIBRARY highs
        PATHS /opt/homebrew/lib /usr/local/lib
)

if (NOT HIGHS_INCLUDE_DIR OR NOT HIGHS_LIBRARY)
    message(WARNING "HiGHS not found! Targets 'lp_solver' and 'highs_test' will not be built.")
else()
    # --- Target 1: The Main LP Solver Application ---
    add_executable(lp_solver
            main_lp.cpp
            # Helper headers (LPSolver.h, LPTree.h, etc are included implicitly)
            CompactIO.h
    )
    target_include_directories(lp_solver PRIVATE ${HIGHS_INCLUDE_DIR})
    target_link_libraries(lp_solver PRIVATE ${HIGHS_LIBRARY})

endif()