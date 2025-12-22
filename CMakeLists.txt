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
add_executable(ITreePloyCA
        ITreePloyCAMain.cpp    # <--- Your new test file
        PolytopeStructs.h
        PolytopeOps.h
        ITreePloyCA.h
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
    add_executable(ITreeSimplex
            ItreeSimplexMain.cpp
            # Helper headers (LPSolver.h, LPTree.h, etc are included implicitly)
            CompactIO.h
    )
    target_include_directories(ITreeSimplex PRIVATE ${HIGHS_INCLUDE_DIR})
    target_link_libraries(ITreeSimplex PRIVATE ${HIGHS_LIBRARY})

endif()

# --- Target 5: cddlib Double Description Test ---

# Hardcode cddlib paths for this Mac (Homebrew)
set(CDD_ROOT "/opt/homebrew/opt/cddlib")
set(CDD_INCLUDE_DIR "${CDD_ROOT}/include")
set(CDD_LIBRARY "${CDD_ROOT}/lib/libcdd.dylib")

if (NOT EXISTS "${CDD_INCLUDE_DIR}/cddlib/cdd.h" OR NOT EXISTS "${CDD_LIBRARY}")
    message(WARNING "cddlib not found at ${CDD_ROOT}! Target 'ddm_test' will not be built.")
else()
    add_executable(ddm_test
            ddm_test.cpp
            FunctionPairGenerator.h
            FunctionPairGenerator.cpp
    )
    target_include_directories(ddm_test PRIVATE ${CDD_INCLUDE_DIR})
    target_link_libraries(ddm_test PRIVATE ${CDD_LIBRARY})
endif()

# --- Target 6: DDM Vertex Computation Tool ---
if (EXISTS "${CDD_INCLUDE_DIR}/cddlib/cdd.h" AND EXISTS "${CDD_LIBRARY}")
    add_executable(ddm_vertex_tree
            ITreeVertexDDMMain.cpp
            DDMSolver.h
            DDMTree.h
            CompactIO.h
    )
    target_include_directories(ddm_vertex_tree PRIVATE ${CDD_INCLUDE_DIR})
    target_link_libraries(ddm_vertex_tree PRIVATE ${CDD_LIBRARY})
else()
    message(WARNING "cddlib not found! Target 'ddm_vertex_tree' will not be built.")
endif()

# --- Target 7: The Test ori poly ---
add_executable(ITreePolyDomain
        ITreePolyDomainMain.cpp    # <--- Your new test file
        PolytopeStructs.h
        PolytopeOps.h
        ITreePolyDomain.h
        CompactIO.h
)

# --- Target 8: Test data generator for FS-tree ---
add_executable(FunctionGenerator
        FunctionPairMain.cpp
        FunctionPairGenerator.h
        FunctionPairGenerator.cpp
)

# --- Target 9: The Function sorting Tree ---
# Create a file named 'new_main.cpp' to use this!
add_executable(FsTree
        FsTreeMain.cpp
        PolytopeStructs.h
        PolytopeOps.h
        FsTree.h
        CompactIO.h
)

## AddressSanitizer for FsTree
#target_compile_options(FsTree PRIVATE
#        -fsanitize=address
#        -fno-omit-frame-pointer
#)
#
#target_link_options(FsTree PRIVATE
#        -fsanitize=address
#)

# --- Target 10: The OnDemand Function sorting Tree ---
# Create a file named 'new_main.cpp' to use this!
add_executable(FsTreeOnDemandPoint
        FsTreeOnDemandMain.cpp
        PolytopeStructs.h
        PolytopeOps.h
        FsTreeOnDemand.h
        CompactIO.h
)