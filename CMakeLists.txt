cmake_minimum_required(VERSION 3.28)
project(pca_sca)

# --- Global Settings ---
# Force C++26 Standard
set(CMAKE_CXX_STANDARD 26)


# Compiler Options
if(MSVC)
    add_compile_options(/std:c++latest /O2 /W4)
else()
    add_compile_options(-std=c++2b -O3 -Wall -Wextra -march=native)
endif()


# --- Target 1: The Data Generation Tool ---
add_executable(gen_data
        generator.cpp       # <--- Correct
        DataGenerator.cpp
)

# --- Target 2: The Main Research Application ---
add_executable(pca_sca
        main.cpp
)