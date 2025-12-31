# Fast Linear Function Sorting

A C++ research codebase for **function sorting** experiments (e.g., ITree / FsTree and on-demand variants). It also includes dataset generators and optional solvers (HiGHS Simplex, cddlib DDM).

> **Build system:** CMake  \
> **C++ standard:** C++26 (`CMAKE_CXX_STANDARD = 26`)

---

## Requirements

### Core

- **CMake ≥ 3.28**
- A C++ compiler that supports **C++26 / latest**
  - macOS: AppleClang (Xcode)
  - Linux: recent GCC/Clang
  - Windows: MSVC (latest)

### Eigen

This project includes Eigen using a Homebrew include path on Apple Silicon:

- `include_directories("/opt/homebrew/include/eigen3")`

Install:

- macOS (Homebrew)
  ```bash
  brew install eigen
  ```
- Ubuntu/Debian
  ```bash
  sudo apt-get update
  sudo apt-get install -y libeigen3-dev
  ```

If Eigen is installed elsewhere, update the include path in `CMakeLists.txt` (or switch to `find_package(Eigen3 ...)`).

---

## Quick start

```bash
git clone <your-repo-url>
cd pca_sca

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Executables will be under `build/`.

---

## Build

### Release (recommended)
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Debug
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

### Build a specific target
```bash
cmake --build build --target FsTree -j
```

---

## Targets

This repository builds multiple executables (as defined in `CMakeLists.txt`).

### Data generation

- **`gen_data`**
  - Sources: `generator.cpp`, `DataGenerator.cpp`
  - Purpose: dataset generation tool.

- **`FunctionGenerator`**
  - Sources: `FunctionPairMain.cpp`, `FunctionPairGenerator.h`, `FunctionPairGenerator.cpp`
  - Purpose: generate function-pair test data (used by FS-tree experiments).

### Polytope / ITree experiments

- **`test_runner`**
  - Sources: `test_polytope.cpp`, `PolytopeStructs.h`, `PolytopeOps.h`
  - Purpose: playground for polytope operations.

- **`ITreePloyCA`**
  - Sources: `ITreePloyCAMain.cpp`, `ITreePloyCA.h`, `PolytopeStructs.h`, `PolytopeOps.h`, `CompactIO.h`
  - Purpose: ITree polytope-centric approach.

- **`ITreePolyDomain`**
  - Sources: `ITreePolyDomainMain.cpp`, `ITreePolyDomain.h`, `PolytopeStructs.h`, `PolytopeOps.h`, `CompactIO.h`
  - Purpose: original poly-domain variant.

### FS-tree experiments

- **`FsTree`**
  - Sources: `FsTreeMain.cpp`, `FsTree.h`, `PolytopeStructs.h`, `PolytopeOps.h`, `CompactIO.h`

- **`FsTreeOnDemandPoint`**
  - Sources: `FsTreeOnDemandMain.cpp`, `FsTreeOnDemand.h`, `PolytopeStructs.h`, `PolytopeOps.h`, `CompactIO.h`

- **`FsTreeOnDemandDomain`**
  - Sources: `FsTreeOnDemandDomainMain.cpp`, `FsTreeOnDemandDomain.h`, `PolytopeStructs.h`, `PolytopeOps.h`, `CompactIO.h`

### Baseline sorting

- **`SimpleSortingFunctionsByPoint`**
  - Source: `SortFunctionPoint.cpp`
  - Purpose: baseline sorting by a single query point.

### Optional targets (built only if dependencies are found)

- **`ITreeSimplex`** (requires **HiGHS**)
- **`ddm_test`**, **`ddm_vertex_tree`** (require **cddlib**)

---

## Optional dependencies

### HiGHS (LP/Simplex)

If HiGHS is found, CMake builds:

- **`ITreeSimplex`** (links against `highs`)

CMake searches:

- headers: `Highs.h` under `/opt/homebrew/include/highs` or `/usr/local/include/highs`
- library: `libhighs` under `/opt/homebrew/lib` or `/usr/local/lib`

Install:

```bash
brew install highs
```

If CMake warns that HiGHS is not found, `ITreeSimplex` will not be built.

---

### cddlib (DDM / Double Description)

If cddlib is found at the Homebrew path, CMake builds:

- **`ddm_test`**
- **`ddm_vertex_tree`**

Expected path (macOS Homebrew):

- `CDD_ROOT = /opt/homebrew/opt/cddlib`
- header: `/opt/homebrew/opt/cddlib/include/cddlib/cdd.h`
- library: `/opt/homebrew/opt/cddlib/lib/libcdd.dylib`

Install:

```bash
brew install cddlib
```

---

## Run examples

(Replace arguments/options based on each program’s CLI.)

```bash
# Data generation
./build/gen_data

# Function-pair generation
./build/FunctionGenerator

# Polytope test playground
./build/test_runner

# FS-tree
./build/FsTree

# On-demand variants
./build/FsTreeOnDemandPoint
./build/FsTreeOnDemandDomain

# Baseline sorting by point
./build/SimpleSortingFunctionsByPoint
```

---

## Notes for Apple Silicon (M1/M2/M3)

Non-MSVC builds use:

- `-march=native` (enables Apple Silicon SIMD / NEON)
- `-O3`
- `-Wall -Wextra`

Eigen is included via:

- `/opt/homebrew/include/eigen3`

---

## Troubleshooting

### “Eigen/Dense not found”

Install Eigen and/or fix the include path:

```bash
brew install eigen
```

### HiGHS targets not built

`ITreeSimplex` is only built if CMake finds both `Highs.h` and `libhighs`.

```bash
brew install highs
```

### cddlib targets not built

`ddm_test` / `ddm_vertex_tree` are only built if cddlib is found at `/opt/homebrew/opt/cddlib`.

```bash
brew install cddlib
```

### Enable AddressSanitizer (optional)

There is commented ASAN setup for `FsTree` in `CMakeLists.txt`. Uncomment it, then rebuild:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target FsTree -j
```

---

## License

Add your license here (e.g., MIT, Apache-2.0), or keep this section for internal research code.

## Citation

If this code supports a paper, add a BibTeX entry here.
