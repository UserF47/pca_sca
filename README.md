# A Polytope-Centric Approach for Efficient Domain Partitioning

---

## 🛠️ Prerequisites

Before building, ensure you have the following installed:

1. **C++ Compiler** — supporting **C++20** or newer  
   (Clang 15+, GCC 11+, AppleClang 15+)
2. **CMake** — version **3.28+**
3. **Eigen3** — required for linear algebra operations

### Install Eigen3

**macOS (Homebrew):**
```bash
brew install eigen
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get install libeigen3-dev
```

---

## 🏗️ Build Instructions

1. **Navigate to the project root**
```bash
cd /path/to/pca_sca
```

2. **Create a build directory**
```bash
mkdir cmake-build-debug
cd cmake-build-debug
```

3. **Configure using CMake**  
   (*CMake is set to auto-detect Apple Silicon optimizations such as `-march=native`.*)
```bash
cmake ..
```

4. **Build**
```bash
make -j
```

This produces two executables:

- **`gen_data`** — generates random hyperplane datasets
- **`pca_sca`** — the main I-Tree constructor

---

## 🚀 How to Run

The workflow consists of two stages:

1. Generate intersection(hyperplane) data
2. Run the solver to construct the I-Tree

---

## Step 1: Generate Data

Use **`gen_data`** to create a binary file containing random hyperplanes.

### Usage
```bash
./gen_data <count> <dimension>
```

### Example
Generate **100 hyperplanes in 2D**:
```bash
./gen_data 100 2
```

This creates:
```
100_hyperplanes_2d.bin
```

---

## Step 2: Run the Solver

Use **`pca_sca`** to load the binary file and build the I-Tree.

### Usage
```bash
./pca_sca <count> <dimension>
```

⚠️ **Important:** The arguments **must match** the filename generated in Step 1.

### Example
```bash
./pca_sca 100 2
```

---

## 📊 Expected Output

A typical run looks like:

```
==========================================
         I-Tree Solver (Incremental Build)
==========================================
Target File: 100_hyperplanes_2d.bin

[1] Loaded Data: 100 hyperplanes in 2 dimensions (0.0002s)
[2] Initializing Domain (Centered Unit Cube)...
        Root Complexity: 4 Vertices, 4 Edges
[3] Building I-Tree (Incremental Insertion)...
        Inserting Plane 100/100... Done.

=== Build Complete ===
Time Taken:   0.0045 seconds
Total Nodes:  234
Leaf Cells:   118
Theory Max (Central): 200
```

---

