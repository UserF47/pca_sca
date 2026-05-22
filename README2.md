## Requirements

### C++ Implementation
- **Compiler**: g++ with C++17 support
- **Libraries**: 
  - Eigen3 (linear algebra)
  - Standard C++ libraries
- **OS**: Linux (tested on Ubuntu 24.04 / WSL2)

### Python Visualization
- Python 3.7+
- Required packages:
```bash
  pip install pandas matplotlib numpy
```

## Building and Running

### Compile Benchmarks

```bash
# Compile all benchmarks
g++ -std=c++17 -O3 -I. -I/usr/include/eigen3 benchmarks/final_with_d5_prisms_csv.cpp -o build/feasibility_benchmark
g++ -std=c++17 -O3 -I. -I/usr/include/eigen3 benchmarks/itree_d2_csv.cpp -o build/itree_d2_benchmark
g++ -std=c++17 -O3 -I. -I/usr/include/eigen3 benchmarks/itree_d3_csv.cpp -o build/itree_d3_benchmark
```

### Run Experiments

```bash
# Create output directory
mkdir -p data

# Run feasibility checking experiments (d=2 through d=10)
./build/feasibility_benchmark
# Output: data/feasibility_data.csv

# Run I-tree construction experiments
./build/itree_d2_benchmark
# Output: data/itree_d2_data.csv

./build/itree_d3_benchmark
# Output: data/itree_d3_data.csv
```

### Generate Visualizations

```bash
cd visualization

# Generate feasibility checking plots
python plot_feasibility.py
# Output: feasibility_d2.png, feasibility_d3.png, ..., feasibility_d10.png
#         feasibility_all_dimensions.png, feasibility_d_sweep.png

# Generate I-tree construction plots
python plot_itree.py
# Output: itree_d2_bar.png, itree_d3_bar.png, itree_combined.png
```

## Experimental Design

### Feasibility Checking Experiments

**d=2-5 (Deterministic Geometric Constructions):**
- **d=2**: Regular n-gons (n = 10, 20, 50, 100, 150, 200, 300, 500, 1000)
- **d=3**: Prisms (n-gon × [0,1])
- **d=4**: Products of polygons (m-gon × n-gon)
- **d=5**: Torus prisms ((m×n)-torus × [0,1])

**d=6-10 (Stochastic Polytope Generation):**
- Start with d-dimensional unit hypercube
- Apply 30-120 random hyperplane splits
- Random normals from N(0,1), offsets from [-2, 2]
- Fixed seed for reproducibility

**Measurements:**
- 1000 trials per configuration
- Vertex-based (baseline): O(n·d)
- Centroid-bucketing (ours): O(d²)
- Speedup = vertex_time / bucket_time

### I-Tree Construction Experiments

**d=2 Configurations:**
- n=100, 10 functions (45 intersections)
- n=500, 15 functions (105 intersections)
- n=1000, 20 functions (190 intersections)

**d=3 Configurations:**
- n=40, 8 functions (28 intersections)
- n=100, 10 functions (45 intersections)
- n=200, 12 functions (66 intersections)

**Function Generation:**
- Linear functions: f_i(x) = a_i · x + b_i
- Coefficients from N(0,1)
- Fixed seed (42 + n) for reproducibility

## Key Results

### Feasibility Checking
| Dimension | Crossover Point | Max Speedup |
|-----------|----------------|-------------|
| d=2 | n ≈ 150-200 | 5.30× |
| d=3 | n ≈ 1000-1400 | 1.47× |
| d=7 | Not visible* | 2.58× |
| d=10 | Not visible* | 4.97× |

*All test points in speedup regime due to construction method

### I-Tree Construction
| Dimension | Configuration | Speedup |
|-----------|--------------|---------|
| d=2 | n=500, 15 funcs | 3.43× |
| d=2 | n=1000, 20 funcs | 2.90× |
| d=3 | n=100, 10 funcs | 1.65× |
| d=3 | n=200, 12 funcs | 1.61× |

### Correctness Validation
- **91,864 total feasibility checks** across all experiments
- **Zero disagreements** between baseline and centroid-bucketing
- **Zero window failures** (2d+1 window guarantee validated)

## Implementation Details

### Core Algorithm (SphericalBucketing_v4.h)

**Indexing Phase (one-time per polytope):**
1. Compute facet normals and centroids
2. Convert normals to spherical coordinates (azimuthal angle θ)
3. Sort facets by θ in O(m log m)
4. Compute midpoint boundaries between consecutive normals

**Query Phase (per feasibility check):**
1. Compute query normal θ_q
2. Binary search to locate bucket: O(log m)
3. Check 2d+1 window candidates: O(d²)
4. Identify F_max and F_min facets
5. Evaluate g(v) only on vertices of extremal facets: O(k·d)

**Recursive Index Propagation (per split):**
1. Edge scan: O(n·d) (unchanged from baseline)
2. Vertex assignment: O(n)
3. Facet assignment: O(m)
4. Compute new shared facet normal: O(d)
5. Inherit parent index and insert new facet: O(m)