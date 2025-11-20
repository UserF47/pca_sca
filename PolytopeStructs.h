#pragma once
#include <vector>
#include <cstdint>
#include <bit> // For std::popcount (C++20/23)
#include <Eigen/Dense>

// --- 1. Combinatorial Layer (The "DNA") ---

struct ConstraintSet {
    // We store bits in chunks of 64.
    // vector<uint64_t> is much faster than std::vector<bool> for bulk operations.
    std::vector<uint64_t> chunks;

    // Resize to hold 'n_planes' bits
    void resize(size_t n_planes) {
        chunks.assign((n_planes + 63) / 64, 0);
    }

    // Set a specific bit (Hyperplane index)
    void set(int index) {
        chunks[index / 64] |= (1ULL << (index % 64));
    }

    // Check a specific bit
    bool test(int index) const {
        return (chunks[index / 64] >> (index % 64)) & 1ULL;
    }

    // Intersection (Bitwise AND)
    // Returns a new set representing constraints present in BOTH
    ConstraintSet operator&(const ConstraintSet& other) const {
        ConstraintSet result = *this;
        // Ideally, this loop is unrolled by the compiler into SIMD instructions
        for (size_t i = 0; i < chunks.size(); ++i) {
            result.chunks[i] &= other.chunks[i];
        }
        return result;
    }

    // Population Count (Number of set bits)
    int count() const {
        int c = 0;
        for (uint64_t chunk : chunks) {
            c += std::popcount(chunk);
        }
        return c;
    }
};

// --- 2. Geometric Layer (The Explicit Graph) ---

struct Vertex {
    int id;
    Eigen::VectorXd position;
    ConstraintSet constraints; // The set of hyperplanes this vertex lies on
};

struct Edge {
    int id;
    int v1, v2;                 // Indices into the Polytope::vertices vector
    std::vector<int> face_ids;  // Indices into Polytope::faces
};

struct Face {
    int id;
    std::vector<int> edge_ids;  // Indices into Polytope::edges (forms a closed loop)
};

struct Polytope {
    std::vector<Vertex> vertices;
    std::vector<Edge> edges;
    std::vector<Face> faces;
    int dim; // Dimension of the ambient space (d)
};