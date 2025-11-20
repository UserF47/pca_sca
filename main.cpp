#include <print>
#include <vector>
#include <string>
#include <format>
#include <stdexcept>
#include <chrono>

#include <Eigen/Dense>

// Project Headers
#include "CompactIO.h"
#include "PolytopeStructs.h"
#include "UnitCube.h"
#include "PolytopeOps.h"
#include "ITree.h"

// Usage: ./pca_sca <count> <dim>
int main(int argc, char* argv[]) {
    // ---------------------------------------------------------
    // 0. Argument Parsing
    // ---------------------------------------------------------
    if (argc != 3) {
        std::println("Usage: {} <count> <dim>", argv[0]);
        std::println("Example: {} 100 2", argv[0]);
        return 1;
    }

    std::string n_str = argv[1];
    std::string d_str = argv[2];
    std::string filename = std::format("{}_hyperplanes_{}d.bin", n_str, d_str);

    std::println("==========================================");
    std::println("   I-Tree Solver (Incremental Build)      ");
    std::println("==========================================");
    std::println("Target File: {}", filename);

    try {
        // ---------------------------------------------------------
        // 1. Load Hyperplane Data
        // ---------------------------------------------------------
        auto t0 = std::chrono::high_resolution_clock::now();

        CompactDataset ds = CompactIO::load(filename);

        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> load_time = t1 - t0;
        std::println("[1] Loaded Data: {} hyperplanes in {} dimensions ({:.4f}s)",
                     ds.count, ds.dim, load_time.count());

        // Convert to Eigen vectors for processing
        std::vector<Eigen::VectorXd> planes;
        planes.reserve(ds.count);
        for (size_t i = 0; i < ds.count; ++i) {
            Eigen::VectorXd h(ds.dim);
            for (size_t j = 0; j < ds.dim; ++j) {
                h[j] = ds.at(i, j);
            }
            planes.push_back(h);
        }

        // ---------------------------------------------------------
        // 2. Initialize Domain (Unit Cube)
        // ---------------------------------------------------------
        // We use the centered cube [-1, 1] to ensure planes through the origin
        // actually slice the domain, maximizing the tree complexity.
        std::println("[2] Initializing Domain (Centered Unit Cube)...");

        // NOTE: Ensure your UnitCube.h supports the 3rd 'true' argument for centered generation
        // If not, remove 'true' (but you will get fewer leaves).
        Polytope root_poly = UnitCubeGenerator::make(ds.dim, ds.count, true);

        std::println("    Root Complexity: {} Vertices, {} Edges",
                     root_poly.vertices.size(), root_poly.edges.size());

        // ---------------------------------------------------------
        // 3. Incremental I-Tree Construction
        // ---------------------------------------------------------
        std::println("[3] Building I-Tree (Incremental Insertion)...");
        auto t2 = std::chrono::high_resolution_clock::now();

        ITreeBuilder builder;

        // The Loop: Insert every plane one by one into the tree
        for (size_t i = 0; i < planes.size(); ++i) {
            // Progress indicator for large datasets
            if (i % 100 == 0 && i > 0) {
                std::print("\r    Inserting Plane {}/{}...", i, planes.size());
            }

            // Strategy: Compute P = h_i intersect D, then filter P down the tree
            builder.insert_plane(root_poly, (int)i, planes);
        }
        std::println("\r    Inserting Plane {}/{}... Done.", planes.size(), planes.size());

        auto t3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> build_time = t3 - t2;

        // ---------------------------------------------------------
        // 4. Statistics & Output
        // ---------------------------------------------------------
        // We count leaves by traversing the tree
        int leaf_count = builder.count_leaves(builder.root.get());

        std::println("\n=== Build Complete ===");
        std::println("Time Taken:   {:.4f} seconds", build_time.count());
        std::println("Total Nodes:  {}", builder.node_counter);
        std::println("Leaf Cells:   {}", leaf_count);

        // Theoretical Maximum for Centered Arrangement
        if (ds.dim == 2) {
            long long n = ds.count;
            long long max_regions = 2 * n; // Central arrangement max is 2N
            std::println("Theory Max (Central): {}", max_regions);
        }

    } catch (const std::exception& e) {
        std::println("\n!!! FATAL ERROR !!!");
        std::println("Reason: {}", e.what());
        return 1;
    }

    return 0;
}