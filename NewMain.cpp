#include <print>
#include <vector>
#include <string>
#include <format>
#include <stdexcept>
#include <chrono>
#include <Eigen/Dense>

// --- Project Headers ---
#include "CompactIO.h"
#include "PolytopeStructs.h"
#include "PolytopeOps.h"
#include "ITree.h"

int main(int argc, char* argv[]) {
    // ---------------------------------------------------------
    // 0. Argument Parsing
    // ---------------------------------------------------------
    if (argc != 3) {
        std::println("Usage: {} <count> <dim>", argv[0]);
        return 1;
    }

    int n_planes = std::stoi(argv[1]);
    int dim = std::stoi(argv[2]);
    std::string filename = std::format("{}_hyperplanes_{}d.bin", n_planes, dim);

    std::println("==========================================");
    std::println("   I-Tree Solver (Iterative Build)        ");
    std::println("Target File: {}", filename);
    std::println("==========================================");

    try {
        // ---------------------------------------------------------
        // 1. Load Data
        // ---------------------------------------------------------
        auto t0 = std::chrono::high_resolution_clock::now();
        CompactDataset ds = CompactIO::load(filename);

        // Convert to Eigen (Batch)
        std::vector<Eigen::VectorXd> planes;
        planes.reserve(ds.count);
        for (size_t i = 0; i < ds.count; ++i) {
            Eigen::VectorXd h(ds.dim);
            for (size_t j = 0; j < ds.dim; ++j) {
                h[j] = ds.at(i, j);
            }
            planes.push_back(h);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        std::println("[1] Loaded {} planes ({:.4f}s)", ds.count, std::chrono::duration<double>(t1-t0).count());

        // ---------------------------------------------------------
        // 2. Initialize Root Domain
        // ---------------------------------------------------------
        // Create Standard Cube [0, 1]^d
        Polytope root_poly = create_hypercube(dim);

        std::println("[2] Initialized Root Domain ([0,1]^{})", dim);

        // ---------------------------------------------------------
        // 3. Build Tree (Iterative)
        // ---------------------------------------------------------
        std::println("[3] Building Tree...");
        auto t2 = std::chrono::high_resolution_clock::now();

        ITreeBuilder builder(root_poly);

        // Pass global reference so nodes can look up plane geometry
        builder.global_planes = &planes;

        // Define the bar width (e.g., 50 characters long)
        const int bar_width = 50;

        for (size_t i = 0; i < planes.size(); ++i) {
            int unique_h_id = (int)i + 1;

            // Update progress every 10 items OR on the very last item
            if (i % 10 == 0 || i == planes.size() - 1) {
                float progress = (float)(i + 1) / planes.size();
                int pos = (int)(bar_width * progress);

                std::print("\r    Progress: [");
                for (int j = 0; j < bar_width; ++j) {
                    if (j < pos) std::print("=");
                    else if (j == pos) std::print(">");
                    else std::print(" ");
                }
                // Print percentage and exact count
                std::print("] {:.1f}% ({}/{})", progress * 100.0, i + 1, planes.size());

                // CRITICAL: Force the console to update immediately
                fflush(stdout);
            }


            // Skip if this plane does NOT partition the root polytope
            int cls = classify_polytope_against_plane(root_poly, planes[i]);
            if (cls != 2) {
                continue;
            }
            // builder.insert_plane(root_poly, planes[i], unique_h_id);
            builder.insert_dfs_non_recursive(root_poly, planes[i], unique_h_id);
            // builder.insert_plane_single_path(root_poly, planes[i], unique_h_id);
        }

        // Important: Print a newline at the end so subsequent output isn't overwritten
        // CORRECT: Empty arguments automatically prints a newline
        std::println(""); // Works everywhere (prints just a newline)

        std::println("\r    Plane {}/{}... Done.", planes.size(), planes.size());

        auto t3 = std::chrono::high_resolution_clock::now();

        // ---------------------------------------------------------
        // 4. Statistics & Output
        // ---------------------------------------------------------
        std::println("\n=== Results ===");
        std::println("Time:         {:.4f} s", std::chrono::duration<double>(t3-t2).count());
        std::println("Total Nodes:  {}", builder.count_nodes());
        std::println("Leaf Cells:   {}", builder.count_leaves());
        std::println("Tree Depth:   {}", builder.compute_depth());   // <-- ADD THIS

    } catch (const std::exception& e) {
        std::println("Error: {}", e.what());
        return 1;
    }

    return 0;
}