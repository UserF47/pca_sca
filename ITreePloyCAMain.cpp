#include <print>
#include <vector>
#include <string>
#include <format>
#include <stdexcept>
#include <chrono>
#include <fstream>
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
    std::string filename = std::format("{}_pairwise_{}d.bin", n_planes, dim);

    // NEW: log file result_{dim}
    std::string log_filename = std::format("result_{}", dim);
    std::ofstream log(log_filename, std::ios::trunc);
    if (!log) {
        throw std::runtime_error(std::format("Failed to open log file: {}", log_filename));
    }

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

            // Insert plane
            builder.insert_dfs_non_recursive(root_poly, planes[i], unique_h_id);

            // === NEW: print stats every 10,000 planes processed ===
            if ((i + 1) % 1000 == 0) {
                auto t_now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(t_now - t2).count();

                auto total_nodes = builder.count_nodes();
                auto leaf_cells  = builder.count_leaves();
                auto depth       = builder.compute_depth();

                // Console
                std::println("\n\n=== Results after {} planes ===", i + 1);
                std::println("Time:         {:.4f} s", elapsed);
                std::println("Total Nodes:  {}", total_nodes);
                std::println("Leaf Cells:   {}", leaf_cells);
                std::println("Tree Depth:   {}", depth);

                // Log file
                log << std::format("=== Results after {} planes ===\n", i + 1);
                log << std::format("Time:         {:.4f} s\n", elapsed);
                log << std::format("Total Nodes:  {}\n", total_nodes);
                log << std::format("Leaf Cells:   {}\n", leaf_cells);
                log << std::format("Tree Depth:   {}\n\n", depth);
                log.flush();
            }
        }

        // Important: Print a newline at the end so subsequent output isn't overwritten
        std::println(""); // Works everywhere (prints just a newline)

        std::println("\r    Plane {}/{}... Done.", planes.size(), planes.size());

        auto t3 = std::chrono::high_resolution_clock::now();

        // ---------------------------------------------------------
        // 4. Statistics & Output (final snapshot)
        // ---------------------------------------------------------
        double total_time = std::chrono::duration<double>(t3 - t2).count();
        auto total_nodes_final = builder.count_nodes();
        auto leaf_cells_final  = builder.count_leaves();
        auto depth_final       = builder.compute_depth();

        std::println("\n=== Results ===");
        std::println("Time:         {:.4f} s", total_time);
        std::println("Total Nodes:  {}", total_nodes_final);
        std::println("Leaf Cells:   {}", leaf_cells_final);
        std::println("Tree Depth:   {}", depth_final);

        // Also write final result to log
        log << "=== Final Results ===\n";
        log << std::format("Time:         {:.4f} s\n", total_time);
        log << std::format("Total Nodes:  {}\n", total_nodes_final);
        log << std::format("Leaf Cells:   {}\n", leaf_cells_final);
        log << std::format("Tree Depth:   {}\n", depth_final);
        log.flush();

    } catch (const std::exception& e) {
        std::println("Error: {}", e.what());
        return 1;
    }

    return 0;
}