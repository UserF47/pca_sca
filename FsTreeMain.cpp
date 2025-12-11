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
#include "FsTree.h"

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
        // 3. Build Tree (Grouped by Function)
        // ---------------------------------------------------------
        std::println("[3] Building Tree (grouped insertion)...");
        auto t2 = std::chrono::high_resolution_clock::now();

        ITreeBuilder builder(root_poly);
        builder.global_planes = &planes;

        // For now, assume each hyperplane corresponds to one "function group":
        // group i contains hyperplane i only.
        // Later, you can replace this with a real Fi-relevant mapping.
        const std::size_t n_functions = planes.size();
        const int bar_width = 50;
        std::size_t planes_inserted = 0;

        for (std::size_t fi = 0; fi < n_functions; ++fi) {
            builder.current_function_id = static_cast<int>(fi) + 1; // Fi is 1-based

            // Here you could loop over all Fi-relevant hyperplanes.
            // For now, we assume only hyperplane fi belongs to group Fi.
            std::size_t plane_index = fi;
            int unique_h_id = static_cast<int>(plane_index) + 1;

            // Progress bar (over all planes)
            float progress = static_cast<float>(planes_inserted + 1) / static_cast<float>(planes.size());
            int pos = static_cast<int>(bar_width * progress);

            std::print("\r    Progress: [");
            for (int j = 0; j < bar_width; ++j) {
                if (j < pos) std::print("=");
                else if (j == pos) std::print(">");
                else std::print(" ");
            }
            std::print("] {:.1f}% ({}/{})", progress * 100.0, planes_inserted + 1, planes.size());
            fflush(stdout);

            // Skip if this plane does NOT partition the root polytope
            int cls = classify_polytope_against_plane(root_poly, planes[plane_index]);
            if (cls != 2) {
                planes_inserted++;
                continue;
            }

            // Group-aware insertion: leaves created here will be marked as
            // relevant for current_function_id and will carry sample points.
            builder.insert_dfs_non_recursive(root_poly, planes[plane_index], unique_h_id);
            planes_inserted++;

            // Periodic stats (every 1000 planes)
            if (planes_inserted % 1000 == 0 || planes_inserted == planes.size()) {
                auto t_now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(t_now - t2).count();

                auto total_nodes = builder.count_nodes();
                auto leaf_cells  = builder.count_leaves();
                auto depth       = builder.compute_depth();

                std::println("\n\n=== Results after {} planes ===", planes_inserted);
                std::println("Time:         {:.4f} s", elapsed);
                std::println("Total Nodes:  {}", total_nodes);
                std::println("Leaf Cells:   {}", leaf_cells);
                std::println("Tree Depth:   {}", depth);

                log << std::format("=== Results after {} planes ===\n", planes_inserted);
                log << std::format("Time:         {:.4f} s\n", elapsed);
                log << std::format("Total Nodes:  {}\n", total_nodes);
                log << std::format("Leaf Cells:   {}\n", leaf_cells);
                log << std::format("Tree Depth:   {}\n\n", depth);
                log.flush();
            }
        }

        std::println("");
        std::println("\r    Plane {}/{}... Done.", planes_inserted, planes.size());

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