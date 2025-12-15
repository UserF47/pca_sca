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
#include "FunctionPairGenerator.h"

int main(int argc, char* argv[]) {
    // ---------------------------------------------------------
    // 0. Argument Parsing
    // ---------------------------------------------------------
    if (argc != 3) {
        std::println("Usage: {} <count> <dim>", argv[0]);
        return 1;
    }

    int n_functions = std::stoi(argv[1]);
    int dim = std::stoi(argv[2]);
    std::string filename = std::format("{}_pairwise_{}d.bin", n_functions, dim);

    // NEW: log file result_{dim}
    std::string log_filename = std::format("result_{}", dim);
    std::ofstream log(log_filename, std::ios::trunc);
    if (!log) {
        throw std::runtime_error(std::format("Failed to open log file: {}", log_filename));
    }

    std::println("==========================================");
    std::println("FS-Tree Solver (Iterative Build)        ");
    std::println("Target File: {}", filename);
    std::println("==========================================");

    try {
        // ---------------------------------------------------------
        // 1. Load Data
        // ---------------------------------------------------------
        auto t0 = std::chrono::high_resolution_clock::now();
        CompactDataset ds = CompactIO::load(filename);
        {
            const std::size_t expected_pairs = static_cast<std::size_t>(n_functions) * (static_cast<std::size_t>(n_functions) - 1) / 2;
            if (ds.count != expected_pairs) {
                throw std::runtime_error(std::format(
                    "Pairwise file count mismatch: expected {} (n*(n-1)/2 for n={}), got {}",
                    expected_pairs, n_functions, ds.count));
            }
            if (static_cast<int>(ds.dim) != dim) {
                throw std::runtime_error(std::format(
                    "Pairwise file dim mismatch: expected {}, got {}",
                    dim, ds.dim));
            }
        }

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

        const int bar_width = 50;
        const std::size_t total_pairs = static_cast<std::size_t>(n_functions) * (static_cast<std::size_t>(n_functions) - 1) / 2;
        std::size_t planes_inserted = 0;

        // Insert in groups: {F1-related}, {F2-related}, ..., {Fn-related}
        for (int fi = 1; fi <= n_functions; ++fi) {
            builder.current_function_id = fi; // Fi is 1-based

            // Fi-related hyperplanes are all pairwise planes (min(fi,fj), max(fi,fj)) for fj != fi

            for (int fj = fi + 1; fj <= n_functions; ++fj) {
                int a = fi;
                int b = fj;

                std::size_t plane_index = Generator::pair_index(a, b, n_functions);
                int unique_h_id = (int)plane_index + 1;

                // Progress bar (over total pairs processed across all groups)
                float progress = static_cast<float>(planes_inserted + 1) /
                                 static_cast<float>(total_pairs);
                int pos = static_cast<int>(bar_width * progress);

                std::print("\r    Progress: [");
                for (int j = 0; j < bar_width; ++j) {
                    if (j < pos) std::print("=");
                    else if (j == pos) std::print(">");
                    else std::print(" ");
                }
                std::print("] {:.1f}% ({}/{})", progress * 100.0, planes_inserted + 1,
                           total_pairs);
                fflush(stdout);

                // Skip if this plane does NOT partition the root polytope
                int cls = classify_polytope_against_plane(root_poly, planes[plane_index]);
                if (cls != 2) {
                    planes_inserted++;
                    continue;
                }

                // Group-aware insertion
                builder.insert_dfs_non_recursive(root_poly, planes[plane_index], unique_h_id);
                planes_inserted++;

                // Periodic stats (every 1000 inserted comparisons)
                if (planes_inserted % 1000 == 0 || planes_inserted == total_pairs) {
                    auto t_now = std::chrono::high_resolution_clock::now();
                    double elapsed = std::chrono::duration<double>(t_now - t2).count();

                    auto total_nodes = builder.count_nodes();
                    auto leaf_cells  = builder.count_leaves();
                    auto depth       = builder.compute_depth();

                    std::println("\n\n=== Results after {} comparisons ===", planes_inserted);
                    std::println("Time:         {:.4f} s", elapsed);
                    std::println("Total Nodes:  {}", total_nodes);
                    std::println("Leaf Cells:   {}", leaf_cells);
                    std::println("Tree Depth:   {}", depth);

                    log << std::format("=== Results after {} comparisons ===\n", planes_inserted);
                    log << std::format("Time:         {:.4f} s\n", elapsed);
                    log << std::format("Total Nodes:  {}\n", total_nodes);
                    log << std::format("Leaf Cells:   {}\n", leaf_cells);
                    log << std::format("Tree Depth:   {}\n\n", depth);
                    log.flush();
                }
            }

            // After finishing the Fi-group, mark all current leaf nodes as Fi-relevant if they are not relevant yet.
            builder.mark_all_leaves_relevant(fi);
        }

        std::println("\r    Comparisons {}/{}... Done.", planes_inserted,
                     total_pairs);

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