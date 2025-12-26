#include <print>
#include <vector>
#include <string>
#include <format>
#include <stdexcept>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <Eigen/Dense>

// --- Project Headers ---
#include "CompactIO.h"
#include "PolytopeStructs.h"
#include "PolytopeOps.h"
#include "FsTreeOnDemand.h"
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

    namespace fs = std::filesystem;

    auto ensure_logs_dir = []() -> fs::path {
        fs::path logs_dir = "logs_FsTree_Storage";
        std::error_code ec;
        fs::create_directories(logs_dir, ec);
        return logs_dir;
    };

    // NEW: log file result_{dim}
    std::string log_filename = std::format("result_{}", dim);
    std::ofstream log(log_filename, std::ios::trunc);
    if (!log) {
        throw std::runtime_error(std::format("Failed to open log file: {}", log_filename));
    }

    // NEW: per-fi FC + relevant-leaf log
    fs::path perf_dir = ensure_logs_dir();
    const std::string perf_base = std::format("FsTreePerf_FC_{}_{}", dim, n_functions);
    fs::path perf_path = perf_dir / (perf_base + ".txt");
    std::ofstream perf_log(perf_path, std::ios::trunc);
    if (!perf_log) {
        throw std::runtime_error(std::format("Failed to open perf log file: {}", perf_path.string()));
    }
    perf_log << "fi\tfc_sec\trelevant_leaves\n";
    perf_log.flush();

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
        GroupPlan group_plan(n_functions);
        std::unique_ptr<ITreeNode>::pointer current_node = builder.root.get();
        current_node->is_on_path = true;


        // planes is std::vector<Eigen::VectorXd>
        Eigen::VectorXd p = Eigen::VectorXd::Constant(planes.front().size(), 0.1);

        run_grouped_insertion(n_functions, group_plan, builder, root_poly, current_node, p);

        double fc_sec = builder.get_fc_time_sec();
        int relevant_leaves = builder.count_relevant_leaves();
        perf_log << std::format("{}\t{:.6f}\t{}\n", n_functions, fc_sec, relevant_leaves);
        perf_log.flush();


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
        std::println("Relevant Leaves: {}", relevant_leaves);

        // Also write final result to log
        log << "=== Final Results ===\n";
        log << std::format("Time:         {:.4f} s\n", total_time);
        log << std::format("Total Nodes:  {}\n", total_nodes_final);
        log << std::format("Leaf Cells:   {}\n", leaf_cells_final);
        log << std::format("Tree Depth:   {}\n", depth_final);
        log << std::format("Relevant Leaves: {}", relevant_leaves);
        log.flush();

        std::println("[Log] PERF: {}", perf_path.string());

    } catch (const std::exception& e) {
        std::println("Error: {}", e.what());
        return 1;
    }

    return 0;
}
