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
#include "ITreePloyCA.h"

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

    namespace fs = std::filesystem;

    auto ensure_logs_dir = []() -> fs::path {
        fs::path logs_dir = "logs";
        std::error_code ec;
        fs::create_directories(logs_dir, ec);
        return logs_dir;
    };

    // Existing log file: result_{dim}
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
        std::println("[1] Loaded {} planes ({:.4f}s)", ds.count, std::chrono::duration<double>(t1 - t0).count());

        // ---------------------------------------------------------
        // 2. Initialize Root Domain
        // ---------------------------------------------------------
        Polytope root_poly = create_hypercube(dim);
        std::println("[2] Initialized Root Domain ([0,1]^{})", dim);

        // ---------------------------------------------------------
        // 3. Build Tree (Iterative)
        // ---------------------------------------------------------
        std::println("[3] Building Tree...");
        auto t2 = std::chrono::high_resolution_clock::now();

        ITreeBuilder builder(root_poly);
        builder.global_planes = &planes;

        // Performance logs (same structure as Simplex)
        fs::path log_dir = ensure_logs_dir();
        const std::string fc_base = std::format("ITreePolyCAPerf_FC_Leaf_{}_{}", dim, n_planes);
        const std::string scale_base = std::format("ITreePolyCAPerf_Scale__Leaf{}_{}", dim, n_planes);
        fs::path fc_path = log_dir / (fc_base + ".txt");
        fs::path scale_path = log_dir / (scale_base + ".txt");
        std::ofstream fc_log(fc_path);
        std::ofstream scale_log(scale_path);
        // Count only actually-inserted planes (cls==2 and inserted)
        int processed = 0;

        // Scale checkpoints at 10..50 minutes (step 5), hard stop at 55 minutes
        auto start_time = t2;
        std::chrono::minutes next_checkpoint(10);
        const std::chrono::minutes checkpoint_step(5);
        const std::chrono::minutes last_checkpoint(50);
        const std::chrono::minutes hard_stop(55);

        const int bar_width = 50;

        for (size_t i = 0; i < planes.size(); ++i) {
            int unique_h_id = (int)i + 1;

            if (i % 10 == 0 || i == planes.size() - 1) {
                float progress = (float)(i + 1) / (float)planes.size();
                int pos = (int)(bar_width * progress);

                std::print("\r    Progress: [");
                for (int j = 0; j < bar_width; ++j) {
                    if (j < pos) std::print("=");
                    else if (j == pos) std::print(">");
                    else std::print(" ");
                }
                std::print("] {:.1f}% ({}/{})", progress * 100.0f, (int)i + 1, (int)planes.size());
                fflush(stdout);
            }

            // Skip if this plane does NOT partition the root polytope
            int cls = classify_polytope_against_plane(root_poly, planes[i]);
            if (cls != 2) {
                continue;
            }

            // Insert plane (counts as one processed insertion)
            builder.insert_dfs_non_recursive(root_poly, planes[i], unique_h_id);
            processed += 1;

            // FC checkpoint every 50 processed insertions: "<processed>\t<fc_time_sec>\t<leaf_nodes>"
            if (processed % 5 == 0) {
                const auto leaf_now = builder.count_leaves();
                fc_log << std::format("{}\t{:.6f}\t{}\n",
                                      processed,
                                      builder.get_fc_time_sec(),
                                      leaf_now);
                fc_log.flush();
            }

            // Scale checkpoints based on elapsed minutes
            auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(
                std::chrono::high_resolution_clock::now() - start_time);

            while (elapsed >= next_checkpoint && next_checkpoint <= last_checkpoint) {
                scale_log << std::format("{}\t{}\n", next_checkpoint.count(), processed);
                scale_log.flush();
                next_checkpoint += checkpoint_step;
            }

            // Hard stop at 55 minutes: (DISABLED)
            /*
            if (elapsed >= hard_stop) {
                scale_log << std::format("{}\t{}\n", hard_stop.count(), processed);
                scale_log.flush();

                // Ensure final FC line exists even if not on a multiple of 50
                if (processed % 50 != 0) {
                    fc_log << std::format("{}\t{:.6f}\n", processed, builder.get_fc_time_sec());
                    fc_log.flush();
                }

                std::println("\n[Scale] Reached {} minutes. Processed {} inserted planes. Exiting.", hard_stop.count(), processed);
                std::println("[Log] FC: {}", fc_path.string());
                std::println("[Log] SCALE: {}", scale_path.string());

                return 0;
            }
            */

            // Keep your periodic tree stats logging (every 1000 raw planes seen)
            if ((i + 1) % 1000 == 0) {
                auto t_now = std::chrono::high_resolution_clock::now();
                double elapsed_sec = std::chrono::duration<double>(t_now - t2).count();

                auto total_nodes = builder.count_nodes();
                auto leaf_cells  = builder.count_leaves();
                auto depth       = builder.compute_depth();

                std::println("\n\n=== Results after {} raw planes ===", (int)i + 1);
                std::println("Time:         {:.4f} s", elapsed_sec);
                std::println("Total Nodes:  {}", total_nodes);
                std::println("Leaf Cells:   {}", leaf_cells);
                std::println("Tree Depth:   {}", depth);

                log << std::format("=== Results after {} raw planes ===\n", (int)i + 1);
                log << std::format("Time:         {:.4f} s\n", elapsed_sec);
                log << std::format("Total Nodes:  {}\n", total_nodes);
                log << std::format("Leaf Cells:   {}\n", leaf_cells);
                log << std::format("Tree Depth:   {}\n\n", depth);
                log.flush();
            }
        }

        std::println("");

        // Final FC checkpoint if we ended not on a multiple of 50
        if (processed > 0 && processed % 50 != 0) {
            const auto leaf_now = builder.count_leaves();
            fc_log << std::format("{}\t{:.6f}\t{}\n",
                                  processed,
                                  builder.get_fc_time_sec(),
                                  leaf_now);
            fc_log.flush();
        }
        fc_log.close();
        scale_log.close();

        std::println("[Log] FC: {}", fc_path.string());
        std::println("[Log] SCALE: {}", scale_path.string());

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