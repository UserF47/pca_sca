#include <print>
#include <vector>
#include <string>
#include <format>
#include <stdexcept>
#include <chrono>
#include <Eigen/Dense>

#include <fstream>
#include <filesystem>

// --- Project Headers ---
#include "CompactIO.h"
#include "PolytopeStructs.h"
#include "PolytopeOps.h"
#include "ITreePolyDomain.h"

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

        namespace fs = std::filesystem;

        auto ensure_logs_dir = []() -> fs::path {
            fs::path logs_dir = "logs";
            std::error_code ec;
            fs::create_directories(logs_dir, ec);
            return logs_dir;
        };

        fs::path log_dir = ensure_logs_dir();

        // FC log (checkpoint every 50 processed planes)
        const std::string fc_base = std::format("ITreePolyDomainStoragePerf_FC_{}_{}", dim, n_planes);
        fs::path fc_path = log_dir / (fc_base + ".txt");
        std::ofstream fc_log(fc_path);

        // Storage log (checkpoint every 50 processed planes)
        const std::string storage_ckpt_base =
            std::format("ITreePolyDomainStoragePerf_Storage_{}_{}", dim, n_planes);
        fs::path storage_ckpt_path = log_dir / (storage_ckpt_base + ".txt");
        std::ofstream storage_ckpt_log(storage_ckpt_path);

        // Scale log (10..50 step 5; hard stop 55)
        const std::string scale_base = std::format("ITreePolyDomainStoragePerf_Scale_{}_{}", dim, n_planes);
        fs::path scale_path = log_dir / (scale_base + ".txt");
        std::ofstream scale_log(scale_path);

        auto start_time = t2;
        std::chrono::minutes next_checkpoint(10);
        const std::chrono::minutes checkpoint_step(5);
        const std::chrono::minutes last_checkpoint(50);
        const std::chrono::minutes hard_stop(55);

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
            builder.insert_plane(planes[i], unique_h_id);
            // builder.insert_plane_single_path(planes[i], unique_h_id);

            // FC checkpoint every 50 processed input planes
            const int processed = (int)i + 1;
            if (processed % 50 == 0) {
                fc_log << std::format("{}\t{:.6f}\n",
                                      processed,
                                      builder.get_fc_time_sec());
                fc_log.flush();

                const std::size_t vertex_bytes =
                    builder.total_vertices_storage_bytes();

                storage_ckpt_log << std::format("{}\t{}\n",
                                                processed,
                                                vertex_bytes);
                storage_ckpt_log.flush();
            }

            // Scale checkpoints
            auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(
                std::chrono::high_resolution_clock::now() - start_time);

            while (elapsed >= next_checkpoint && next_checkpoint <= last_checkpoint) {
                scale_log << std::format("{}\t{}\n", next_checkpoint.count(), processed);
                scale_log.flush();
                next_checkpoint += checkpoint_step;
            }

            // Hard stop at 55 minutes
            if (elapsed >= hard_stop) {
                scale_log << std::format("{}\t{}\n", hard_stop.count(), processed);
                scale_log.flush();

                // ensure final FC line exists
                if (processed % 50 != 0) {
                    fc_log << std::format("{}\t{:.6f}\n", processed, builder.get_fc_time_sec());
                    fc_log.flush();
                }

                std::println("\n[Scale] Reached {} minutes. Processed {} planes. Exiting.", hard_stop.count(), processed);
                std::println("[Log] FC: {}", fc_path.string());
                std::println("[Log] SCALE: {}", scale_path.string());
                return 0;
            }
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

        const int total_processed = (int)planes.size();
        if (total_processed % 50 != 0) {
            fc_log << std::format("{}\t{:.6f}\n",
                                  total_processed,
                                  builder.get_fc_time_sec());
        }
        fc_log.close();
        scale_log.close();
        storage_ckpt_log.close();

        // ---------------------------------------------------------
        // Vertex Storage Log (PolyDomain)
        // ---------------------------------------------------------
        const std::string storage_base =
            std::format("ITreePolyDomainStorage_{}_{}", dim, n_planes);
        fs::path storage_path = log_dir / (storage_base + ".txt");

        std::ofstream storage_log(storage_path);
        const std::size_t vertex_bytes =
            builder.total_vertices_storage_bytes();

        storage_log << std::format(
            "Total Vertices Storage: {} bytes ({:.3f} MiB)\n",
            vertex_bytes,
            (double)vertex_bytes / (1024.0 * 1024.0)
        );
        storage_log.close();

        std::println("[Log] STORAGE: {}", storage_path.string());

        std::println("[Log] FC: {}", fc_path.string());
        std::println("[Log] SCALE: {}", scale_path.string());
        std::println("[Log] STORAGE-CHECKPOINT: {}", storage_ckpt_path.string());

    } catch (const std::exception& e) {
        std::println("Error: {}", e.what());
        return 1;
    }

    return 0;
}