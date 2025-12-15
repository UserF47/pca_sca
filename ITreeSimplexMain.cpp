#include <print>
#include <vector>
#include <string>
#include <format>
#include <stdexcept>
#include <chrono>
#include <Eigen/Dense>

#include "CompactIO.h"
#include "LPTree.h" 

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::println("Usage: {} <count> <dim>", argv[0]);
        return 1;
    }

    int n_planes = std::stoi(argv[1]);
    int dim = std::stoi(argv[2]);
    std::string filename = std::format("{}_hyperplanes_{}d.bin", n_planes, dim);

    std::println("==========================================");
    std::println("   I-Tree Solver (HiGHS LP Engine)        ");
    std::println("==========================================");

    try {
        // 1. Load Data
        auto t0 = std::chrono::high_resolution_clock::now();
        CompactDataset ds = CompactIO::load(filename);
        
        std::vector<Eigen::VectorXd> planes;
        planes.reserve(ds.count);
        for (size_t i = 0; i < ds.count; ++i) {
            Eigen::VectorXd h(ds.dim);
            for (size_t j = 0; j < ds.dim; ++j) h[j] = ds.at(i, j);

            // std::print("Plane {:4d}: [", (int)i + 1);
            // for (int k = 0; k < h.size(); ++k) {
            //     std::print("{}{}", h[k], (k + 1 < h.size() ? ", " : ""));
            // }
            // std::println("]");

            planes.push_back(h);
        }
        std::println("[1] Loaded {} planes", ds.count);

        // 2. Build Tree
        std::println("[2] Building Tree...");
        auto t2 = std::chrono::high_resolution_clock::now();

        LPTreeBuilder builder(dim);

        const int bar_width = 50;

        for (size_t i = 0; i < planes.size(); ++i) {
            int unique_h_id = (int)i + 1; 
            
            if (i % 10 == 0 || i == planes.size() - 1) {
                float progress = (float)(i + 1) / planes.size();
                int pos = (int)(bar_width * progress);
                std::print("\r    Progress: [");
                for (int j = 0; j < bar_width; ++j) {
                    if (j < pos) std::print("=");
                    else if (j == pos) std::print(">");
                    else std::print(" ");
                }
                std::print("] {:.1f}% ({}/{})", progress * 100.0, i + 1, planes.size());
                fflush(stdout);
            }

            builder.insert(planes[i]);
        }
        std::println(""); 

        auto t3 = std::chrono::high_resolution_clock::now();

        // 3. Results
        std::println("\n=== Results ===");
        std::println("Time:         {:.4f} s", std::chrono::duration<double>(t3-t2).count());
        std::println("Total Nodes:  {}", builder.count_nodes());
        std::println("Leaf Cells:   {}", builder.count_leaves());

    } catch (const std::exception& e) {
        std::println("Error: {}", e.what());
        return 1;
    }

    return 0;
}