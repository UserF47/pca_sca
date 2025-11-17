#include <print>
#include <vector>
#include <string>
#include <format>
#include <stdexcept>
#include "CompactIO.h"

// Usage: ./pca_sca <count> <dim>
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::println("Usage: {} <count> <dim>", argv[0]);
        std::println("Example: {} 1000000 2", argv[0]);
        return 1;
    }

    std::string n_str = argv[1];
    std::string d_str = argv[2];

    // 1. Reconstruct filename from arguments
    std::string filename = std::format("{}_hyperplanes_{}d.bin", n_str, d_str);

    std::println(">>> FS-Tree Solver Application");
    std::println("Loading data from '{}'...", filename);

    // 2. Load Data
    try {
        CompactDataset ds = CompactIO::load(filename);
        std::println("SUCCESS: Loaded {} hyperplanes ({}D).", ds.count, ds.dim);

        // --- YOUR RESEARCH CODE GOES HERE ---

        // Iterating and printing the first 100 hyperplanes
        std::println("\n--- Verifying Hyperplane Data (First 100) ---");
        size_t print_limit = std::min((size_t)100, (size_t)ds.count);

        for (size_t i = 0; i < print_limit; ++i) {
            std::print("H[{}]: ", i);

            // Loop through each coefficient (dimension) for this hyperplane
            for (size_t j = 0; j < ds.dim; ++j) {
                // Use .at() helper from CompactIO.h to get the coefficient
                std::print("a{}={} ", j + 1, ds.at(i, j));
            }
            std::println(""); // Newline for the next hyperplane
        }

        if (ds.count > print_limit) {
            std::println("... (and {} more)", ds.count - print_limit);
        }

        // ------------------------------------

        std::println("\nReady for FS-Tree logic.");


    } catch (const std::exception& e) {
        std::println("--- FATAL ERROR ---");
        std::println("Could not load or read data file.");
        std::println("Details: {}", e.what());
        std::println("Please run the generator first:");
        std::println("./gen_data {} {}", n_str, d_str);
        return 1;
    }

    return 0;
}