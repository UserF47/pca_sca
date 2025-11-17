#include "DataGenerator.h"
#include "CompactIO.h"
#include <random>
#include <string>
#include <format>
#include <print>

namespace Generator {

    void run(size_t n, size_t dim) {
        // 1. Dynamic Filename (e.g., "1000000_hyperplanes_2d.bin")
        std::string filename = std::format("{}_hyperplanes_{}d.bin", n, dim);

        std::println(">>> Running Data Generator");
        std::println("Target Count: {} hyperplanes", n);
        std::println("Target Dim:   {}D", dim);

        // 2. Setup Random Generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(-100, 100);

        CompactDataset ds;
        ds.count = n;
        ds.dim = dim;
        ds.data.reserve(n * dim); // Reserve memory upfront

        // 3. Generate Loop
        size_t count = 0;
        while (count < n) {
            bool all_zero = true;
            std::vector<int8_t> plane_coeffs;
            plane_coeffs.reserve(dim);

            // Generate one hyperplane
            for (size_t d = 0; d < dim; ++d) {
                int val = distrib(gen);
                if (val != 0) all_zero = false;
                plane_coeffs.push_back(static_cast<int8_t>(val));
            }

            // Constraint: Skip if the normal vector is [0, 0, ..., 0]
            if (all_zero) {
                continue;
            }

            // Add the valid plane to our main data vector
            ds.data.insert(ds.data.end(), plane_coeffs.begin(), plane_coeffs.end());
            count++;
        }

        // 4. Save to disk
        try {
            CompactIO::save(filename, ds);
            std::println("SUCCESS: Saved to '{}'", filename);
        } catch (const std::exception& e) {
            std::println("Error saving file: {}", e.what());
        }
    }
}