#include <print>
#include <string>
#include <stdexcept>
#include "DataGenerator.h"

// Usage: ./gen_data <count> <dim>
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::println("Usage: {} <count> <dim>", argv[0]);
        std::println("Example: {} 1000000 2", argv[0]);
        return 1;
    }

    try {
        size_t n = std::stoull(argv[1]);
        size_t d = std::stoull(argv[2]);

        if (d == 0) {
            std::println("Error: Dimension must be > 0");
            return 1;
        }

        // Call the real logic from DataGenerator.cpp
        Generator::run(n, d);

    } catch (const std::exception& e) {
        std::println("Error: Invalid arguments. {}", e.what());
        return 1;
    }

    return 0;
}