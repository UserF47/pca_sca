#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <filesystem>
#include <format>
#include <string>
#include <iomanip>
#include <algorithm>
#include <chrono>

struct EvalResult {
    uint64_t id;
    double value;
};

static bool parse_point(int argc, char** argv, uint64_t dim, std::vector<double>& p_out) {
    if (argc == 4) {
        double p = std::stod(argv[3]);
        p_out.assign(dim, p);
        return true;
    }
    if (argc == static_cast<int>(3 + dim)) {
        p_out.resize(dim);
        for (uint64_t i = 0; i < dim; ++i) p_out[i] = std::stod(argv[3 + i]);
        return true;
    }
    return false;
}

static double eval_dot(const std::vector<int8_t>& coeffs,
                       uint64_t func_id,
                       uint64_t dim,
                       const std::vector<double>& p) {
    const size_t base = static_cast<size_t>(func_id) * dim;
    double sum = 0.0;
    for (uint64_t j = 0; j < dim; ++j) {
        sum += static_cast<int>(coeffs[base + j]) * p[j];
    }
    return sum;
}

static void print_function_row(const std::vector<int8_t>& coeffs,
                               uint64_t id,
                               uint64_t dim) {
    const size_t base = static_cast<size_t>(id) * dim;
    std::cout << "(";
    for (uint64_t j = 0; j < dim; ++j) {
        std::cout << static_cast<int>(coeffs[base + j]);
        if (j + 1 < dim) std::cout << ", ";
    }
    std::cout << ")";
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " <n> <dim> <p>\n"
                  << "  " << argv[0] << " <n> <dim> <p0> ... <p{dim-1}>\n";
        return 1;
    }

    const uint64_t n   = std::stoull(argv[1]);
    const uint64_t dim = std::stoull(argv[2]);

    std::vector<double> p;
    if (!parse_point(argc, argv, dim, p)) {
        std::cerr << "Invalid point input\n";
        return 1;
    }

    const std::string func_file =
        std::format("{}_functions_{}d.bin", n, dim);

    std::ifstream fin(func_file, std::ios::binary);
    if (!fin) {
        std::cerr << "Failed to open file: " << func_file << "\n";
        return 1;
    }

    // ---- Read header ----
    uint64_t file_n = 0, file_dim = 0;
    fin.read(reinterpret_cast<char*>(&file_n), sizeof(uint64_t));
    fin.read(reinterpret_cast<char*>(&file_dim), sizeof(uint64_t));

    // ---- Read coefficients ----
    const size_t total_coeffs = static_cast<size_t>(file_n) * file_dim;
    std::vector<int8_t> coeffs(total_coeffs);
    fin.read(reinterpret_cast<char*>(coeffs.data()),
             static_cast<std::streamsize>(total_coeffs));

    // ================= TIMED SECTION =================
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();

    // ---- Evaluate ----
    std::vector<EvalResult> results;
    results.reserve(file_n);
    for (uint64_t i = 0; i < file_n; ++i) {
        results.push_back({i, eval_dot(coeffs, i, file_dim, p)});
    }

    // ---- Sort ----
    std::stable_sort(results.begin(), results.end(),
                     [](const EvalResult& a, const EvalResult& b) {
                         return a.value < b.value;
                     });

    auto t1 = Clock::now();
    // =================================================

    double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    double elapsed_sec = elapsed_ms / 1000.0;

    // ---- Output ----
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "Point p = [";
    for (uint64_t i = 0; i < dim; ++i) {
        std::cout << p[i];
        if (i + 1 < dim) std::cout << ", ";
    }
    std::cout << "]\n\n";

    // std::cout << "Sorted by f(p) ascending:\n";
    // std::cout << "rank\tid\tf(p)\tA\n";
    // for (size_t r = 0; r < results.size(); ++r) {
    //     const auto& e = results[r];
    //     std::cout << r << "\t" << e.id << "\t" << e.value << "\t";
    //     print_function_row(coeffs, e.id, file_dim);
    //     std::cout << "\n";
    // }

    std::cout << "\nRunning time (compute + sort): "
              << elapsed_ms << " ms ("
              << elapsed_sec << " s)\n";

    return 0;
}