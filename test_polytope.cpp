#include <iostream>
#include "PolytopeOps.h" // Includes PolytopeStructs.h and print_polytope()

int main() {
    std::cout << "=== 2D Split Test (Diagonal Cut) ===\n";
    std::cout << "Splitting Unit Square with hyperplane x - y = 0\n\n";

    // 1. Create Standard Square [0,1]^2
    Polytope square = create_hypercube(2);

    // 2. Define Hyperplane: x - y = 0
    // Normal: [1, -1]
    // This passes through (0,0) and (1,1), which are existing vertices.
    Eigen::VectorXd H(2);
    H << 1.0, -1.0;

    int h_id = 555;

    // 3. Perform Split
    // Returns a pair {Positive_Polytope, Negative_Polytope}
    auto result = split_polytope(square, H, h_id);
    Polytope& p_pos = result.first;
    Polytope& p_neg = result.second;

    // 4. Print Results
    std::cout << "--- P_Positive (x > y: Bottom-Right) ---\n";
    // Expecting 3 vertices: (0,0), (1,0), (1,1)
    print_polytope(p_pos);
    std::cout << p_pos.vertices.size() << " vertices found.\n";

    std::cout << "\n--- P_Negative (x < y: Top-Left) ---\n";
    // Expecting 3 vertices: (0,0), (0,1), (1,1)
    print_polytope(p_neg);
    std::cout << p_neg.vertices.size() << " vertices found.\n";

    return 0;
}