#include <iostream>
#include <cddlib/setoper.h>
#include <cddlib/cdd.h>

// Note: cddlib uses an abstract number type `mytype`, which is a 1D array.
// We must use dd_set_d(...) to assign values, and read them as x[0].

// Test: unit square in 2D
// 0 <= x <= 1, 0 <= y <= 1
// Inequalities in cddlib row format: [b, a1, a2] meaning a1*x + a2*y <= b

int main() {
    dd_set_global_constants();  // must be called before using cddlib

    dd_ErrorType err = dd_NoError;

    const int m = 4;   // number of inequalities
    const int d = 2;   // dimension

    // Create matrix with m rows and d+1 columns: [b, a1, a2]
    dd_MatrixPtr M = dd_CreateMatrix(m, d + 1);
    M->representation = dd_Inequality;

    // -x <= 0  (x >= 0)
    dd_set_d(M->matrix[0][0], 0.0);
    dd_set_d(M->matrix[0][1], -1.0);
    dd_set_d(M->matrix[0][2],  0.0);

    // -y <= 0  (y >= 0)
    dd_set_d(M->matrix[1][0], 0.0);
    dd_set_d(M->matrix[1][1], 0.0);
    dd_set_d(M->matrix[1][2], -1.0);

    //  x <= 1
    dd_set_d(M->matrix[2][0], 1.0);
    dd_set_d(M->matrix[2][1], 1.0);
    dd_set_d(M->matrix[2][2], 0.0);

    //  y <= 1
    dd_set_d(M->matrix[3][0], 1.0);
    dd_set_d(M->matrix[3][1], 0.0);
    dd_set_d(M->matrix[3][2], 1.0);

    // Run Double Description: H-rep -> Polyhedron
    dd_PolyhedraPtr P = dd_DDMatrix2Poly(M, &err);
    if (err != dd_NoError) {
        std::cerr << "Error in dd_DDMatrix2Poly: " << err << std::endl;
        dd_FreeMatrix(M);
        dd_free_global_constants();
        return 1;
    }

    // Extract generators (vertices/rays)
    dd_MatrixPtr G = dd_CopyGenerators(P);

    std::cout << "Generators (V-representation):\n";
    for (int i = 0; i < G->rowsize; ++i) {
        double type = G->matrix[i][0][0];  // 1 = vertex, 0 = ray
        if (type == 1.0) {
            std::cout << "Vertex: (";
            for (int j = 1; j <= d; ++j) {
                std::cout << G->matrix[i][j][0];
                if (j < d) std::cout << ", ";
            }
            std::cout << ")\n";
        } else {
            std::cout << "Non-vertex generator (type=" << type << ")\n";
        }
    }

    // Cleanup
    dd_FreeMatrix(G);
    dd_FreePolyhedra(P);
    dd_FreeMatrix(M);
    dd_free_global_constants();

    return 0;
}