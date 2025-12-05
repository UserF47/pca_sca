#pragma once

#include <vector>
#include <Eigen/Dense>
#include <cddlib/setoper.h>
#include <cddlib/cdd.h>

// Local epsilon for DDMSolver (do NOT rely on GEOM_EPS)
static constexpr double DDM_EPS = 1e-7;

// Expects LinearConstraint to be defined as:
// struct LinearConstraint {
//     Eigen::VectorXd normal;  // a in a^T x <= rhs
//     double rhs;              // b in a^T x <= b
// };

// Defines a half-space constraint: normal * x <= rhs
struct LinearConstraint {
    Eigen::VectorXd normal;
    double rhs;
};

// NOTE:
// 1. This function assumes dd_set_global_constants() has been called once
//    before the first call, and dd_free_global_constants() will be called
//    once at program shutdown.
// 2. It ignores rays (unbounded directions) and only returns point generators (vertices).
inline std::vector<Eigen::VectorXd>
computeVerticesWithDDM(const std::vector<LinearConstraint>& consts) {
    std::vector<Eigen::VectorXd> vertices;

    if (consts.empty()) {
        return vertices;
    }

    // All constraints must have the same dimension.
    const int dim = static_cast<int>(consts.front().normal.size());
    const int m   = static_cast<int>(consts.size());

    // Basic sanity check: all normals same size.
    for (const auto& c : consts) {
        if (c.normal.size() != dim) {
            throw std::runtime_error("computeVerticesWithDDM: inconsistent constraint dimensions");
        }
    }

    dd_ErrorType err = dd_NoError;

    // cddlib representation:
    // Each row [b, a1, ..., ad] encodes: b + a1*x1 + ... + ad*xd >= 0
    // Our constraints are normal^T x <= rhs.
    // So we rewrite a^T x <= b as: b - a^T x >= 0  =>  b + (-a)^T x >= 0
    dd_MatrixPtr M = dd_CreateMatrix(m, dim + 1);
    M->representation = dd_Inequality;

    for (int i = 0; i < m; ++i) {
        const auto& c = consts[i];

        // b term
        dd_set_d(M->matrix[i][0], c.rhs);

        // a_j = -normal_j
        for (int j = 0; j < dim; ++j) {
            const double aj = -c.normal(j);
            dd_set_d(M->matrix[i][j + 1], aj);
        }
    }

    dd_PolyhedraPtr P = dd_DDMatrix2Poly(M, &err);
    if (err != dd_NoError) {
        dd_FreeMatrix(M);
        throw std::runtime_error("computeVerticesWithDDM: dd_DDMatrix2Poly failed");
    }

    dd_MatrixPtr G = dd_CopyGenerators(P);
    if (!G) {
        dd_FreePolyhedra(P);
        dd_FreeMatrix(M);
        return vertices;
    }

    // G: rowsize x colsize, each entry is mytype (double[1] in floating build)
    // Column 0: generator type (1 = point/vertex, 0 = ray, etc.)
    // Columns 1..dim: coordinates.
    for (int i = 0; i < G->rowsize; ++i) {
        const double type = G->matrix[i][0][0];
        // Keep only vertices (type == 1)
        if (std::abs(type - 1.0) > 1e-9) {
            continue;
        }

        Eigen::VectorXd v(dim);
        for (int j = 0; j < dim; ++j) {
            v(j) = G->matrix[i][j + 1][0];
        }
        vertices.push_back(std::move(v));
    }

    dd_FreeMatrix(G);
    dd_FreePolyhedra(P);
    dd_FreeMatrix(M);

    return vertices;
}

// Classify a set of vertices against a hyperplane H (same convention as above).
// Returns:
//   2  -> vertices lie on both sides of H (intersection)
//   1  -> all vertices are on the positive side (d >  GEOM_EPS)
//  -1  -> all vertices are on the negative side or on the plane (d <= GEOM_EPS)
inline int classify_vertices_against_plane(const std::vector<Eigen::VectorXd>& vertices,
                                           const Eigen::VectorXd& H) {
    bool has_pos = false;
    bool has_neg = false;

    for (const auto& v : vertices) {
        // Optional: sanity check on dimension match in debug builds
        // assert(v.size() == H.size());
        double d = H.dot(v);
        if (d > DDM_EPS) {
            has_pos = true;
        } else if (d < -DDM_EPS) {
            has_neg = true;
        }
        if (has_pos && has_neg) {
            return 2;
        }
    }

    if (has_pos && !has_neg) {
        return 1;
    }
    return -1;
}
