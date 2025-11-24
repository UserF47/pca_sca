#pragma once

// 1. Include the Data Structures
#include "PolytopeStructs.h"

// 2. Include Standard Libraries needed for logic/printing
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <map>
#include <algorithm>

// Note the 'inline' keyword!
inline Polytope create_hypercube(int dim) {
    Polytope cube;
    cube.dim = dim;
    size_t num_vertices = 1ULL << dim;

    for (size_t i = 0; i < num_vertices; ++i) {
        Eigen::VectorXd position(dim);
        std::vector<int> plane_ids;
        plane_ids.reserve(dim);

        for (int k = 0; k < dim; ++k) {
            bool is_upper_bound = (i >> k) & 1;
            if (is_upper_bound) {
                position[k] = 1.0;
                plane_ids.push_back(-(2 * k + 2));
            } else {
                position[k] = 0.0;
                plane_ids.push_back(-(2 * k + 1));
            }
        }
        cube.add_vertex(position, plane_ids);
    }

    int edge_id_counter = 0;
    for (int i = 0; i < num_vertices; ++i) {
        for (int k = 0; k < dim; ++k) {
            int neighbor_idx = i ^ (1 << k);
            if (i < neighbor_idx) {
                Edge e{edge_id_counter++, static_cast<int>(i), neighbor_idx};
                cube.edges.push_back(e);
            }
        }
    }
    return cube;
}

inline void print_polytope(const Polytope& poly) {
    if (poly.vertices.empty()) {
        std::cout << "Empty Polytope\n";
        return;
    }

    std::cout << "=== Polytope Dump (Dim: " << poly.vertices[0].position.size() << ") ===\n";
    std::cout << "Vertices: " << poly.vertices.size() << " | Edges: " << poly.edges.size() << "\n\n";

    std::cout << "--- Vertices ---\n";
    for (const auto& v : poly.vertices) {
        std::cout << "V[" << std::setw(2) << v.id << "] Coords: (";
        for (int k = 0; k < v.position.size(); ++k) {
            std::cout << std::fixed << std::setprecision(2) << v.position[k];
            if (k < v.position.size() - 1) std::cout << ", ";
        }
        std::cout << ")";

        std::cout << " | Constraints: { ";
        if (poly.constraints.count(v.id)) {
            const auto& c_list = poly.constraints.at(v.id);
            for (size_t k = 0; k < c_list.size(); ++k) {
                std::cout << c_list[k];
                if (k < c_list.size() - 1) std::cout << ", ";
            }
        }
        std::cout << " }\n";
    }

    std::cout << "\n--- Edges ---\n";
    for (const auto& e : poly.edges) {
        std::cout << "E[" << std::setw(2) << e.id << "]: "
                  << "V" << e.v1 << " <---> V" << e.v2 << "\n";
    }
    std::cout << "========================================\n";
}


// --- Helper: Generate Combinations ---
// Generates all subsets of size k from the input vector
// We use this to generate the keys for the hashmap
inline void generate_combinations_recursive(
    const std::vector<int>& input,
    std::vector<int>& current,
    int start_index,
    int k,
    std::vector<std::vector<int>>& result)
{
    if (k == 0) {
        result.push_back(current);
        return;
    }
    for (int i = start_index; i <= (int)input.size() - k; ++i) {
        current.push_back(input[i]);
        generate_combinations_recursive(input, current, i + 1, k - 1, result);
        current.pop_back();
    }
}

inline std::vector<std::vector<int>> get_combinations(const std::vector<int>& input, int k) {
    std::vector<std::vector<int>> result;
    std::vector<int> current;
    current.reserve(k);
    generate_combinations_recursive(input, current, 0, k, result);
    return result;
}

inline Polytope slice_polytope(const Polytope& P, const Eigen::VectorXd& H, int h_id) {
    Polytope result_poly;
    result_poly.dim = P.dim;

    // Helper to track which ORIGINAL vertices have been added to the result
    // Key: Old Vertex ID, Value: New Vertex ID in result_poly
    std::unordered_map<int, int> old_to_new_map;

    double eps = 1e-9;

    // --- PHASE 1: Geometry (Find Vertices) ---
    for (const auto& edge : P.edges) {
        const Vertex& v1 = P.vertices[edge.v1];
        const Vertex& v2 = P.vertices[edge.v2];

        double d1 = v1.position.dot(H); // Add bias here if needed
        double d2 = v2.position.dot(H);

        // CASE A: Vertex 1 lies exactly on the plane
        if (std::abs(d1) < eps) {
            if (old_to_new_map.find(v1.id) == old_to_new_map.end()) {
                // Copy the vertex, but ADD the new hyperplane constraint
                std::vector<int> new_cons = P.constraints.at(v1.id);
                new_cons.push_back(h_id);
                std::sort(new_cons.begin(), new_cons.end());

                // Add to result
                result_poly.add_vertex(v1.position, new_cons);
                old_to_new_map[v1.id] = result_poly.vertices.back().id;
            }
        }

        // CASE B: Vertex 2 lies exactly on the plane
        // (Note: We check both independently because an edge might lie entirely on the plane)
        if (std::abs(d2) < eps) {
            if (old_to_new_map.find(v2.id) == old_to_new_map.end()) {
                std::vector<int> new_cons = P.constraints.at(v2.id);
                new_cons.push_back(h_id);
                std::sort(new_cons.begin(), new_cons.end());

                result_poly.add_vertex(v2.position, new_cons);
                old_to_new_map[v2.id] = result_poly.vertices.back().id;
            }
        }

        // CASE C: Strict Crossing (One positive, one negative)
        // We only calculate intersection if NEITHER is on the plane (to avoid duplicates with A/B)
        if (d1 > eps && d2 < -eps || d1 < -eps && d2 > eps) {

            double t = d1 / (d1 - d2);
            Eigen::VectorXd new_pos = v1.position - t * (v1.position - v2.position);

            // Merge constraints from v1 and v2
            std::vector<int> new_constraints;
            const auto& c1 = P.constraints.at(v1.id);
            const auto& c2 = P.constraints.at(v2.id);

            std::set_intersection(
                c1.begin(), c1.end(),
                c2.begin(), c2.end(),
                std::back_inserter(new_constraints)
            );
            new_constraints.push_back(h_id);
            std::sort(new_constraints.begin(), new_constraints.end());

            result_poly.add_vertex(new_pos, new_constraints);
        }
    }

    // --- PHASE 2: Topology (Reconnect Edges) ---
    // This remains exactly the same as before. Your streaming logic handles
    // the reused vertices perfectly because they now include the 'h_id' constraint.

    std::map<std::vector<int>, int> pending_partners;
    int edge_id_counter = 0;
    int d = P.dim;

    for (const auto& v : result_poly.vertices) {
        const std::vector<int>& c_ids = result_poly.constraints.at(v.id);

        // Important: Old vertices on the plane might have > d constraints.
        // Generating all combinations of (d-1) ensures they connect
        // to both their "old" neighbors (if on plane) and "new" neighbors.
        std::vector<std::vector<int>> keys = get_combinations(c_ids, d - 1);

        for (const auto& key : keys) {
            auto it = pending_partners.find(key);
            if (it != pending_partners.end()) {
                int partner_id = it->second;
                // Avoid self-loops (rare degenerate case)
                if (partner_id != v.id) {
                    Edge new_edge{edge_id_counter++, partner_id, v.id};
                    result_poly.edges.push_back(new_edge);
                    pending_partners.erase(it);
                }
            } else {
                pending_partners[key] = v.id;
            }
        }
    }

    return result_poly;
}


// Returns {P_positive, P_negative}
inline std::pair<Polytope, Polytope> split_polytope(const Polytope& P, const Eigen::VectorXd& H, int h_id) {
    Polytope p_pos, p_neg;
    p_pos.dim = P.dim;
    p_neg.dim = P.dim;

    double eps = 1e-9;

    // We need to store new intersection vertices temporarily to add them to both polytopes
    // Structure: Position, Constraints
    struct TempVertex {
        Eigen::VectorXd pos;
        std::vector<int> cons;
    };
    std::vector<TempVertex> new_intersections;

    // Track which ORIGINAL vertices go where
    // 0 = on plane (both), 1 = pos, -1 = neg
    std::vector<int> v_side(P.vertices.size());

    // --- PHASE 1: Classify & Collect Original Vertices ---
    for (const auto& v : P.vertices) {
        double d = v.position.dot(H);

        if (d > eps) {
            v_side[v.id] = 1;
            p_pos.add_vertex(v.position, P.constraints.at(v.id));
        } else if (d < -eps) {
            v_side[v.id] = -1;
            p_neg.add_vertex(v.position, P.constraints.at(v.id));
        } else {
            // Exactly on plane -> Add to BOTH, and append new constraint
            v_side[v.id] = 0;

            std::vector<int> new_cons = P.constraints.at(v.id);
            // Check if h_id exists to avoid duplicates (optional but good)
            bool has_id = false;
            for(int c : new_cons) if(c == h_id) has_id = true;
            if(!has_id) {
                new_cons.push_back(h_id);
                std::sort(new_cons.begin(), new_cons.end());
            }

            p_pos.add_vertex(v.position, new_cons);
            p_neg.add_vertex(v.position, new_cons);
        }
    }

    // --- PHASE 2: Find Intersections (The Cut) ---
    // We iterate edges. If an edge crosses, we generate ONE intersection
    // and add it to both result polytopes.
    for (const auto& edge : P.edges) {
        int s1 = v_side[edge.v1];
        int s2 = v_side[edge.v2];

        // Strict Crossing check: One is 1, one is -1.
        // (If one is 0, the edge touches the plane but doesn't "cross" in a way that generates a NEW vertex)
        if ((s1 == 1 && s2 == -1) || (s1 == -1 && s2 == 1)) {
            const Vertex& v1 = P.vertices[edge.v1];
            const Vertex& v2 = P.vertices[edge.v2];

            double d1 = v1.position.dot(H);
            double d2 = v2.position.dot(H);

            double t = d1 / (d1 - d2);
            Eigen::VectorXd new_pos = v1.position - t * (v1.position - v2.position);

            // Merge constraints
            std::vector<int> new_constraints;
            const auto& c1 = P.constraints.at(v1.id);
            const auto& c2 = P.constraints.at(v2.id);
            std::set_intersection(c1.begin(), c1.end(), c2.begin(), c2.end(), std::back_inserter(new_constraints));
            new_constraints.push_back(h_id);
            std::sort(new_constraints.begin(), new_constraints.end());

            // Add to BOTH
            p_pos.add_vertex(new_pos, new_constraints);
            p_neg.add_vertex(new_pos, new_constraints);
        }
    }

    // --- PHASE 3: Rebuild Topology (Edges) ---
    // Helper lambda to avoid code duplication for p_pos and p_neg
    auto rebuild_edges = [&](Polytope& poly) {
        std::map<std::vector<int>, int> pending;
        int edge_cnt = 0;
        int d = poly.dim;

        for (const auto& v : poly.vertices) {
            const std::vector<int>& c_ids = poly.constraints.at(v.id);
            std::vector<std::vector<int>> keys = get_combinations(c_ids, d - 1);

            for (const auto& key : keys) {
                auto it = pending.find(key);
                if (it != pending.end()) {
                    int partner = it->second;
                    if (partner != v.id) {
                        poly.edges.push_back({edge_cnt++, partner, v.id});
                        pending.erase(it);
                    }
                } else {
                    pending[key] = v.id;
                }
            }
        }
    };

    rebuild_edges(p_pos);
    rebuild_edges(p_neg);

    return {p_pos, p_neg};
}