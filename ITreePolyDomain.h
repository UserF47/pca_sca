#pragma once
#include <memory>
#include <vector>
#include <print>
#include <queue>
#include <Eigen/Dense>
#include "PolytopeStructs.h"
#include "PolytopeOps.h"

#include <chrono>

struct LinearConstraint {
    Eigen::VectorXd normal;
    double rhs;
};

inline int classify_vertices_against_plane(const std::vector<Eigen::VectorXd>& vertices,
                                           const Eigen::VectorXd& h_vec,
                                           double eps = 1e-9) {
    bool has_pos = false;
    bool has_neg = false;

    for (const auto& v : vertices) {
        double d = h_vec.dot(v);
        if (d > eps) {
            has_pos = true;
        } else if (d < -eps) {
            has_neg = true;
        }
        if (has_pos && has_neg) {
            return 2;  // plane partitions the polytope
        }
    }

    if (has_pos && !has_neg) return 1;  // all on positive side (or on plane)
    if (has_neg && !has_pos) return -1; // all on negative side (or on plane)
    return 0;                            // all (approximately) on the plane
}

struct ITreeNode {
    int splitting_plane_id = -1;
    std::vector<int> stored_planes;

    // Geometric state associated with this node:
    // - poly: the polytope (cell) corresponding to this node in the partition
    // - vertices: optional cached copy of its vertex set (can be empty if not filled yet)
    Polytope poly;
    std::vector<Eigen::VectorXd> vertices;

    std::unique_ptr<ITreeNode> left;
    std::unique_ptr<ITreeNode> right;

    bool is_leaf() const { return splitting_plane_id == -1; }
};

struct InsertionJob {
    ITreeNode* node;
    Polytope fragment;
};

class ITreeBuilder {
private:
    std::chrono::nanoseconds fc_time_ns{0};

public:
    std::unique_ptr<ITreeNode> root;
    const std::vector<Eigen::VectorXd>* global_planes = nullptr;

    explicit ITreeBuilder(const Polytope& root_domain) {
        root = std::make_unique<ITreeNode>();

        // Initialize root geometric state from the given root polytope.
        // We treat root_domain as the cell associated with the root node.
        root->poly = root_domain;

        // If the Polytope already stores its vertices, copy them.
        // Otherwise, replace this with your own vertex computation routine.
        if (!root_domain.vertices.empty()) {
            // Convert from Polytope::vertices (std::vector<Vertex>) to
            // the node's cached vertex coordinates (std::vector<Eigen::VectorXd>).
            root->vertices.clear();
            root->vertices.reserve(root_domain.vertices.size());
            for (const auto& v : root_domain.vertices) {
                // Adjust 'position' to the actual coordinate member of Vertex if different.
                root->vertices.push_back(v.position);
            }
        } else {
            // e.g., root->vertices = compute_vertices(root_domain);
            root->vertices.clear();
        }
    }

    void insert_plane(const Eigen::VectorXd& h_vec, int h_id) {
        // We traverse the tree cells and see where the NEW plane h_vec actually
        // partitions the current polytope. For each leaf cell that is cut by
        // h_vec, we split its polytope into two children and then drop the
        // parent polytope (keeping only its vertices) to save memory.

        std::queue<ITreeNode*> q;
        q.push(root.get());

        while (!q.empty()) {
            ITreeNode* node = q.front();
            q.pop();

            // Ensure we have a vertex set to classify against this plane.
            if (node->vertices.empty() && !node->poly.vertices.empty()) {
                node->vertices.clear();
                node->vertices.reserve(node->poly.vertices.size());
                for (const auto& v : node->poly.vertices) {
                    // Copy coordinate vector from Vertex into the cached Eigen::VectorXd list.
                    node->vertices.push_back(v.position);
                }
            }

            // If there are still no vertices, this cell is degenerate or empty.
            if (node->vertices.empty()) {
                continue;
            }

            using clock = std::chrono::high_resolution_clock;

            auto fc0 = clock::now();
            int cls = classify_vertices_against_plane(node->vertices, h_vec);
            auto fc1 = clock::now();
            fc_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(fc1 - fc0);

            if (cls != 2) {
                // Plane does not partition this cell; skip its subtree.
                continue;
            }

            if (node->is_leaf()) {
                // Leaf that is actually cut by h_vec -> split its polytope.
                auto sp0 = clock::now();
                auto split_result = split_polytope(node->poly, h_vec, h_id);
                auto sp1 = clock::now();
                fc_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(sp1 - sp0);

                Polytope& p_pos = split_result.first;   // H(x) >= 0 side
                Polytope& p_neg = split_result.second;  // H(x) <= 0 side

                node->splitting_plane_id = h_id;
                node->stored_planes.push_back(h_id);

                node->left  = std::make_unique<ITreeNode>();
                node->right = std::make_unique<ITreeNode>();

                // Assign child polytopes.
                node->left->poly  = std::move(p_neg);
                node->right->poly = std::move(p_pos);

                // Cache child vertices from their polytopes (if available).
                node->left->vertices.clear();
                node->left->vertices.reserve(node->left->poly.vertices.size());
                for (const auto& v : node->left->poly.vertices) {
                    node->left->vertices.push_back(v.position);
                }

                node->right->vertices.clear();
                node->right->vertices.reserve(node->right->poly.vertices.size());
                for (const auto& v : node->right->poly.vertices) {
                    node->right->vertices.push_back(v.position);
                }

                // Drop the parent polytope to save memory; we keep only its vertices.
                node->poly = Polytope{};
            } else {
                // Internal node that is cut by h_vec somewhere inside its cell:
                // continue traversal into both children (if they exist).
                if (node->left) {
                    q.push(node->left.get());
                }
                if (node->right) {
                    q.push(node->right.get());
                }
            }
        }
    }
    // A single-path insertion that follows the unique subdomain containing
    // a fixed point P = [0.5, 0.5, ..., 0.5]. This version is aligned with
    // the full BFS insert logic: it uses polytope slicing/classification, but
    // only descends along one path and accumulates the corresponding
    // inequality set along that path.
    void insert_plane_single_path(const Polytope& root_domain,
                                  const Eigen::VectorXd& h_vec,
                                  int h_id) {
        if (!global_planes) {
            throw std::runtime_error("ITreeBuilder::insert_plane_single_path: global_planes is nullptr");
        }

        // Fixed point P = [0.5, 0.5, ..., 0.5]
        int dim = static_cast<int>(h_vec.size());
        Eigen::VectorXd P = Eigen::VectorXd::Constant(dim, 0.5);

        // 1) Initial slice of the root domain by the NEW plane h_vec
        Polytope fragment = slice_polytope(root_domain, h_vec, h_id);
        if (fragment.vertices.empty()) {
            std::cerr << "[insert_plane_single_path] WARNING: initial slice empty for plane id "
                      << h_id << "\n";
            return;
        }

        // Accumulate inequalities in standard form a^T x <= b along the path
        // that contains P. For this single-path version, we only track
        // inequalities induced by existing split planes (not storing cube
        // bounds here).
        std::vector<LinearConstraint> active_constraints;
        active_constraints.reserve(32);

        ITreeNode* current = root.get();

        while (true) {
            // CASE A: Leaf -> attach the new plane here and stop
            if (current->is_leaf()) {
                // Sanity check: P must satisfy all accumulated inequalities
                if (!point_satisfies_constraints(P, active_constraints)) {
                    std::cerr << "[insert_plane_single_path] ERROR: fixed point P does not "
                                 "satisfy active inequality set for h_id "
                              << h_id << "\n";
                }

                std::cerr << "[insert_plane_single_path] Active inequality count for h_id "
                          << h_id << ": " << active_constraints.size() << "\n";

                // Insert the new splitting plane at this leaf
                current->splitting_plane_id = h_id;
                current->stored_planes.push_back(h_id);

                // Decide which child domain nominally contains P with respect
                // to the NEW plane h_vec.
                double valP_new = h_vec.dot(P);
                if (valP_new < 0.0) {
                    current->left = std::make_unique<ITreeNode>();
                } else if (valP_new > 0.0) {
                    current->right = std::make_unique<ITreeNode>();
                } else {
                    // P lies exactly on the new plane; in this degenerate case
                    // we can create both children or neither. Here we create
                    // both for completeness.
                    current->left  = std::make_unique<ITreeNode>();
                    current->right = std::make_unique<ITreeNode>();
                }

                break;
            }

            // CASE B: Internal node -> follow the side that contains P
            int existing_id = current->splitting_plane_id;
            const Eigen::VectorXd& h_existing = (*global_planes)[existing_id - 1];

            // Classify the current fragment against the existing plane
            int cls = classify_polytope_against_plane(fragment, h_existing);
            double valP = h_existing.dot(P);

            LinearConstraint lc;

            if (cls == -1) {
                // Fragment entirely on H_existing(x) <= 0 side
                lc.normal = h_existing;
                lc.rhs = 0.0;
                active_constraints.push_back(lc);

                // Sanity: P should also be on this side (up to tolerance)
                current = current->left.get();
                // fragment remains unchanged
                continue;
            }

            if (cls == 1) {
                // Fragment entirely on H_existing(x) >= 0 side
                lc.normal = -h_existing; // (-H)(x) <= 0  <=>  H(x) >= 0
                lc.rhs = 0.0;
                active_constraints.push_back(lc);

                current = current->right.get();
                // fragment remains unchanged
                continue;
            }

            // True partition: fragment crosses h_existing, so split it
            auto split_result = split_polytope(fragment, h_existing, existing_id);
            Polytope p_pos = std::move(split_result.first);  // H >= 0
            Polytope p_neg = std::move(split_result.second); // H <= 0

            if (valP <= 0.0) {
                // P is on the H <= 0 side: follow LEFT and keep p_neg
                lc.normal = h_existing;
                lc.rhs = 0.0;
                active_constraints.push_back(lc);

                fragment = std::move(p_neg);
                current = current->left.get();
            } else {
                // P is on the H >= 0 side: follow RIGHT and keep p_pos
                lc.normal = -h_existing; // (-H)(x) <= 0  <=>  H(x) >= 0
                lc.rhs = 0.0;
                active_constraints.push_back(lc);

                fragment = std::move(p_pos);
                current = current->right.get();
            }
        }
    }

    // Check whether a point P satisfies the given inequality set in standard form
    // a^T x <= b for each LinearConstraint.
    bool point_satisfies_constraints(const Eigen::VectorXd& P,
                                     const std::vector<LinearConstraint>& constraints,
                                     double eps = 1e-9) const {
        for (const auto& lc : constraints) {
            double val = lc.normal.dot(P) - lc.rhs; // a^T P - b
            if (val > eps) return false;
        }
        return true;
    }

    int count_nodes(const ITreeNode* node = nullptr) {
        if (!node) node = root.get();
        int c = 1;
        if (node->left) c += count_nodes(node->left.get());
        if (node->right) c += count_nodes(node->right.get());
        return c;
    }

    int count_leaves(const ITreeNode* node = nullptr) {
        if (!node) node = root.get();
        if (node->is_leaf()) return 1;
        return count_leaves(node->left.get()) + count_leaves(node->right.get());
    }

    // Compute maximum depth of the I-Tree
    int compute_depth(const ITreeNode* node = nullptr) const {
        if (!node) node = root.get();
        if (!node) return 0;
        if (!node->left && !node->right) return 1;
        int left_depth  = node->left  ? compute_depth(node->left.get())  : 0;
        int right_depth = node->right ? compute_depth(node->right.get()) : 0;
        return 1 + std::max(left_depth, right_depth);
    }


    double get_fc_time_sec() const {
        return (double)fc_time_ns.count() / 1e9;
    }
};