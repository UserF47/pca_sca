#pragma once
#include <memory>
#include <vector>
#include <print>
#include <queue>
#include <chrono>
#include <Eigen/Dense>
#include "PolytopeStructs.h"
#include "PolytopeOps.h"

struct LinearConstraint {
    Eigen::VectorXd normal;
    double rhs;
};

struct ITreeNode {
    int splitting_plane_id = -1;
    // std::vector<int> stored_planes;

    std::unique_ptr<ITreeNode> left;
    std::unique_ptr<ITreeNode> right;

    bool is_leaf() const { return splitting_plane_id == -1; }
};

struct InsertionJob {
    ITreeNode* node;
    Polytope fragment;
};

// Rough memory usage estimate for a Polytope in bytes.
inline std::size_t estimate_polytope_memory(const Polytope& P) {
    std::size_t bytes = 0;

    // Vertices (shallow estimate: Vertex object only; Eigen storage is dynamic and not fully counted here)
    bytes += P.vertices.size() * sizeof(Vertex);

    // Edges
    bytes += P.edges.size() * sizeof(Edge);

    // Constraint map (key + vector header)
    bytes += P.constraints.size() * (sizeof(int) + sizeof(std::vector<int>));
    for (const auto& kv : P.constraints) {
        bytes += kv.second.size() * sizeof(int);
    }

    return bytes;
}

class ITreeBuilder {
public:
    std::unique_ptr<ITreeNode> root;
    const std::vector<Eigen::VectorXd>* global_planes = nullptr;

    ITreeBuilder(const Polytope& root_domain) {
        root = std::make_unique<ITreeNode>();
    }

    void insert_plane(const Polytope& root_domain, const Eigen::VectorXd& h_vec, int h_id) {

        // 1. Initial Slice
        Polytope initial_fragment = slice_polytope(root_domain, h_vec, h_id);

        // 2. Setup Queue
        std::queue<InsertionJob> q;
        q.push({root.get(), std::move(initial_fragment)});

        // 3. Process Queue
        while (!q.empty()) {
            InsertionJob job = std::move(q.front());
            q.pop();

            ITreeNode* current_node = job.node;

            // CASE A: Leaf -> Split it
            if (current_node->is_leaf()) {
                current_node->splitting_plane_id = h_id;
                // current_node->stored_planes.push_back(h_id);

                current_node->left = std::make_unique<ITreeNode>();
                current_node->right = std::make_unique<ITreeNode>();

                continue;
            }

            // CASE B: Internal Node -> Filter down

            int existing_id = current_node->splitting_plane_id;
            const Eigen::VectorXd& h_existing = (*global_planes)[existing_id - 1];

            // Use vertex-based classifier to decide how this fragment sits relative to the existing plane
            int cls = classify_polytope_against_plane(job.fragment, h_existing);

            if (cls == -1) {
                // All H(v) <= 0 (within tolerance) -> fragment belongs to LEFT side only
                q.push({current_node->left.get(), std::move(job.fragment)});
                continue;
            }

            if (cls == 1) {
                // All H(v) >= 0 (within tolerance) -> fragment belongs to RIGHT side only
                q.push({current_node->right.get(), std::move(job.fragment)});
                continue;
            }

            // 3) True partition case: min_d < -eps && max_d > eps -> need to SPLIT
            auto split_result = split_polytope(job.fragment, h_existing, existing_id);
            Polytope& p_pos = split_result.first;
            Polytope& p_neg = split_result.second;

            q.push({current_node->left.get(), std::move(p_neg)});
            q.push({current_node->right.get(), std::move(p_pos)});

        }
    }

    // DFS (non-recursive) insertion version: memory-efficient alternative to BFS
    void insert_dfs_non_recursive(const Polytope& root_domain,
                                  const Eigen::VectorXd& h_vec,
                                  int h_id) {
        if (!global_planes) {
            throw std::runtime_error("ITreeBuilder::insert_dfs_non_recursive: global_planes is nullptr");
        }

        // 1. Initial slice
        Polytope initial_fragment = slice_polytope(root_domain, h_vec, h_id);
        if (initial_fragment.vertices.empty()) {
            // New plane does not intersect domain
            return;
        }

        // Explicit stack for DFS
        std::vector<InsertionJob> stack;
        // stack.reserve(64);
        stack.push_back({root.get(), std::move(initial_fragment)});

        // Track time for periodic logging
        using Clock = std::chrono::steady_clock;
        auto last_print = Clock::now();

        // DFS loop
        while (!stack.empty()) {
            // --- Debug: print approximate memory usage of the DFS stack every 30 seconds ---
            auto now = Clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_print).count() >= 30) {
                std::size_t stack_bytes = stack.capacity() * sizeof(InsertionJob);
                std::size_t stack_poly_bytes = 0;
                for (const auto& j : stack) {
                    stack_poly_bytes += estimate_polytope_memory(j.fragment);
                }
                std::println("[DFS] stack size = {}, capacity = {}, stack buffer ≈ {:.2f} KB, polys ≈ {:.2f} KB",
                             stack.size(),
                             stack.capacity(),
                             stack_bytes / 1024.0,
                             stack_poly_bytes / 1024.0);
                last_print = now;
            }
            // --------------------------------------------------------------

            InsertionJob job = std::move(stack.back());
            stack.pop_back();

            ITreeNode* current_node = job.node;

            // CASE A: Leaf -> attach new plane
            if (current_node->is_leaf()) {
                current_node->splitting_plane_id = h_id;
                // current_node->stored_planes.push_back(h_id);
                current_node->left  = std::make_unique<ITreeNode>();
                current_node->right = std::make_unique<ITreeNode>();
                continue;
            }

            // CASE B: Internal node -> classify fragment using existing plane
            int existing_id = current_node->splitting_plane_id;
            const Eigen::VectorXd& h_existing = (*global_planes)[existing_id - 1];

            int cls = classify_polytope_against_plane(job.fragment, h_existing);

            if (cls == -1) {
                // Entire fragment on <= side
                if (current_node->left) {
                    stack.push_back({current_node->left.get(), std::move(job.fragment)});
                }
                continue;
            }

            if (cls == 1) {
                // Entire fragment on >= side
                if (current_node->right) {
                    stack.push_back({current_node->right.get(), std::move(job.fragment)});
                }
                continue;
            }

            // CASE C: True split
            auto split_result = split_polytope(job.fragment, h_existing, existing_id);
            Polytope p_pos = std::move(split_result.first);   // H_existing >= 0
            Polytope p_neg = std::move(split_result.second);  // H_existing <= 0

            // Push in DFS order — right then left
            if (current_node->right) {
                stack.push_back({current_node->right.get(), std::move(p_pos)});
            }
            if (current_node->left) {
                stack.push_back({current_node->left.get(), std::move(p_neg)});
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
                // current->stored_planes.push_back(h_id);

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
};