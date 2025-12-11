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
    // The hyperplane that splits this node's domain.
    // -1 means "no splitting plane yet" (true leaf).
    int splitting_plane_id = -1;

    // Function this leaf is currently "relevant" for (Fi).
    // -1 means "no target function associated".
    int target_function_id = -1;

    // True iff this node is currently a relevant leaf for some Fi
    // AND should hold a sample point.
    bool is_relevant_leaf = false;

    // A sample point inside this node's subdomain.
    // Only meaningful when is_relevant_leaf == true.
    Eigen::VectorXd sample_point;

    std::unique_ptr<ITreeNode> left;
    std::unique_ptr<ITreeNode> right;

    // Leaf = has no splitting plane yet.
    bool is_leaf() const { return splitting_plane_id == -1; }
};

inline void clear_relevance(ITreeNode& node) {
    node.is_relevant_leaf = false;
    node.target_function_id = -1;
    node.sample_point.resize(0); // empty vector
}

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

    // NEW: current function whose group we are inserting.
    int current_function_id = -1;

    ITreeBuilder(const Polytope& root_domain) {
        root = std::make_unique<ITreeNode>();
    }

    // DFS (non-recursive) insertion with group-aware semantics.
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
        stack.push_back({root.get(), std::move(initial_fragment)});

        using Clock = std::chrono::steady_clock;
        auto last_print = Clock::now();

        while (!stack.empty()) {
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

            InsertionJob job = std::move(stack.back());
            stack.pop_back();

            ITreeNode* current_node = job.node;

            // === CASE 1: Node is a relevant leaf for some Fj ===
            if (current_node->is_leaf() && current_node->is_relevant_leaf) {
                // We are in the special case of your description:
                // - N is a relevant leaf with target_function_id = Fj
                // - we are inserting a hyperplane from group Fi (current_function_id)
                // - use the sample point P to decide which side to descend

                // Ensure children exist (empty) as requested
                if (!current_node->left) {
                    current_node->left = std::make_unique<ITreeNode>();
                }
                if (!current_node->right) {
                    current_node->right = std::make_unique<ITreeNode>();
                }

                // Use the current hyperplane h_vec and the sample point P
                const Eigen::VectorXd& P = current_node->sample_point;
                double val = h_vec.dot(P); // h(P)

                // This node is becoming internal
                current_node->splitting_plane_id = h_id;
                clear_relevance(*current_node); // remove sample + target function from internal node

                // Decide which child receives this fragment (Fi vs Fj relation;
                // using sign of h(P) as a proxy for f_i - f_j > 0 vs < 0).
                if (val > 0.0) {
                    // Insert into LEFT subtree
                    stack.push_back({current_node->left.get(), std::move(job.fragment)});
                } else if (val < 0.0) {
                    // Insert into RIGHT subtree
                    stack.push_back({current_node->right.get(), std::move(job.fragment)});
                } else {
                    // Degenerate: P lies exactly on the hyperplane, push fragment to both sides
                    stack.push_back({current_node->left.get(), job.fragment});
                    stack.push_back({current_node->right.get(), job.fragment});
                }
                continue;
            }

            // === CASE 2: Pure leaf with no relevance yet ===
            if (current_node->is_leaf()) {
                // Attach the new splitting plane here
                current_node->splitting_plane_id = h_id;
                clear_relevance(*current_node); // if any stale info

                // Split the fragment by this new plane
                auto split_result = split_polytope(job.fragment, h_vec, h_id);
                Polytope p_pos = std::move(split_result.first);   // H >= 0
                Polytope p_neg = std::move(split_result.second);  // H <= 0

                current_node->left  = std::make_unique<ITreeNode>();
                current_node->right = std::make_unique<ITreeNode>();

                // Pick a sample point from the current fragment for this group
                // and assign it to both children as "relevant leaves"
                if (!job.fragment.vertices.empty() && current_function_id != -1) {
                    const Vertex& v0 = job.fragment.vertices.front();
                    Eigen::VectorXd P(v0.position.size());
                    for (int d = 0; d < P.size(); ++d) {
                        P[d] = v0.position[d];
                    }

                    current_node->left->is_relevant_leaf = true;
                    current_node->left->target_function_id = current_function_id;
                    current_node->left->sample_point = P;

                    current_node->right->is_relevant_leaf = true;
                    current_node->right->target_function_id = current_function_id;
                    current_node->right->sample_point = P;
                }

                // No need to push children onto the stack for this plane:
                // we've fully propagated this hyperplane at this node.
                continue;
            }

            // === CASE 3: Internal node -> classify fragment using existing plane ===
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

            // CASE 3C: True split
            auto split_result = split_polytope(job.fragment, h_existing, existing_id);
            Polytope p_pos = std::move(split_result.first);   // H >= 0
            Polytope p_neg = std::move(split_result.second);  // H <= 0

            if (current_node->right) {
                stack.push_back({current_node->right.get(), std::move(p_pos)});
            }
            if (current_node->left) {
                stack.push_back({current_node->left.get(), std::move(p_neg)});
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