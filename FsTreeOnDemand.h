#pragma once
#include <memory>
#include <vector>
#include <print>
#include <queue>
#include <chrono>
#include <Eigen/Dense>
#include "PolytopeStructs.h"
#include "PolytopeOps.h"

#include "FunctionPairGenerator.h"

struct LinearConstraint {
    Eigen::VectorXd normal;
    double rhs;
};

// ---------------------------------------------------------
// Group plan: Fi -> list of plane_index (pair_index)
// ---------------------------------------------------------
struct GroupPlan {
    std::vector<std::vector<int>> fi_to_planes; // 1-based indexing

    explicit GroupPlan(int n_functions) {
        fi_to_planes.resize(n_functions + 1);
        for (int fi = 1; fi <= n_functions; ++fi) {
            auto& v = fi_to_planes[fi];
            // Default (current behavior): consecutive Fi-related pairs (fi, fj) for fj in (fi+1..n]
            v.reserve(std::max(0, n_functions - fi));
            for (int fj = fi + 1; fj <= n_functions; ++fj) {
                const std::size_t idx = Generator::pair_index(fi, fj, n_functions);
                v.push_back(static_cast<int>(idx));
            }
        }
    }

    const std::vector<int>& get(int fi) const {
        return fi_to_planes[fi];
    }
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
    // bool is_leaf() const { return splitting_plane_id == -1; }
    bool is_leaf() const { return left == nullptr && right == nullptr; }
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
    // FC time for FsTree: classify_polytope_against_plane(...) + split_polytope(...)
    // Accumulated across the whole run (all inserted planes).
    std::chrono::nanoseconds fc_time_ns{0};

    double get_fc_time_sec() const {
        return static_cast<double>(fc_time_ns.count()) / 1e9;
    }

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
                                  int h_id,
                                  int fi,
                                  std::size_t n_functions,
                                  const Eigen::VectorXd& p) {

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

        while (!stack.empty()) {
            InsertionJob job = std::move(stack.back());
            stack.pop_back();

            ITreeNode* current_node = job.node;

            // === CASE 1: Node is a relevant leaf for some Fj ===
            if (current_node->is_relevant_leaf) {
                int tree_target_function = current_node->target_function_id;
                std::size_t plane_index = Generator::pair_index(tree_target_function, fi, n_functions);

                const Eigen::VectorXd& h_pair = (*global_planes)[plane_index];

                // Use the current hyperplane h_vec and the sample point P
                const Eigen::VectorXd& P = current_node->sample_point;
                double val = h_pair.dot(P); // h(P)

                // This node is becoming internal
                // current_node->splitting_plane_id = h_id;
                // clear_relevance(*current_node); // remove sample + target function from internal node

                // Decide which child receives this fragment (Fi vs Fj relation;
                // using sign of h(P) as a proxy for f_i - f_j > 0 vs < 0).
                if (val < 0.0) {
                    // Insert into LEFT subtree
                    stack.push_back({current_node->left.get(), std::move(job.fragment)});
                } else if (val > 0.0) {
                    // Insert into RIGHT subtree
                    stack.push_back({current_node->right.get(), std::move(job.fragment)});
                } else {
                    continue;
                }
                continue;
            }

            // === CASE 2: Pure leaf with no relevance yet ===
            if (current_node->is_leaf()) {
                // Attach the new splitting plane here
                current_node->splitting_plane_id = h_id;
                current_node->sample_point.resize(0);

                current_node->target_function_id = fi;

                current_node->left  = std::make_unique<ITreeNode>();
                current_node->right = std::make_unique<ITreeNode>();

                // Pick a sample point from the current fragment for this group
                // and assign it to both children as "relevant leaves"
                if (!job.fragment.vertices.empty()) {
                    const Vertex& v0 = job.fragment.vertices.front();
                    Eigen::VectorXd P(v0.position.size());
                    for (int d = 0; d < P.size(); ++d) {
                        P[d] = v0.position[d];
                    }

                    current_node->left->sample_point = P;
                    current_node->left->target_function_id = fi;

                    current_node->right->sample_point = P;
                    current_node->right->target_function_id = fi;
                }

                // No need to push children onto the stack for this plane:
                // we've fully propagated this hyperplane at this node.
                continue;
            }

            // === CASE 3: Internal node -> route using point p against existing plane ===
            int existing_id = current_node->splitting_plane_id;
            const Eigen::VectorXd& h_existing = (*global_planes)[existing_id - 1];
            const double val_p = h_existing.dot(p);
            if (val_p < 0.0) {
                // Route only to LEFT child
                if (current_node->left) {
                    stack.push_back({current_node->left.get(), std::move(job.fragment)});
                }
                continue;
            } else if (val_p > 0.0) {
                // Route only to RIGHT child
                if (current_node->right) {
                    stack.push_back({current_node->right.get(), std::move(job.fragment)});
                }
                continue;
            }
            // If val_p == 0, fall back to exact classification/splitting below.

            using FCClock = std::chrono::high_resolution_clock;

            auto fc0 = FCClock::now();
            int cls = classify_polytope_against_plane(job.fragment, h_existing);
            auto fc1 = FCClock::now();
            fc_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(fc1 - fc0);

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
            auto sp0 = FCClock::now();
            auto split_result = split_polytope(job.fragment, h_existing, existing_id);
            auto sp1 = FCClock::now();
            fc_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(sp1 - sp0);
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

    // Count leaf nodes whose is_relevant_leaf == true (ITERATIVE to avoid stack overflow)
    int count_relevant_leaves() const {
        if (!root) return 0;

        int cnt = 0;
        std::vector<const ITreeNode*> st;
        st.push_back(root.get());

        while (!st.empty()) {
            const ITreeNode* node = st.back();
            st.pop_back();
            if (!node) continue;
            if (node->is_relevant_leaf) {cnt += 1;}
            if (node->is_leaf()) {
                // if (node->is_relevant_leaf) cnt += 1;
                continue;
            }

            if (node->left)  st.push_back(node->left.get());
            if (node->right) st.push_back(node->right.get());
        }
        return cnt;
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

    void mark_all_leaves_relevant(int fi) {
        if (!root) return;
        std::vector<ITreeNode*> st;
        st.push_back(root.get());

        while (!st.empty()) {
            ITreeNode* node = st.back();
            st.pop_back();
            if (!node) continue;

            // Internal node: keep traversing
            if (!node->is_leaf()) {
                if (node->right) st.push_back(node->right.get());
                if (node->left)  st.push_back(node->left.get());
                continue;
            }

            // Leaf node: set relevance if not already relevant
            // Only mark leaves that already belong to Fi.
            if (!node->is_relevant_leaf && node->target_function_id == fi) {
                node->is_relevant_leaf = true;

                if (!node->left)  node->left  = std::make_unique<ITreeNode>();
                if (!node->right) node->right = std::make_unique<ITreeNode>();
            }
        }
    }

    void mark_all_leaves_relevant_from(ITreeNode* start_node, int fi) {
        if (!start_node) return;

        std::vector<ITreeNode*> st;
        st.push_back(start_node);

        while (!st.empty()) {
            ITreeNode* node = st.back();
            st.pop_back();
            if (!node) continue;

            // Internal node: keep traversing
            if (!node->is_leaf()) {
                if (node->right) st.push_back(node->right.get());
                if (node->left)  st.push_back(node->left.get());
                continue;
            }

            // Leaf node: set relevance if not already relevant
            // Only mark leaves that already belong to Fi
            if (!node->is_relevant_leaf && node->target_function_id == fi) {
                node->is_relevant_leaf = true;

                if (!node->left)  node->left  = std::make_unique<ITreeNode>();
                if (!node->right) node->right = std::make_unique<ITreeNode>();
            }
        }
    }
};

static void run_grouped_insertion(int n_functions,
                                 const GroupPlan& group_plan,
                                 ITreeBuilder& builder,
                                 const Polytope& root_poly,
                                 std::unique_ptr<ITreeNode>::pointer current_node,
                                 const Eigen::VectorXd& p)
{
    if (!builder.global_planes) {
        throw std::runtime_error("builder.global_planes is null");
    }
    const auto& planes = *builder.global_planes;

    // Insert in groups: {F1-related}, {F2-related}, ..., {Fn-related}
    for (int fi = 1; fi <= n_functions; ++fi) {
        builder.current_function_id = fi; // Fi is 1-based

        for (int plane_index_i : group_plan.get(fi)) {
            const std::size_t plane_index = static_cast<std::size_t>(plane_index_i);
            const int unique_h_id = static_cast<int>(plane_index) + 1;

            // Skip if this plane does NOT partition the root polytope
            const int cls = classify_polytope_against_plane(root_poly, planes[plane_index]);
            if (cls != 2) {
                continue;
            }

            // Group-aware insertion
            builder.insert_dfs_non_recursive(root_poly, planes[plane_index], unique_h_id, fi, n_functions, p);
        }

        // After finishing the Fi-group, mark all current leaf nodes as Fi-relevant if they are not relevant yet.
        builder.mark_all_leaves_relevant_from(current_node, fi);
    }
}
