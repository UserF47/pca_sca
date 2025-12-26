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

    bool all_groups_empty() const {
        // fi_to_planes is 1-based; index 0 is unused
        for (std::size_t fi = 1; fi < fi_to_planes.size(); ++fi) {
            if (!fi_to_planes[fi].empty()) {
                return false;
            }
        }
        return true;
    }
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

    if (has_pos && !has_neg) return 1;  // all on the positive side (or on a plane)
    if (has_neg && !has_pos) return -1; // all on the negative side (or on a plane)
    return 0;                            // all (approximately) on the plane
}


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

    bool is_on_path = false;

    // A sample point inside this node's subdomain.
    // Only meaningful when is_relevant_leaf == true.
    Eigen::VectorXd sample_point;

    std::unique_ptr<ITreeNode> left;
    std::unique_ptr<ITreeNode> right;

    Polytope poly;
    std::vector<Eigen::VectorXd> vertices;

    // Per-node group plan storage: group_plan[fi] is a list of plane indices for function Fi.
    // 1-based indexing; index 0 is unused.
    std::vector<std::vector<int>> group_plan;

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
};


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

    // DFS (non-recursive) insertion with group-aware semantics.
    void insert_dfs_non_recursive(std::unique_ptr<ITreeNode>::pointer cur_node,
                                  const Eigen::VectorXd& h_vec,
                                  int h_id,
                                  int fi,
                                  std::size_t n_functions,
                                  const Eigen::VectorXd& p)
    {
        if (!global_planes) {
            throw std::runtime_error("ITreeBuilder::insert_dfs_non_recursive: global_planes is nullptr");
        }

        // Explicit stack for DFS
        std::vector<InsertionJob> stack;
        stack.push_back({cur_node});

        while (!stack.empty()) {
            InsertionJob job = std::move(stack.back());
            stack.pop_back();

            ITreeNode* node = job.node;

            // If there are still no vertices, this cell is degenerate or empty.
            if (node->vertices.empty()) {
                continue;
            }

            // === CASE 1: Node is a relevant leaf for some Fj ===
            if (node->is_relevant_leaf) {
                int tree_target_function = node->target_function_id;
                std::size_t plane_index = Generator::pair_index(tree_target_function, fi, n_functions);

                const Eigen::VectorXd& h_pair = (*global_planes)[plane_index];

                // Use the current hyperplane h_vec and the sample point P
                const Eigen::VectorXd& P = node->sample_point;
                double val = h_pair.dot(P); // h(P)

                // This node is becoming internal
                // current_node->splitting_plane_id = h_id;
                // clear_relevance(*current_node); // remove sample + target function from internal node

                // Decide which child receives this fragment (Fi vs Fj relation;
                // using sign of h(P) as a proxy for f_i - f_j > 0 vs < 0).
                if (val < 0.0) {
                    // Insert into LEFT subtree
                    stack.push_back({node->left.get()});
                } else if (val > 0.0) {
                    // Insert into RIGHT subtree
                    stack.push_back({node->right.get()});
                } else {
                    continue;
                }
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
                node->target_function_id = fi;
                const double val_p = h_vec.dot(p);


                // Leaf that is actually cut by h_vec -> split its polytope.
                auto sp0 = clock::now();
                auto split_result = split_polytope(node->poly, h_vec, h_id);
                auto sp1 = clock::now();
                fc_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(sp1 - sp0);

                Polytope& p_pos = split_result.first;   // H(x) >= 0 side
                Polytope& p_neg = split_result.second;  // H(x) <= 0 side

                node->splitting_plane_id = h_id;

                node->left  = std::make_unique<ITreeNode>();
                node->right = std::make_unique<ITreeNode>();

                node->left->group_plan.assign(n_functions + 1, {});
                node->right->group_plan.assign(n_functions + 1, {});

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

                if (val_p < 0.0) {
                    node->left->is_on_path = true;
                    node->right->group_plan[fi].push_back(h_id);
                } else {
                    node->right->is_on_path = true;
                    node->left->group_plan[fi].push_back(h_id);
                }
            }
            else {
                if (node->left && node->left->is_on_path) {
                    // Route only to LEFT child
                    stack.push_back({node->left.get()});
                }
                if (node->right && node->right->is_on_path) {
                    // Route only to RIGHT child
                    stack.push_back({node->right.get()});
                }
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
            builder.insert_dfs_non_recursive(current_node, planes[plane_index], unique_h_id, fi, n_functions, p);
        }

        // After finishing the Fi-group, mark all current leaf nodes as Fi-relevant if they are not relevant yet.
        builder.mark_all_leaves_relevant_from(current_node, fi);
    }
}
