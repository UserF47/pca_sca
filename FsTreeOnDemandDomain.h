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

#include <unordered_set>   // for highlight set
#include <unordered_map>   // for DOT id mapping
#include <cstdlib>         // for std::system

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

    Polytope input_poly;

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
    std::chrono::nanoseconds fc_time_ns_2{0};

    double get_fc_time_sec() const {
        return static_cast<double>(fc_time_ns.count()) / 1e9;
    }

    double get_fc_time_sec_2() const {
        return static_cast<double>(fc_time_ns_2.count()) / 1e9;
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
                                  std::size_t n_functions)
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
                continue;
            }

            if (node->is_leaf()) {
                node->target_function_id = fi;

                // std::println("[insert_dfs_non_recursive] Split polytope");
                //
                // std::println("[insert_dfs_non_recursive] poly");
                // print_polytope(node->poly);
                // std::println("[insert_dfs_non_recursive] input domain");
                // print_polytope(node->input_poly);

                int cls3 = classify_vertices_against_plane(node->vertices, h_vec);
                // std::println("[insert_dfs_non_recursive] cls3 {}", cls3);

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
                node->left->target_function_id = fi;

                node->right->vertices.clear();
                node->right->vertices.reserve(node->right->poly.vertices.size());
                for (const auto& v : node->right->poly.vertices) {
                    node->right->vertices.push_back(v.position);
                }
                node->right->target_function_id = fi;

                // Drop the parent polytope to save memory; we keep only its vertices.
                node->poly = Polytope{};

                // print_polytope(node->left->poly);
                // print_polytope(node->right->poly);

                auto sp2 = clock::now();
                auto split_res = split_polytope(node->input_poly, h_vec, h_id);
                auto sp3 = clock::now();
                fc_time_ns_2 += std::chrono::duration_cast<std::chrono::nanoseconds>(sp3 - sp2);
                // int cls2 = classify_polytope_against_plane_v2(node->input_poly, h_vec);

                // std::println("[insert_dfs_non_recursive] cls2 {}", cls2);
                //
                // Polytope& p1 = split_res.first;   // H(x) >= 0 side
                // Polytope& p2 = split_res.second;  // H(x) <= 0 side
                //
                // print_polytope(p1);
                // print_polytope(p2);
                //
                // node->left->is_on_path = true;
                // node->right->is_on_path = true;
                //
                // node->left->input_poly  = std::move(p2);
                // node->right->input_poly = std::move(p1);
                //
                // std::println("[insert_dfs_non_recursive] End");
                // continue;

                int cls2 = classify_polytope_against_plane_v2(node->input_poly, h_vec);

                if (cls2 == -1) {
                    node->left->is_on_path = true;
                    node->right->group_plan[fi].push_back(h_id);

                    node->left->input_poly = node->input_poly;
                } else if (cls2 == 1) {
                    node->right->is_on_path = true;
                    node->left->group_plan[fi].push_back(h_id);

                    node->right->input_poly = node->input_poly;
                } else {
                    node->left->is_on_path = true;
                    node->right->is_on_path = true;

                    auto split_res = split_polytope(node->input_poly, h_vec, h_id);

                    Polytope& p1 = split_res.first;   // H(x) >= 0 side
                    Polytope& p2 = split_res.second;  // H(x) <= 0 side

                    // print_polytope(p1);
                    // print_polytope(p2);

                    node->left->input_poly  = std::move(p2);
                    node->right->input_poly = std::move(p1);
                }
            }
            else {
                if (node->left && node->left->is_on_path) {
                    // Route only to LEFT child
                    stack.push_back({node->left.get()});
                    node->right->group_plan[fi].push_back(h_id);
                }
                if (node->right && node->right->is_on_path) {
                    // Route only to RIGHT child
                    stack.push_back({node->right.get()});
                    node->left->group_plan[fi].push_back(h_id);
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

    // Count nodes (leaf OR internal) whose is_relevant_leaf == true (ITERATIVE to avoid stack overflow)
    int count_relevant_nodes() const {
        if (!root) return 0;

        int cnt = 0;
        std::vector<const ITreeNode*> st;
        st.push_back(root.get());

        while (!st.empty()) {
            const ITreeNode* node = st.back();
            st.pop_back();
            if (!node) continue;

            if (node->is_relevant_leaf) {
                cnt += 1;
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

    void mark_all_leaves_relevant_from(ITreeNode* start_node, int fi, int n_functions) {
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
            // Only mark leaves that already belong to Fi and are on-path.
            if (!node->is_relevant_leaf && node->target_function_id == fi && node->is_on_path == true) {

                // Relevant-leaf routing in insert_dfs_non_recursive uses node->sample_point.
                // Pick a deterministic point (a cached vertex) if it's missing.
                if (node->sample_point.size() == 0) {
                    if (!node->vertices.empty()) {
                        node->sample_point = node->vertices.front();
                    } else if (!node->poly.vertices.empty()) {
                        node->sample_point = node->poly.vertices.front().position;
                    } else {
                        // Degenerate leaf: cannot route without a point.
                        continue;
                    }
                }

                node->is_relevant_leaf = true;

                // Ensure children exist (insert_dfs_non_recursive assumes left/right non-null for relevant leaves)
                if (!node->left)  node->left  = std::make_unique<ITreeNode>();
                if (!node->right) node->right = std::make_unique<ITreeNode>();

                // Init/copy group plans (copy if parent already has one)
                if (!node->group_plan.empty()) {
                    node->left->group_plan  = node->group_plan;
                    node->right->group_plan = node->group_plan;
                } else {
                    node->left->group_plan.assign(n_functions + 1, {});
                    node->right->group_plan.assign(n_functions + 1, {});
                }

                // IMPORTANT: do NOT std::move(node->poly) twice.
                // Give both children the same polytope. (Assumes Polytope is copyable.)
                node->left->poly  = node->poly;
                node->right->poly = node->poly;

                node->left->input_poly  = node->input_poly;
                node->right->input_poly = node->input_poly;

                // print_polytope(node->left->input_poly);
                // print_polytope(node->right->input_poly);

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

                // Optional memory saving: parent becomes a routing gate now
                node->poly = Polytope{};
            }
        }
    }

    // Reset is_on_path flags for the whole tree (or a subtree) so a new query point can be used.
    void reset_on_path_flags(ITreeNode* start = nullptr) {
        ITreeNode* s = start ? start : root.get();
        if (!s) return;

        std::vector<ITreeNode*> st;
        st.push_back(s);
        while (!st.empty()) {
            ITreeNode* node = st.back();
            st.pop_back();
            if (!node) continue;

            node->is_on_path = false;

            if (node->left)  st.push_back(node->left.get());
            if (node->right) st.push_back(node->right.get());
        }
    }

    // Return ALL leaf nodes reached by point p.
    // If node->is_relevant_leaf == true, expand BOTH children (no routing).
    // Only leaf nodes are appended to the result.
    std::vector<ITreeNode*> find_leaves_by_point(const Eigen::VectorXd& p,
                                                 ITreeNode* start = nullptr,
                                                 bool verbose = false) const
    {
        std::vector<ITreeNode*> leaves;

        ITreeNode* s = start ? start : root.get();
        if (!s) return leaves;

        if (!global_planes) {
            // Without planes, we cannot route; only return if start is already a leaf.
            if (s->is_leaf()) leaves.push_back(s);
            return leaves;
        }

        std::vector<ITreeNode*> st;
        st.push_back(s);

        while (!st.empty()) {
            ITreeNode* node = st.back();
            st.pop_back();
            if (!node) continue;

            // Rule: only append leaves
            if (node->is_leaf()) {
                if (verbose) {
                    std::println("[find_leaves_by_point] leaf ptr={}", static_cast<const void*>(node));
                }
                leaves.push_back(node);
                continue;
            }

            // Relevant: expand both children
            if (node->is_relevant_leaf) {
                if (verbose) {
                    std::println("[find_leaves_by_point] relevant ptr={} -> expand both",
                                 static_cast<const void*>(node));
                }
                if (node->left)  st.push_back(node->left.get());
                if (node->right) st.push_back(node->right.get());
                // If children are missing, we stop this branch (do not append).
                continue;
            }

            // Normal internal: route by splitting plane
            const int hid = node->splitting_plane_id;
            if (hid <= 0) {
                if (verbose) {
                    std::println("[find_leaves_by_point] ptr={} internal but hid={} -> stop branch",
                                 static_cast<const void*>(node), hid);
                }
                continue; // not a leaf => do not append
            }

            const Eigen::VectorXd& h = (*global_planes)[hid - 1];
            const double val = h.dot(p);

            if (verbose) {
                std::println("[find_leaves_by_point] ptr={} hid={} val={:.6f}",
                             static_cast<const void*>(node), hid, val);
            }

            if (val < 0.0) {
                if (node->left) st.push_back(node->left.get());
            } else {
                if (node->right) st.push_back(node->right.get());
            }
        }

        return leaves;
    }

    void export_tree_to_dot(const std::string& dot_path,
                            const ITreeNode* start,
                            const std::unordered_set<const ITreeNode*>& highlight) const
    {
        const ITreeNode* s = start ? start : root.get();
        std::ofstream out(dot_path, std::ios::trunc);
        if (!out) throw std::runtime_error(std::format("Failed to open DOT file: {}", dot_path));

        out << "digraph FsTree {\n";
        out << "  rankdir=TB;\n";
        out << "  node [shape=box, fontname=\"Helvetica\", fontsize=10];\n";
        out << "  edge [fontname=\"Helvetica\", fontsize=9];\n";
        if (!s) { out << "}\n"; return; }

        std::unordered_map<const ITreeNode*, int> id;
        id.reserve(1024);

        auto get_id = [&id](const ITreeNode* n) -> int {
            auto it = id.find(n);
            if (it != id.end()) return it->second;
            int new_id = (int)id.size();
            id.emplace(n, new_id);
            return new_id;
        };

        std::vector<const ITreeNode*> st{ s };
        while (!st.empty()) {
            const ITreeNode* n = st.back();
            st.pop_back();
            if (!n) continue;

            const int nid = get_id(n);

            const bool hi = highlight.contains(n);
            out << "  n" << nid << " [label=\""
                << "fi=" << n->target_function_id << "\\n"
                << "rel=" << (n->is_relevant_leaf ? 1 : 0) << "\\n"
                << "path=" << (n->is_on_path ? 1 : 0)
                << "\"";

            if (hi) {
                out << ", style=filled, fillcolor=\"yellow\"";
            }
            out << "];\n";

            if (n->left) {
                const ITreeNode* L = n->left.get();
                out << "  n" << nid << " -> n" << get_id(L) << " [label=\"L\"];\n";
                st.push_back(L);
            }
            if (n->right) {
                const ITreeNode* R = n->right.get();
                out << "  n" << nid << " -> n" << get_id(R) << " [label=\"R\"];\n";
                st.push_back(R);
            }
        }

        out << "}\n";
    }

    // Export to PDF by invoking Graphviz `dot`. Returns true on success.
    // This will create an intermediate DOT file next to the PDF.
    bool export_tree_to_pdf(const std::string& pdf_path,
                            const ITreeNode* start,
                            const std::unordered_set<const ITreeNode*>& highlight) const
    {
        const std::string dot_path = pdf_path + ".dot";
        export_tree_to_dot(dot_path, start, highlight);
        const std::string cmd = std::format("dot -Tpdf \"{}\" -o \"{}\"", dot_path, pdf_path);
        return std::system(cmd.c_str()) == 0;
    }

};

static void run_grouped_insertion(int n_functions,
                                 const GroupPlan& group_plan,
                                 ITreeBuilder& builder,
                                 const Polytope& root_poly,
                                 std::unique_ptr<ITreeNode>::pointer current_node)
{
    if (!builder.global_planes) {
        throw std::runtime_error("builder.global_planes is null");
    }

    const auto& planes = *builder.global_planes;

    // Insert in groups: {F1-related}, {F2-related}, ..., {Fn-related}
    for (int fi = 1; fi <= n_functions; ++fi) {
        builder.current_function_id = fi; // Fi is 1-based

        // NEW: skip empty Fi-group
        if (group_plan.get(fi).empty()) {
            continue;
        }

        for (int plane_index_i : group_plan.get(fi)) {
            const std::size_t plane_index = static_cast<std::size_t>(plane_index_i);
            const int unique_h_id = static_cast<int>(plane_index) + 1;

            // Skip if this plane does NOT partition the root polytope
            const int cls = classify_polytope_against_plane(root_poly, planes[plane_index]);
            if (cls != 2) {
                continue;
            }

            // Group-aware insertion
            builder.insert_dfs_non_recursive(current_node, planes[plane_index], unique_h_id, fi, n_functions);
        }

        const int depth_now = builder.compute_depth(current_node);
        // std::println("[FsTreeOnDemand] After Fi = {}, tree depth = {}", fi, depth_now);
        const int rel_nodes = builder.count_relevant_nodes();
        // std::println("[FsTreeOnDemand] After Fi = {}, count_relevant_nodes= {}", fi, rel_nodes);
        // After finishing the Fi-group, mark all current leaf nodes as Fi-relevant if they are not relevant yet.
        builder.mark_all_leaves_relevant_from(current_node, fi, n_functions);
    }
}
