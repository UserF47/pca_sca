#pragma once
#include <memory>
#include <vector>
#include <print>
#include <queue>
#include <Eigen/Dense>
#include "PolytopeStructs.h"
#include "PolytopeOps.h"

struct ITreeNode {
    int splitting_plane_id = -1;
    std::vector<int> stored_planes;

    std::unique_ptr<ITreeNode> left;
    std::unique_ptr<ITreeNode> right;

    bool is_leaf() const { return splitting_plane_id == -1; }
};

struct InsertionJob {
    ITreeNode* node;
    Polytope fragment;
};

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

        // UPDATE 1: Filter out empty or single-point (corner touch) intersections
        if (initial_fragment.vertices.size() < 2) return;

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
                current_node->stored_planes.push_back(h_id);

                current_node->left = std::make_unique<ITreeNode>();
                current_node->right = std::make_unique<ITreeNode>();

                continue;
            }

            // CASE B: Internal Node -> Filter down
            int existing_id = current_node->splitting_plane_id;
            const Eigen::VectorXd& h_existing = (*global_planes)[existing_id - 1];

            auto split_result = split_polytope(job.fragment, h_existing, existing_id);
            Polytope& p_pos = split_result.first;
            Polytope& p_neg = split_result.second;

            // UPDATE 2: Only propagate if we have a valid geometric fragment (>= 2 vertices)

            // Negative (Left) Child
            if (p_neg.vertices.size() >= 2) {
                q.push({current_node->left.get(), std::move(p_neg)});
            }

            // Positive (Right) Child
            if (p_pos.vertices.size() >= 2) {
                q.push({current_node->right.get(), std::move(p_pos)});
            }
        }
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
};