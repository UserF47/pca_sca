#pragma once
#include <memory>
#include <vector>
#include <print>
#include "PolytopeStructs.h"
#include "PolytopeOps.h"

struct ITreeNode {
    int id;
    int splitting_plane_idx = -1;
    std::vector<int> stored_planes; // Leaf storage

    std::unique_ptr<ITreeNode> left;
    std::unique_ptr<ITreeNode> right;

    bool is_leaf() const { return splitting_plane_idx == -1; }
};

class ITreeBuilder {
public:
    int node_counter = 0;
    std::unique_ptr<ITreeNode> root;
    const std::vector<Eigen::VectorXd>* global_planes = nullptr;

    ITreeBuilder() {
        root = std::make_unique<ITreeNode>();
        root->id = node_counter++;
    }

    void insert_plane(const Polytope& root_domain, int h_new_idx, const std::vector<Eigen::VectorXd>& planes) {
        global_planes = &planes;
        auto split_res = PolytopeOps::split(root_domain, planes[h_new_idx], h_new_idx);

        if (!split_res.intersection || split_res.intersection->vertices.empty()) return;

        insert_recursive(root.get(), *split_res.intersection, h_new_idx);
    }

    // Added this Helper
    int count_leaves(const ITreeNode* node) {
        if (!node) return 0;
        if (node->is_leaf()) return 1;
        return count_leaves(node->left.get()) + count_leaves(node->right.get());
    }

private:
    void insert_recursive(ITreeNode* node, const Polytope& P, int h_new_idx) {
        if (node->is_leaf()) {
            node->splitting_plane_idx = h_new_idx;
            node->stored_planes.push_back(h_new_idx);

            node->left = std::make_unique<ITreeNode>();
            node->left->id = node_counter++;

            node->right = std::make_unique<ITreeNode>();
            node->right->id = node_counter++;
            return;
        }

        const Eigen::VectorXd& h_node = (*global_planes)[node->splitting_plane_idx];
        auto result = PolytopeOps::split(P, h_node, node->splitting_plane_idx);

        if (result.neg && !result.neg->vertices.empty()) {
            insert_recursive(node->left.get(), *result.neg, h_new_idx);
        }
        if (result.pos && !result.pos->vertices.empty()) {
            insert_recursive(node->right.get(), *result.pos, h_new_idx);
        }
    }
};