#pragma once
#include <memory>
#include <vector>
#include <queue>
#include <Eigen/Dense>
#include "DDMSolver.h"


struct LPTreeNode {
    int splitting_plane_id = -1; // Index into LPTreeBuilder::m_history
    std::unique_ptr<LPTreeNode> left;
    std::unique_ptr<LPTreeNode> right;

    // Half-space constraints describing the cell at this node (normal · x <= rhs).
    std::vector<LinearConstraint> constraints;

    // Vertices of the cell associated with this node (optional: may be empty if not computed yet).
    std::vector<Eigen::VectorXd> vertices;

    bool is_leaf() const { return splitting_plane_id == -1; }
};


class LPTreeBuilder {
private:
    std::vector<Eigen::VectorXd> m_history;
    int m_dim;


public:
    std::unique_ptr<LPTreeNode> root;

    LPTreeBuilder(int dim) : m_dim(dim) {
        root = std::make_unique<LPTreeNode>();

        // 1. Initialize root constraint set: 0 <= x_i <= 1 for all i.
        // We encode each as normal^T x <= rhs:
        //   x_i <= 1      ->  e_i^T x <= 1
        //   x_i >= 0      -> -e_i^T x <= 0
        root->constraints.clear();
        root->constraints.reserve(2 * m_dim);

        for (int i = 0; i < m_dim; ++i) {
            Eigen::VectorXd e = Eigen::VectorXd::Zero(m_dim);
            e(i) = 1.0;

            // x_i <= 1
            root->constraints.push_back(LinearConstraint{e, 1.0});
            // -x_i <= 0  (i.e., x_i >= 0)
            root->constraints.push_back(LinearConstraint{-e, 0.0});
        }

        // 2. Compute the vertex set of the root cell using DDM and store it.
        root->vertices = computeVerticesWithDDM(root->constraints);
    }

    // ---------------------------------------------------------
    // ITERATIVE INSERTION (BFS-Based, Work-Queue with plane_id)
    // ---------------------------------------------------------
    void insert(const Eigen::VectorXd& h_vec) {
        // 1. Register the new plane in history and get its id
        m_history.push_back(h_vec);
        const int plane_id = static_cast<int>(m_history.size()) - 1;

        // 2. Work item for BFS: node + its constraints
        struct WorkItem {
            LPTreeNode* node;
            std::vector<LinearConstraint> constraints;
        };

        // 3. Initialize queue with root node and root's constraint set
        std::queue<WorkItem> q;
        q.push(WorkItem{root.get(), root->constraints});

        // 4. BFS loop (layer-by-layer)
        while (!q.empty()) {
            WorkItem item = q.front();
            q.pop();

            LPTreeNode* node = item.node;
            std::vector<LinearConstraint>& constraints = item.constraints;
            const Eigen::VectorXd& current_plane = m_history[plane_id];

            // A. Geometric pruning: use vertex-based cut test with DDM.
            int cls = classify_vertices_against_plane(node->vertices, current_plane);
            if (cls != 2) {
                // Either the region is infeasible (no vertices), or the plane does not
                // actually cut this cell (all vertices on one side). In both cases,
                // we stop traversal for this node here.
                continue;
            }

            // B. Logic for leaf vs internal node
            if (node->is_leaf()) {
                // CUTTING A LEAF -> SPLIT
                node->splitting_plane_id = plane_id;
                node->left  = std::make_unique<LPTreeNode>();
                node->right = std::make_unique<LPTreeNode>();

                // For a fresh split at this leaf, we know the current cell is described by `constraints`.
                // Left child: current_plane(x) <= 0
                {
                    std::vector<LinearConstraint> left_constraints = constraints;
                    left_constraints.push_back(LinearConstraint{current_plane, 0.0});
                    node->left->constraints = left_constraints;
                    node->left->vertices = computeVerticesWithDDM(node->left->constraints);
                }

                // Right child: current_plane(x) >= 0  ->  -current_plane(x) <= 0
                {
                    std::vector<LinearConstraint> right_constraints = constraints;
                    right_constraints.push_back(LinearConstraint{-current_plane, 0.0});
                    node->right->constraints = right_constraints;
                    node->right->vertices = computeVerticesWithDDM(node->right->constraints);
                }

                // The new plane is established at this node. No further processing needed
                // for its children during this insertion of this plane.
                continue;
            }

            // CUTTING AN INTERNAL NODE -> enqueue children (next BFS layer)
            int existing_id = node->splitting_plane_id;
            const Eigen::VectorXd& h_existing = m_history[existing_id];

            // LEFT child: h_existing(x) <= 0
            std::vector<LinearConstraint> left_constraints = constraints;
            left_constraints.push_back({h_existing, 0.0});
            q.push(WorkItem{node->left.get(), std::move(left_constraints)});

            // RIGHT child: h_existing(x) >= 0  =>  -h_existing(x) <= 0
            std::vector<LinearConstraint> right_constraints = constraints;
            right_constraints.push_back({-h_existing, 0.0});
            q.push(WorkItem{node->right.get(), std::move(right_constraints)});
        }
    }

    // Statistics
    int count_nodes(const LPTreeNode* node = nullptr) {
        if (!node) node = root.get();
        int c = 1;
        if (node->left) c += count_nodes(node->left.get());
        if (node->right) c += count_nodes(node->right.get());
        return c;
    }

    int count_leaves(const LPTreeNode* node = nullptr) {
        if (!node) node = root.get();
        if (node->is_leaf()) return 1;
        return count_leaves(node->left.get()) + count_leaves(node->right.get());
    }

};