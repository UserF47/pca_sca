#pragma once
#include <memory>
#include <vector>
#include <queue>
#include <Eigen/Dense>
#include "LPSolver.h" //

// Defines a half-space constraint: normal * x <= rhs
struct LinearConstraint {
    Eigen::VectorXd normal;
    double rhs;
};

struct LPTreeNode {
    int splitting_plane_id = -1; // Index into LPTreeBuilder::m_history
    std::unique_ptr<LPTreeNode> left;
    std::unique_ptr<LPTreeNode> right;
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

        // 3. Initialize queue with root node and empty constraint set
        std::queue<WorkItem> q;
        q.push(WorkItem{root.get(), {}});

        // 4. BFS loop (layer-by-layer)
        while (!q.empty()) {
            WorkItem item = q.front();
            q.pop();

            LPTreeNode* node = item.node;
            std::vector<LinearConstraint>& constraints = item.constraints;
            const Eigen::VectorXd& current_plane = m_history[plane_id];

            // A. Geometric pruning: compute min/max of the plane over this cell
            auto [min_val, max_val] = solve_lp_min_max(constraints, current_plane);

            // If region is empty (infeasible), skip this branch
            if (min_val > max_val) continue;

            // Check if the plane CUTS the cell (min < 0 < max)
            double eps = 1e-9;
            bool cuts = (min_val < -eps) && (max_val > eps);

            if (!cuts) {
                // Plane is strictly positive or negative. It doesn't split this cell.
                // We stop traversal for this node here.
                continue;
            }

            // B. Logic for leaf vs internal node
            if (node->is_leaf()) {
                // CUTTING A LEAF -> SPLIT
                node->splitting_plane_id = plane_id;
                node->left  = std::make_unique<LPTreeNode>();
                node->right = std::make_unique<LPTreeNode>();
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

private:
    std::pair<double, double> solve_lp_min_max(const std::vector<LinearConstraint>& consts, const Eigen::VectorXd& objective) {
        int m = (int)consts.size();
        Eigen::MatrixXd A(m, m_dim);
        Eigen::VectorXd b(m);

        for(int i=0; i<m; ++i) {
            A.row(i) = consts[i].normal;
            b(i) = consts[i].rhs;
        }

        LPResult res_max = LPSolver::solve(A, b, objective);
        double max_val = (res_max.status == LPStatus::Optimal) ? res_max.objective_value : -1e9;

        if (res_max.status != LPStatus::Optimal) return {1.0, -1.0};

        LPResult res_min = LPSolver::solve(A, b, -objective);
        double min_val = (res_min.status == LPStatus::Optimal) ? -res_min.objective_value : 1e9;

        return {min_val, max_val};
    }
};