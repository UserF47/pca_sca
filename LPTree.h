#pragma once
#include <memory>
#include <vector>
#include <stack>
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

// Helper struct for the iterative stack
struct StackItem {
    enum Type { PROCESS_NODE, PUSH_CONSTRAINT, POP_CONSTRAINT } type;
    LPTreeNode* node = nullptr;
    LinearConstraint constraint; // Only used if type == PUSH_CONSTRAINT

    // Constructors for convenience
    static StackItem Process(LPTreeNode* n) {
        return {PROCESS_NODE, n, {}};
    }
    static StackItem Push(const LinearConstraint& c) {
        return {PUSH_CONSTRAINT, nullptr, c};
    }
    static StackItem Pop() {
        return {POP_CONSTRAINT, nullptr, {}};
    }
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
    // ITERATIVE INSERTION (Stack-Based)
    // ---------------------------------------------------------
    void insert(const Eigen::VectorXd& h_vec) {
        // 1. Register the new plane
        m_history.push_back(h_vec);
        int new_id = (int)m_history.size() - 1;

        // --- Print Debug Info ---
        // std::print("Inserting Plane {}: [", new_id);
        // for (int k = 0; k < h_vec.size(); ++k) {
        //     std::print("{:.2f}{}", h_vec[k], (k < h_vec.size() - 1) ? ", " : "");
        // }
        // std::println("]");
        // ------------------------

        // 2. Setup the explicit stack and constraint list
        std::vector<LinearConstraint> constraints;
        std::stack<StackItem> stack;

        // Start by processing the root
        stack.push(StackItem::Process(root.get()));

        // 3. Iterative Loop
        while (!stack.empty()) {
            StackItem item = stack.top();
            stack.pop();

            // --- Handle Constraint Management ---
            if (item.type == StackItem::POP_CONSTRAINT) {
                constraints.pop_back();
                continue;
            }
            if (item.type == StackItem::PUSH_CONSTRAINT) {
                constraints.push_back(item.constraint);
                continue;
            }

            // --- Handle Node Processing (item.type == PROCESS_NODE) ---
            LPTreeNode* node = item.node;

            // A. Geometric Pruning Check
            auto [min_val, max_val] = solve_lp_min_max(constraints, h_vec);

            // If region is empty (infeasible), skip.
            if (min_val > max_val) continue;

            // Check if the plane CUTS the cell (Min < 0 < Max)
            double eps = 1e-9;
            bool cuts = (min_val < -eps) && (max_val > eps);

            if (!cuts) {
                // Plane is strictly positive or negative. It doesn't split this cell.
                // We stop traversal for this branch here.
                continue;
            }

            // B. Logic for Split vs Internal
            if (node->is_leaf()) {
                // CUTTING A LEAF -> SPLIT
                node->splitting_plane_id = new_id;
                node->left = std::make_unique<LPTreeNode>();
                node->right = std::make_unique<LPTreeNode>();
                // The new plane is established. No further recursion needed for this branch.
                continue;
            }

            // CUTTING AN INTERNAL NODE -> TRAVERSE CHILDREN
            // We need to visit Left and Right.
            // Stack is LIFO, so we push the LAST operation first.

            // We want sequence:
            // 1. Push(LeftConstraint) -> Process(Left) -> Pop()
            // 2. Push(RightConstraint) -> Process(Right) -> Pop()

            int existing_id = node->splitting_plane_id;
            const Eigen::VectorXd& h_existing = m_history[existing_id];

            // --- Operations for Right Child (pushed first, processed last) ---
            // Constraint: h_existing >= 0  =>  -h_existing <= 0
            stack.push(StackItem::Pop());
            stack.push(StackItem::Process(node->right.get()));
            stack.push(StackItem::Push({ -h_existing, 0.0 }));

            // --- Operations for Left Child (pushed last, processed first) ---
            // Constraint: h_existing <= 0
            stack.push(StackItem::Pop());
            stack.push(StackItem::Process(node->left.get()));
            stack.push(StackItem::Push({ h_existing, 0.0 }));
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