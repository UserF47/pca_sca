#pragma once

#include <vector>
#include <unordered_map>
#include <Eigen/Dense>

struct Vertex {
    int id;
    Eigen::VectorXd position;
};

struct Edge {
    int id;
    int v1;
    int v2;
};

struct Polytope {
    int dim; // Dimension of the polytope

    std::vector<Vertex> vertices;
    std::vector<Edge> edges;
    std::unordered_map<int, std::vector<int>> constraints;

    void add_vertex(const Eigen::VectorXd& pos, const std::vector<int>& plane_ids) {
        int new_index = vertices.size();
        vertices.push_back({new_index, pos});
        constraints[new_index] = plane_ids;
    }
};