#pragma once
#include "PolytopeStructs.h"
#include <cmath>
#include <print>
#include <map>

class UnitCubeGenerator {
public:
    // reserves_planes: How many bits to allocate in ConstraintSet
    // centered: If true, generates [-1, 1]^d. If false, [0, 1]^d.
    static Polytope make(int dim, size_t reserved_planes, bool centered = false) {
        Polytope cube;
        cube.dim = dim;
        int num_verts = 1 << dim; // 2^d

        // 1. Vertices
        cube.vertices.resize(num_verts);
        for (int i = 0; i < num_verts; ++i) {
            cube.vertices[i].id = i;
            cube.vertices[i].position = Eigen::VectorXd::Zero(dim);
            cube.vertices[i].constraints.resize(reserved_planes);

            for (int b = 0; b < dim; ++b) {
                // Base value 0 or 1
                double val = ((i >> b) & 1) ? 1.0 : 0.0;

                // Center adjustment: Map [0,1] -> [-1,1]
                if (centered) {
                    val = (val * 2.0) - 1.0;
                }

                cube.vertices[i].position[b] = val;
            }
        }

        // 2. Edges (Hamming distance = 1)
        std::map<std::pair<int, int>, int> edge_lookup;
        int edge_counter = 0;

        for (int i = 0; i < num_verts; ++i) {
            for (int b = 0; b < dim; ++b) {
                int neighbor = i ^ (1 << b);
                if (i < neighbor) {
                    Edge e;
                    e.id = edge_counter;
                    e.v1 = i;
                    e.v2 = neighbor;
                    cube.edges.push_back(e);
                    edge_lookup[{i, neighbor}] = edge_counter;
                    edge_counter++;
                }
            }
        }

        // 3. Faces (2D Squares)
        int face_counter = 0;
        for (int d1 = 0; d1 < dim; ++d1) {
            for (int d2 = d1 + 1; d2 < dim; ++d2) {
                for (int v = 0; v < num_verts; ++v) {
                    bool d1_is_zero = !((v >> d1) & 1);
                    bool d2_is_zero = !((v >> d2) & 1);

                    if (d1_is_zero && d2_is_zero) {
                        int v00 = v;
                        int v10 = v | (1 << d1);
                        int v11 = v | (1 << d1) | (1 << d2);
                        int v01 = v | (1 << d2);

                        Face f;
                        f.id = face_counter++;

                        int e1 = edge_lookup[{v00, v10}];
                        int e2 = edge_lookup[{v10, v11}];
                        int e3 = edge_lookup[{std::min(v01, v11), std::max(v01, v11)}];
                        int e4 = edge_lookup[{v00, v01}];

                        f.edge_ids = {e1, e2, e3, e4};
                        cube.faces.push_back(f);

                        cube.edges[e1].face_ids.push_back(f.id);
                        cube.edges[e2].face_ids.push_back(f.id);
                        cube.edges[e3].face_ids.push_back(f.id);
                        cube.edges[e4].face_ids.push_back(f.id);
                    }
                }
            }
        }

        return cube;
    }
};