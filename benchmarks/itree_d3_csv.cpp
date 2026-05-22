#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>
#include <queue>
#include <set>
#include <map>
#include "PolytopeOps.h"
#include "SphericalBucketing_v4.h"

using namespace std;
using clock_type = chrono::high_resolution_clock;

const string OUTPUT_DIR = "/mnt/c/Users/sahil/Documents/DrCaiAFRL";

struct LinearFunction {
    Eigen::VectorXd a;
    double b;
    
    double eval(const Eigen::VectorXd& x) const {
        return a.dot(x) + b;
    }
};

Eigen::Vector3d cross3d(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    Eigen::Vector3d result;
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
    return result;
}

struct Facet3D {
    int v1_idx, v2_idx;
    Eigen::VectorXd normal;
    double angle;
};

struct BucketIndex3D {
    vector<Facet3D> sorted_facets;
    vector<double> boundaries;
};

struct RoutingPolytope3D {
    vector<Eigen::VectorXd> vertices;
    vector<pair<int,int>> edges;
    Eigen::VectorXd hyperplane_normal;
    double hyperplane_offset;
    BucketIndex3D bucket_index;
};

Polytope create_prism(int n) {
    Polytope poly;
    poly.dim = 3;
    
    for (int i = 0; i < n; ++i) {
        double angle = 2.0 * M_PI * i / n;
        Vertex v;
        v.id = i;
        v.position = Eigen::VectorXd(3);
        v.position[0] = cos(angle);
        v.position[1] = sin(angle);
        v.position[2] = 0.0;
        poly.vertices.push_back(v);
    }
    
    for (int i = 0; i < n; ++i) {
        double angle = 2.0 * M_PI * i / n;
        Vertex v;
        v.id = n + i;
        v.position = Eigen::VectorXd(3);
        v.position[0] = cos(angle);
        v.position[1] = sin(angle);
        v.position[2] = 1.0;
        poly.vertices.push_back(v);
    }
    
    return poly;
}

Eigen::VectorXd computeCentroid3D(const vector<Eigen::VectorXd>& vertices) {
    Eigen::VectorXd centroid = Eigen::VectorXd::Zero(3);
    for (const auto& v : vertices) {
        centroid += v;
    }
    return centroid / vertices.size();
}

void buildBucketIndex3D(RoutingPolytope3D& P) {
    if (P.vertices.size() < 3 || P.edges.empty()) return;
    
    Eigen::VectorXd centroid = computeCentroid3D(P.vertices);
    
    for (const auto& [i, j] : P.edges) {
        Facet3D facet;
        facet.v1_idx = i;
        facet.v2_idx = j;
        
        Eigen::VectorXd edge = P.vertices[j] - P.vertices[i];
        Eigen::VectorXd n_plane = P.hyperplane_normal.normalized();
        
        Eigen::Vector3d cross_result = cross3d(n_plane, edge);
        facet.normal = cross_result;
        facet.normal.normalize();
        
        Eigen::VectorXd edge_center = (P.vertices[i] + P.vertices[j]) / 2.0;
        if (facet.normal.dot(edge_center - centroid) < 0) {
            facet.normal = -facet.normal;
        }
        
        facet.angle = atan2(facet.normal[1], facet.normal[0]);
        P.bucket_index.sorted_facets.push_back(facet);
    }
    
    sort(P.bucket_index.sorted_facets.begin(), P.bucket_index.sorted_facets.end(),
         [](const auto& a, const auto& b) { return a.angle < b.angle; });
    
    int m = P.bucket_index.sorted_facets.size();
    P.bucket_index.boundaries.resize(m);
    
    for (int k = 0; k < m; ++k) {
        double theta_k = P.bucket_index.sorted_facets[k].angle;
        double theta_k1 = P.bucket_index.sorted_facets[(k + 1) % m].angle;
        
        if (theta_k1 < theta_k) theta_k1 += 2.0 * M_PI;
        
        double mid = (theta_k + theta_k1) / 2.0;
        if (mid > M_PI) mid -= 2.0 * M_PI;
        
        P.bucket_index.boundaries[k] = mid;
    }
}

RoutingPolytope3D computeIntersection_3D(const Polytope& D, const Eigen::VectorXd& h, double offset) {
    RoutingPolytope3D result;
    result.hyperplane_normal = h;
    result.hyperplane_offset = offset;
    
    const double tol = 1e-8;
    int n = D.vertices.size() / 2;
    
    map<pair<int,int>, int> edge_to_vertex;
    
    auto addVertex = [&](const Eigen::VectorXd& v, int domain_v1_id, int domain_v2_id) {
        auto edge_key = make_pair(min(domain_v1_id, domain_v2_id), 
                                   max(domain_v1_id, domain_v2_id));
        if (edge_to_vertex.count(edge_key) == 0) {
            int idx = result.vertices.size();
            result.vertices.push_back(v);
            edge_to_vertex[edge_key] = idx;
            return idx;
        }
        return edge_to_vertex[edge_key];
    };
    
    for (const auto& v : D.vertices) {
        double val = h.dot(v.position) + offset;
        if (abs(val) < tol) {
            addVertex(v.position, v.id, v.id);
        }
    }
    
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        const auto& v1 = D.vertices[i];
        const auto& v2 = D.vertices[j];
        
        double val1 = h.dot(v1.position) + offset;
        double val2 = h.dot(v2.position) + offset;
        
        if (val1 * val2 < -tol*tol) {
            double lambda = -val1 / (val2 - val1);
            Eigen::VectorXd new_v = v1.position + lambda * (v2.position - v1.position);
            addVertex(new_v, v1.id, v2.id);
        }
    }
    
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        const auto& v1 = D.vertices[n + i];
        const auto& v2 = D.vertices[n + j];
        
        double val1 = h.dot(v1.position) + offset;
        double val2 = h.dot(v2.position) + offset;
        
        if (val1 * val2 < -tol*tol) {
            double lambda = -val1 / (val2 - val1);
            Eigen::VectorXd new_v = v1.position + lambda * (v2.position - v1.position);
            addVertex(new_v, v1.id, v2.id);
        }
    }
    
    for (int i = 0; i < n; ++i) {
        const auto& v1 = D.vertices[i];
        const auto& v2 = D.vertices[n + i];
        
        double val1 = h.dot(v1.position) + offset;
        double val2 = h.dot(v2.position) + offset;
        
        if (val1 * val2 < -tol*tol) {
            double lambda = -val1 / (val2 - val1);
            Eigen::VectorXd new_v = v1.position + lambda * (v2.position - v1.position);
            addVertex(new_v, v1.id, v2.id);
        }
    }
    
    if (result.vertices.size() >= 3) {
        for (size_t i = 0; i < result.vertices.size(); ++i) {
            size_t j = (i + 1) % result.vertices.size();
            result.edges.push_back({(int)i, (int)j});
        }
    }
    
    return result;
}

int findBucket(const vector<double>& boundaries, double theta_q) {
    int m = boundaries.size();
    if (m == 0) return 0;
    
    while (theta_q > M_PI) theta_q -= 2.0 * M_PI;
    while (theta_q < -M_PI) theta_q += 2.0 * M_PI;
    
    if (theta_q <= boundaries[0]) return 0;
    
    auto it = lower_bound(boundaries.begin(), boundaries.end(), theta_q);
    if (it == boundaries.end()) return m - 1;
    return distance(boundaries.begin(), it);
}

bool vertexBased_3D(const RoutingPolytope3D& P, const Eigen::VectorXd& h, double offset) {
    if (P.vertices.empty()) return false;
    
    double g_min = 1e100, g_max = -1e100;
    for (const auto& v : P.vertices) {
        double val = h.dot(v) + offset;
        g_min = min(g_min, val);
        g_max = max(g_max, val);
        if (g_min < 0 && g_max > 0) return true;
    }
    
    return g_min <= 0 && g_max >= 0;
}

bool bucketBased_3D(const RoutingPolytope3D& P, const Eigen::VectorXd& h, double offset) {
    if (P.vertices.empty() || P.bucket_index.sorted_facets.empty()) {
        return vertexBased_3D(P, h, offset);
    }
    
    Eigen::VectorXd n_hat = h.normalized();
    double theta_q = atan2(n_hat[1], n_hat[0]);
    
    int k_pos = findBucket(P.bucket_index.boundaries, theta_q);
    double theta_neg = atan2(-n_hat[1], -n_hat[0]);
    int k_neg = findBucket(P.bucket_index.boundaries, theta_neg);
    
    const auto& facet_max = P.bucket_index.sorted_facets[k_pos];
    const auto& facet_min = P.bucket_index.sorted_facets[k_neg];
    
    double g_max = -1e100;
    g_max = max(g_max, h.dot(P.vertices[facet_max.v1_idx]) + offset);
    g_max = max(g_max, h.dot(P.vertices[facet_max.v2_idx]) + offset);
    
    double g_min = 1e100;
    g_min = min(g_min, h.dot(P.vertices[facet_min.v1_idx]) + offset);
    g_min = min(g_min, h.dot(P.vertices[facet_min.v2_idx]) + offset);
    
    return g_min <= 0 && g_max >= 0;
}

bool isEmpty(const RoutingPolytope3D& P) {
    return P.vertices.empty();
}

pair<RoutingPolytope3D, RoutingPolytope3D> splitRoutingPolytope_3D(
    const RoutingPolytope3D& P,
    const Eigen::VectorXd& h,
    double offset
) {
    RoutingPolytope3D P1, P2;
    P1.hyperplane_normal = P.hyperplane_normal;
    P1.hyperplane_offset = P.hyperplane_offset;
    P2.hyperplane_normal = P.hyperplane_normal;
    P2.hyperplane_offset = P.hyperplane_offset;
    
    const double tol = 1e-8;
    
    for (size_t i = 0; i < P.vertices.size(); ++i) {
        double val = h.dot(P.vertices[i]) + offset;
        
        if (val <= tol) {
            P1.vertices.push_back(P.vertices[i]);
        }
        if (val >= -tol) {
            P2.vertices.push_back(P.vertices[i]);
        }
    }
    
    if (P.vertices.size() < 100) {
        for (size_t i = 0; i < P.vertices.size(); ++i) {
            for (size_t j = i + 1; j < P.vertices.size(); ++j) {
                const auto& v1 = P.vertices[i];
                const auto& v2 = P.vertices[j];
                
                double val1 = h.dot(v1) + offset;
                double val2 = h.dot(v2) + offset;
                
                if (val1 * val2 < -tol*tol) {
                    double lambda = -val1 / (val2 - val1);
                    Eigen::VectorXd vmid = v1 + lambda * (v2 - v1);
                    
                    P1.vertices.push_back(vmid);
                    P2.vertices.push_back(vmid);
                }
            }
        }
    }
    
    if (P1.vertices.size() >= 3) {
        for (size_t i = 0; i < P1.vertices.size(); ++i) {
            P1.edges.push_back({(int)i, (int)((i + 1) % P1.vertices.size())});
        }
    }
    
    if (P2.vertices.size() >= 3) {
        for (size_t i = 0; i < P2.vertices.size(); ++i) {
            P2.edges.push_back({(int)i, (int)((i + 1) % P2.vertices.size())});
        }
    }
    
    return {P1, P2};
}

struct ITreeNode {
    Polytope domain;
    pair<int,int> func_pair;
    ITreeNode* left;
    ITreeNode* right;
    bool is_leaf;
    
    ITreeNode() : func_pair({-1,-1}), left(nullptr), right(nullptr), is_leaf(true) {}
};

template<typename FeasibilityFunc>
ITreeNode* buildITree(
    const vector<LinearFunction>& functions,
    const vector<pair<int,int>>& intersections,
    const Polytope& initial_domain,
    FeasibilityFunc feasibility_check,
    int& total_checks,
    int& avg_P_size,
    bool use_buckets
) {
    ITreeNode* root = new ITreeNode();
    root->domain = initial_domain;
    root->is_leaf = true;
    
    int total_P_vertices = 0;
    int P_count = 0;
    
    for (const auto& [i, j] : intersections) {
        const auto& fi = functions[i];
        const auto& fj = functions[j];
        
        Eigen::VectorXd h = fi.a - fj.a;
        double offset = fi.b - fj.b;
        
        RoutingPolytope3D P_ij = computeIntersection_3D(initial_domain, h, offset);
        if (isEmpty(P_ij)) continue;
        
        if (use_buckets) {
            buildBucketIndex3D(P_ij);
        }
        
        queue<pair<ITreeNode*, RoutingPolytope3D>> Q;
        Q.push({root, P_ij});
        
        while (!Q.empty()) {
            auto [node, P] = Q.front();
            Q.pop();
            
            if (isEmpty(P)) continue;
            
            total_P_vertices += P.vertices.size();
            P_count++;
            
            if (node->is_leaf) {
                node->is_leaf = false;
                node->func_pair = {i, j};
                node->left = new ITreeNode();
                node->right = new ITreeNode();
                node->left->is_leaf = true;
                node->right->is_leaf = true;
                
                auto [D1, D2] = split_polytope(node->domain, h, offset);
                node->left->domain = D1;
                node->right->domain = D2;
                
                continue;
            }
            
            const auto& [fi_node, fj_node] = node->func_pair;
            const auto& fi_split = functions[fi_node];
            const auto& fj_split = functions[fj_node];
            
            Eigen::VectorXd h_node = fi_split.a - fj_split.a;
            double offset_node = fi_split.b - fj_split.b;
            
            total_checks++;
            bool intersects = feasibility_check(P, h_node, offset_node);
            
            if (!intersects) {
                double test_val = h_node.dot(P.vertices[0]) + offset_node;
                if (test_val > 0) {
                    Q.push({node->right, P});
                } else {
                    Q.push({node->left, P});
                }
            } else {
                auto [P1, P2] = splitRoutingPolytope_3D(P, h_node, offset_node);
                
                if (use_buckets) {
                    if (!isEmpty(P1)) buildBucketIndex3D(P1);
                    if (!isEmpty(P2)) buildBucketIndex3D(P2);
                }
                
                if (!isEmpty(P1)) Q.push({node->left, P1});
                if (!isEmpty(P2)) Q.push({node->right, P2});
            }
        }
    }
    
    avg_P_size = P_count > 0 ? total_P_vertices / P_count : 0;
    return root;
}

void countNodes(ITreeNode* node, int& leaf_count, int& internal_count) {
    if (!node) return;
    if (node->is_leaf) {
        leaf_count++;
    } else {
        internal_count++;
        countNodes(node->left, leaf_count, internal_count);
        countNodes(node->right, leaf_count, internal_count);
    }
}

int main() {
    string csv_path = OUTPUT_DIR + "/itree_d3_data.csv";
    
    cout << "\n========================================================================\n";
    cout << "I-Tree d=3: Recursive Index Propagation - Generating CSV Data\n";
    cout << "========================================================================\n";
    cout << "Output: " << csv_path << "\n\n";
    
    ofstream csv(csv_path);
    if (!csv.is_open()) {
        cerr << "ERROR: Could not open " << csv_path << endl;
        return 1;
    }
    
    csv << "n,num_funcs,num_intersections,nodes,checks,avg_P,vertex_ms,bucket_ms,speedup\n";
    
    vector<tuple<int, int, string>> test_cases = {
        {20, 8, "n=40, 8 funcs (28 inters)"},
        {50, 10, "n=100, 10 funcs (45 inters)"},
        {100, 12, "n=200, 12 funcs (66 inters)"}
    };
    
    for (const auto& [n, num_funcs, desc] : test_cases) {
        cout << desc << "...\n";
        
        Polytope prism = create_prism(n);
        
        mt19937 rng(42 + n);
        normal_distribution<double> dist(0.0, 1.0);
        
        vector<LinearFunction> functions;
        for (int i = 0; i < num_funcs; ++i) {
            LinearFunction f;
            f.a = Eigen::VectorXd(3);
            f.a[0] = dist(rng);
            f.a[1] = dist(rng);
            f.a[2] = dist(rng);
            f.b = dist(rng);
            functions.push_back(f);
        }
        
        vector<pair<int,int>> intersections;
        for (int i = 0; i < num_funcs; ++i) {
            for (int j = i+1; j < num_funcs; ++j) {
                intersections.push_back({i, j});
            }
        }
        
        // VERTEX-BASED
        int vb_checks = 0, vb_avg_P = 0;
        auto t0 = clock_type::now();
        ITreeNode* tree_vb = buildITree(functions, intersections, prism, vertexBased_3D, vb_checks, vb_avg_P, false);
        auto t1 = clock_type::now();
        double vb_ms = chrono::duration<double, milli>(t1 - t0).count();
        
        int vb_leaves = 0, vb_internal = 0;
        countNodes(tree_vb, vb_leaves, vb_internal);
        
        // BUCKET-BASED
        int bucket_checks = 0, bucket_avg_P = 0;
        auto t2 = clock_type::now();
        ITreeNode* tree_bucket = buildITree(functions, intersections, prism, bucketBased_3D, bucket_checks, bucket_avg_P, true);
        auto t3 = clock_type::now();
        double bucket_ms = chrono::duration<double, milli>(t3 - t2).count();
        
        int bucket_leaves = 0, bucket_internal = 0;
        countNodes(tree_bucket, bucket_leaves, bucket_internal);
        
        double speedup = vb_ms / bucket_ms;
        
        csv << (2*n) << "," << num_funcs << "," << intersections.size() << ","
            << (vb_leaves + vb_internal) << "," << vb_checks << "," << vb_avg_P << ","
            << vb_ms << "," << bucket_ms << "," << speedup << "\n";
        
        cout << "  Nodes: " << (vb_leaves + vb_internal) 
             << ", Avg P: " << vb_avg_P
             << ", Speedup: " << fixed << setprecision(2) << speedup << "x\n";
    }
    
    csv.close();
    
    cout << "\n========================================================================\n";
    cout << "SUCCESS! Data saved to: " << csv_path << "\n";
    cout << "========================================================================\n\n";
    
    return 0;
}
