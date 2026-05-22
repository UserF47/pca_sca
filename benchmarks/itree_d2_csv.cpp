#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>
#include <queue>
#include <set>
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

struct BucketIndex2D {
    vector<pair<Eigen::VectorXd, double>> sorted_normals;
    vector<double> boundaries;
};

struct RoutingPolytope2D {
    Eigen::VectorXd v1, v2;
    Eigen::VectorXd hyperplane_normal;
    double hyperplane_offset;
    BucketIndex2D bucket_index;
};

Polytope create_polygon(int n) {
    Polytope poly;
    poly.dim = 2;
    
    for (int i = 0; i < n; ++i) {
        double angle = 2.0 * M_PI * i / n;
        Vertex v;
        v.id = i;
        v.position = Eigen::VectorXd(2);
        v.position[0] = cos(angle);
        v.position[1] = sin(angle);
        poly.vertices.push_back(v);
    }
    
    return poly;
}

void buildBucketIndex(RoutingPolytope2D& P) {
    Eigen::VectorXd edge = P.v2 - P.v1;
    edge.normalize();
    
    Eigen::VectorXd n1 = edge;
    Eigen::VectorXd n2 = -edge;
    
    double angle1 = atan2(n1[1], n1[0]);
    double angle2 = atan2(n2[1], n2[0]);
    
    P.bucket_index.sorted_normals = {{n1, angle1}, {n2, angle2}};
    
    sort(P.bucket_index.sorted_normals.begin(), P.bucket_index.sorted_normals.end(),
         [](const auto& a, const auto& b) { return a.second < b.second; });
    
    double theta1 = P.bucket_index.sorted_normals[0].second;
    double theta2 = P.bucket_index.sorted_normals[1].second;
    
    if (theta2 < theta1) theta2 += 2.0 * M_PI;
    double mid = (theta1 + theta2) / 2.0;
    if (mid > M_PI) mid -= 2.0 * M_PI;
    
    P.bucket_index.boundaries = {mid};
}

BucketIndex2D inheritAndUpdate(const BucketIndex2D& parent_idx, const Eigen::VectorXd& new_normal) {
    BucketIndex2D child_idx;
    
    if (!parent_idx.sorted_normals.empty()) {
        child_idx.sorted_normals.push_back(parent_idx.sorted_normals[0]);
    }
    
    double new_angle = atan2(new_normal[1], new_normal[0]);
    child_idx.sorted_normals.push_back({new_normal, new_angle});
    
    sort(child_idx.sorted_normals.begin(), child_idx.sorted_normals.end(),
         [](const auto& a, const auto& b) { return a.second < b.second; });
    
    if (child_idx.sorted_normals.size() >= 2) {
        double theta1 = child_idx.sorted_normals[0].second;
        double theta2 = child_idx.sorted_normals[1].second;
        
        if (theta2 < theta1) theta2 += 2.0 * M_PI;
        double mid = (theta1 + theta2) / 2.0;
        if (mid > M_PI) mid -= 2.0 * M_PI;
        
        child_idx.boundaries = {mid};
    }
    
    return child_idx;
}

RoutingPolytope2D computeIntersection_2D(const Polytope& D, const Eigen::VectorXd& h, double offset) {
    RoutingPolytope2D result;
    result.hyperplane_normal = h;
    result.hyperplane_offset = offset;
    
    const double tol = 1e-8;
    vector<Eigen::VectorXd> intersection_points;
    
    for (const auto& v : D.vertices) {
        double val = h.dot(v.position) + offset;
        if (abs(val) < tol) {
            intersection_points.push_back(v.position);
        }
    }
    
    for (size_t i = 0; i < D.vertices.size(); ++i) {
        size_t j = (i + 1) % D.vertices.size();
        
        const auto& v1 = D.vertices[i];
        const auto& v2 = D.vertices[j];
        
        double val1 = h.dot(v1.position) + offset;
        double val2 = h.dot(v2.position) + offset;
        
        if (val1 * val2 < -tol*tol) {
            double lambda = -val1 / (val2 - val1);
            intersection_points.push_back(v1.position + lambda * (v2.position - v1.position));
        }
    }
    
    if (intersection_points.size() >= 2) {
        result.v1 = intersection_points[0];
        result.v2 = intersection_points[1];
    }
    
    return result;
}

bool vertexBased_2D(const RoutingPolytope2D& P, const Eigen::VectorXd& h, double offset) {
    double val1 = h.dot(P.v1) + offset;
    double val2 = h.dot(P.v2) + offset;
    return min(val1, val2) <= 0 && max(val1, val2) >= 0;
}

bool bucketBased_2D(const RoutingPolytope2D& P, const Eigen::VectorXd& h, double offset) {
    if (P.bucket_index.sorted_normals.empty()) {
        return vertexBased_2D(P, h, offset);
    }
    
    Eigen::VectorXd n_hat = h.normalized();
    double theta_q = atan2(n_hat[1], n_hat[0]);
    
    double dist1 = abs(theta_q - P.bucket_index.sorted_normals[0].second);
    double dist2 = P.bucket_index.sorted_normals.size() > 1 ? 
                   abs(theta_q - P.bucket_index.sorted_normals[1].second) : 1e10;
    
    if (dist1 > M_PI) dist1 = 2.0 * M_PI - dist1;
    if (dist2 > M_PI) dist2 = 2.0 * M_PI - dist2;
    
    bool v1_is_max = (dist1 < dist2);
    
    double g_max = v1_is_max ? h.dot(P.v1) + offset : h.dot(P.v2) + offset;
    double g_min = v1_is_max ? h.dot(P.v2) + offset : h.dot(P.v1) + offset;
    
    return g_min <= 0 && g_max >= 0;
}

pair<RoutingPolytope2D, RoutingPolytope2D> splitRoutingPolytope(
    const RoutingPolytope2D& P,
    const Eigen::VectorXd& h,
    double offset,
    bool use_buckets
) {
    RoutingPolytope2D P1, P2;
    P1.hyperplane_normal = P.hyperplane_normal;
    P1.hyperplane_offset = P.hyperplane_offset;
    P2.hyperplane_normal = P.hyperplane_normal;
    P2.hyperplane_offset = P.hyperplane_offset;
    
    const double tol = 1e-8;
    
    double val1 = h.dot(P.v1) + offset;
    double val2 = h.dot(P.v2) + offset;
    
    if (val1 * val2 < -tol*tol) {
        double lambda = -val1 / (val2 - val1);
        Eigen::VectorXd vmid = P.v1 + lambda * (P.v2 - P.v1);
        
        P1.v1 = (val1 <= 0) ? P.v1 : vmid;
        P1.v2 = (val2 <= 0) ? P.v2 : vmid;
        
        P2.v1 = (val1 >= 0) ? P.v1 : vmid;
        P2.v2 = (val2 >= 0) ? P.v2 : vmid;
        
        if (use_buckets) {
            Eigen::VectorXd split_normal = h.normalized();
            P1.bucket_index = inheritAndUpdate(P.bucket_index, split_normal);
            P2.bucket_index = inheritAndUpdate(P.bucket_index, -split_normal);
        }
    } else {
        if (val1 <= 0 && val2 <= 0) {
            P1 = P;
            P2.v1 = P2.v2 = Eigen::VectorXd::Zero(2);
        } else {
            P2 = P;
            P1.v1 = P1.v2 = Eigen::VectorXd::Zero(2);
        }
    }
    
    return {P1, P2};
}

bool isEmpty(const RoutingPolytope2D& P) {
    return (P.v1 - P.v2).norm() < 1e-10;
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
    bool use_buckets
) {
    ITreeNode* root = new ITreeNode();
    root->domain = initial_domain;
    root->is_leaf = true;
    
    for (const auto& [i, j] : intersections) {
        const auto& fi = functions[i];
        const auto& fj = functions[j];
        
        Eigen::VectorXd h = fi.a - fj.a;
        double offset = fi.b - fj.b;
        
        RoutingPolytope2D P_ij = computeIntersection_2D(initial_domain, h, offset);
        if (isEmpty(P_ij)) continue;
        
        if (use_buckets) {
            buildBucketIndex(P_ij);
        }
        
        queue<pair<ITreeNode*, RoutingPolytope2D>> Q;
        Q.push({root, P_ij});
        
        while (!Q.empty()) {
            auto [node, P] = Q.front();
            Q.pop();
            
            if (isEmpty(P)) continue;
            
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
                double test_val = h_node.dot(P.v1) + offset_node;
                if (test_val > 0) {
                    Q.push({node->right, P});
                } else {
                    Q.push({node->left, P});
                }
            } else {
                auto [P1, P2] = splitRoutingPolytope(P, h_node, offset_node, use_buckets);
                if (!isEmpty(P1)) Q.push({node->left, P1});
                if (!isEmpty(P2)) Q.push({node->right, P2});
            }
        }
    }
    
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
    string csv_path = OUTPUT_DIR + "/itree_d2_data.csv";
    
    cout << "\n========================================================================\n";
    cout << "I-Tree d=2: Recursive Index Propagation - Generating CSV Data\n";
    cout << "========================================================================\n";
    cout << "Output: " << csv_path << "\n\n";
    
    ofstream csv(csv_path);
    if (!csv.is_open()) {
        cerr << "ERROR: Could not open " << csv_path << endl;
        return 1;
    }
    
    csv << "n,num_funcs,num_intersections,nodes,checks,vertex_ms,bucket_ms,speedup\n";
    
    vector<tuple<int, int, string>> test_cases = {
        {100, 10, "n=100, 10 funcs (45 inters)"},
        {500, 15, "n=500, 15 funcs (105 inters)"},
        {1000, 20, "n=1000, 20 funcs (190 inters)"}
    };
    
    for (const auto& [n, num_funcs, desc] : test_cases) {
        cout << desc << "...\n";
        
        Polytope polygon = create_polygon(n);
        
        mt19937 rng(42 + n);
        normal_distribution<double> dist(0.0, 1.0);
        
        vector<LinearFunction> functions;
        for (int i = 0; i < num_funcs; ++i) {
            LinearFunction f;
            f.a = Eigen::VectorXd(2);
            f.a[0] = dist(rng);
            f.a[1] = dist(rng);
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
        int vb_checks = 0;
        auto t0 = clock_type::now();
        ITreeNode* tree_vb = buildITree(functions, intersections, polygon, vertexBased_2D, vb_checks, false);
        auto t1 = clock_type::now();
        double vb_ms = chrono::duration<double, milli>(t1 - t0).count();
        
        int vb_leaves = 0, vb_internal = 0;
        countNodes(tree_vb, vb_leaves, vb_internal);
        
        // BUCKET-BASED
        int bucket_checks = 0;
        auto t2 = clock_type::now();
        ITreeNode* tree_bucket = buildITree(functions, intersections, polygon, bucketBased_2D, bucket_checks, true);
        auto t3 = clock_type::now();
        double bucket_ms = chrono::duration<double, milli>(t3 - t2).count();
        
        int bucket_leaves = 0, bucket_internal = 0;
        countNodes(tree_bucket, bucket_leaves, bucket_internal);
        
        double speedup = vb_ms / bucket_ms;
        
        csv << n << "," << num_funcs << "," << intersections.size() << ","
            << (vb_leaves + vb_internal) << "," << vb_checks << ","
            << vb_ms << "," << bucket_ms << "," << speedup << "\n";
        
        cout << "  Nodes: " << (vb_leaves + vb_internal) 
             << ", Speedup: " << fixed << setprecision(2) << speedup << "x\n";
    }
    
    csv.close();
    
    cout << "\n========================================================================\n";
    cout << "SUCCESS! Data saved to: " << csv_path << "\n";
    cout << "========================================================================\n\n";
    
    return 0;
}
