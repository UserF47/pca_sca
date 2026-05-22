#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cmath>
#include <cstdlib>
#include "PolytopeOps.h"
#include "SphericalBucketing_v4.h"

using namespace std;
using clock_type = chrono::high_resolution_clock;

// Output directory in Windows filesystem
const string OUTPUT_DIR = "/mnt/c/Users/sahil/Documents/DrCaiAFRL";

Polytope create_regular_polygon(int n) {
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
        poly.constraints[i] = {i, (i + n - 1) % n};
    }
    
    return poly;
}

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
    
    for (int i = 0; i < 2*n; ++i) {
        poly.constraints[i] = {0, 1, 2};
    }
    
    return poly;
}

Polytope create_product_polygon(int m, int n) {
    Polytope poly;
    poly.dim = 4;
    
    for (int i = 0; i < m; ++i) {
        double angle1 = 2.0 * M_PI * i / m;
        for (int j = 0; j < n; ++j) {
            double angle2 = 2.0 * M_PI * j / n;
            Vertex v;
            v.id = i * n + j;
            v.position = Eigen::VectorXd(4);
            v.position[0] = cos(angle1);
            v.position[1] = sin(angle1);
            v.position[2] = cos(angle2);
            v.position[3] = sin(angle2);
            poly.vertices.push_back(v);
        }
    }
    
    for (int i = 0; i < m*n; ++i) {
        poly.constraints[i] = {0, 1, 2, 3};
    }
    
    return poly;
}

Polytope create_5d_prism(int m, int n) {
    Polytope poly;
    poly.dim = 5;
    
    for (int i = 0; i < m; ++i) {
        double angle1 = 2.0 * M_PI * i / m;
        for (int j = 0; j < n; ++j) {
            double angle2 = 2.0 * M_PI * j / n;
            Vertex v;
            v.id = i * n + j;
            v.position = Eigen::VectorXd(5);
            v.position[0] = cos(angle1);
            v.position[1] = sin(angle1);
            v.position[2] = cos(angle2);
            v.position[3] = sin(angle2);
            v.position[4] = 0.0;
            poly.vertices.push_back(v);
        }
    }
    
    for (int i = 0; i < m; ++i) {
        double angle1 = 2.0 * M_PI * i / m;
        for (int j = 0; j < n; ++j) {
            double angle2 = 2.0 * M_PI * j / n;
            Vertex v;
            v.id = m*n + i * n + j;
            v.position = Eigen::VectorXd(5);
            v.position[0] = cos(angle1);
            v.position[1] = sin(angle1);
            v.position[2] = cos(angle2);
            v.position[3] = sin(angle2);
            v.position[4] = 1.0;
            poly.vertices.push_back(v);
        }
    }
    
    for (int i = 0; i < 2*m*n; ++i) {
        poly.constraints[i] = {0, 1, 2, 3, 4};
    }
    
    return poly;
}

Polytope generate_with_splits(int d, int num_splits) {
    mt19937 rng(12345 + d);
    Polytope poly = create_hypercube(d);
    
    normal_distribution<double> norm_dist(0.0, 1.0);
    
    for (int i = 0; i < num_splits; ++i) {
        Eigen::VectorXd h(d);
        for (int j = 0; j < d; ++j) h[j] = norm_dist(rng);
        h.normalize();
        
        auto [p1, p2] = split_polytope(poly, h, -(1000 + i));
        if (p1.vertices.size() >= p2.vertices.size() && !p1.vertices.empty()) {
            poly = move(p1);
        } else if (!p2.vertices.empty()) {
            poly = move(p2);
        }
    }
    
    return poly;
}

void run_test_csv(const Polytope& poly, ofstream& csv, int dim) {
    int n = poly.vertices.size();
    
    try {
        Eigen::VectorXd centroid = compute_polytope_centroid_v4(poly);
        BucketIndex idx = buildBucketIndex_v4(poly, centroid);
        vector<Facet> facets = extract_facets_v4(poly, centroid);
        
        int m = facets.size();
        if (m == 0) {
            cout << "  d=" << dim << ", n=" << n << " FAILED (no facets)\n";
            return;
        }
        
        int total_k = 0;
        for (const auto& f : facets) total_k += f.vertex_positions.size();
        double avg_k = (double)total_k / m;
        
        mt19937 rng(99999);
        vector<Eigen::VectorXd> normals(1000);
        vector<double> offsets(1000);
        for (int i = 0; i < 1000; ++i) {
            normal_distribution<double> dist(0.0, 1.0);
            Eigen::VectorXd h(poly.dim);
            for (int j = 0; j < poly.dim; ++j) h[j] = dist(rng);
            normals[i] = h.normalized();
            offsets[i] = uniform_real_distribution<double>(-2.0, 2.0)(rng);
        }
        
        for (int i = 0; i < 50; ++i) {
            vertexBasedFeasibility(poly, normals[i], offsets[i]);
            bucketFeasibility_v4(poly, idx, facets, normals[i], offsets[i]);
        }
        
        auto t0 = clock_type::now();
        for (int i = 0; i < 1000; ++i) vertexBasedFeasibility(poly, normals[i], offsets[i]);
        auto t1 = clock_type::now();
        double vb_us = chrono::duration<double, micro>(t1 - t0).count() / 1000;
        
        auto t2 = clock_type::now();
        for (int i = 0; i < 1000; ++i) bucketFeasibility_v4(poly, idx, facets, normals[i], offsets[i]);
        auto t3 = clock_type::now();
        double cb_us = chrono::duration<double, micro>(t3 - t2).count() / 1000;
        
        double speedup = (cb_us > 0) ? vb_us / cb_us : 0.0;
        
        csv << dim << "," << n << "," << m << "," << avg_k << "," 
            << vb_us << "," << cb_us << "," << speedup << "\n";
        
        cout << "  d=" << dim << ", n=" << setw(6) << n << " -> speedup=" 
             << fixed << setprecision(2) << speedup << "x\n";
             
    } catch (const exception& e) {
        cout << "  d=" << dim << ", n=" << n << " FAILED\n";
    }
}

int main() {
    string csv_path = OUTPUT_DIR + "/feasibility_data.csv";
    
    cout << "\n========================================================================\n";
    cout << "Feasibility Checking Benchmark - Generating CSV Data\n";
    cout << "========================================================================\n";
    cout << "Output: " << csv_path << "\n\n";
    
    ofstream csv(csv_path);
    if (!csv.is_open()) {
        cerr << "ERROR: Could not open " << csv_path << endl;
        cerr << "Make sure the directory exists!\n";
        return 1;
    }
    
    csv << "dimension,n,m,avg_k,vertex_us,bucket_us,speedup\n";
    
    cout << "d=2...\n";
    for (int n : {10, 20, 50, 100, 150, 200, 300, 500, 1000}) {
        run_test_csv(create_regular_polygon(n), csv, 2);
    }
    
    cout << "\nd=3...\n";
    for (int base : {50, 100, 200, 300, 500, 700, 1000, 1500, 2000}) {
        run_test_csv(create_prism(base), csv, 3);
    }
    
    cout << "\nd=4...\n";
    for (auto [m, n] : vector<pair<int,int>>{{10,10}, {15,15}, {20,20}, {25,25}, {30,30}, {40,40}, {50,50}, {60,60}, {80,80}}) {
        run_test_csv(create_product_polygon(m, n), csv, 4);
    }
    
    cout << "\nd=5...\n";
    for (auto [m, n] : vector<pair<int,int>>{{7,7}, {10,10}, {12,12}, {15,15}, {18,18}, {20,20}, {25,25}, {30,30}, {35,35}}) {
        run_test_csv(create_5d_prism(m, n), csv, 5);
    }
    
    cout << "\nd=6...\n";
    for (int splits : {50, 60, 70, 80, 90, 100, 110, 120}) {
        run_test_csv(generate_with_splits(6, splits), csv, 6);
    }
    
    cout << "\nd=7...\n";
    for (int splits : {40, 50, 60, 70, 80, 90, 100}) {
        run_test_csv(generate_with_splits(7, splits), csv, 7);
    }
    
    cout << "\nd=8...\n";
    for (int splits : {40, 50, 60, 70, 80, 90}) {
        run_test_csv(generate_with_splits(8, splits), csv, 8);
    }
    
    cout << "\nd=9...\n";
    for (int splits : {30, 40, 50, 60, 70, 80}) {
        run_test_csv(generate_with_splits(9, splits), csv, 9);
    }
    
    cout << "\nd=10...\n";
    for (int splits : {30, 40, 50, 60, 70}) {
        run_test_csv(generate_with_splits(10, splits), csv, 10);
    }
    
    csv.close();
    
    cout << "\n========================================================================\n";
    cout << "SUCCESS! Data saved to: " << csv_path << "\n";
    cout << "========================================================================\n\n";
    
    return 0;
}
