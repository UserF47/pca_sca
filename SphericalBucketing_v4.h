#pragma once
#include <vector>
#include <cmath>
#include <map>
#include <algorithm>
#include <limits>
#include <unordered_set>
#include <Eigen/Dense>
#include "PolytopeStructs.h"
#include "PolytopeOps.h"

/**
 * V4: MAXIMUM EARLY TERMINATION
 * Scan F_max until g > 0 found, then scan F_min until g < 0 found
 */

struct Facet {
    std::vector<int> vertex_indices;
    std::vector<Eigen::VectorXd> vertex_positions;
    Eigen::VectorXd normal;
};

struct BucketIndex {
    int K;
    std::vector<std::vector<int>> buckets;
    std::vector<int> redirect;
};

inline int hashNormal(const Eigen::VectorXd& n, int K) {
    if (n.size() < 2 || K <= 0) return 0;
    double theta = std::atan2(n(1), n(0));
    if (theta < 0) theta += 2.0 * M_PI;
    int k = static_cast<int>(theta / (2.0 * M_PI) * K) % K;
    return k;
}

inline void buildRedirectTable(BucketIndex& idx) {
    int K = idx.K;
    idx.redirect.assign(K, -1);
    
    int last = -1;
    for (int k = 0; k < K; ++k) {
        if (!idx.buckets[k].empty()) {
            last = k;
            idx.redirect[k] = k;
        } else if (last != -1) {
            idx.redirect[k] = last;
        }
    }
    
    last = -1;
    for (int k = K - 1; k >= 0; --k) {
        if (!idx.buckets[k].empty()) {
            last = k;
        } else if (last != -1) {
            if (idx.redirect[k] == -1 || 
                std::abs(k - last) < std::abs(k - idx.redirect[k])) {
                idx.redirect[k] = last;
            }
        }
    }
}

inline Eigen::VectorXd compute_facet_normal_v4(
    const std::vector<Eigen::VectorXd>& vertices,
    const Eigen::VectorXd& polytope_centroid,
    int d
) {
    if (vertices.size() < (size_t)d) {
        throw std::runtime_error("compute_facet_normal: need at least d vertices");
    }
    
    Eigen::MatrixXd A(d - 1, d);
    for (int i = 0; i < d - 1; ++i) {
        A.row(i) = vertices[i + 1] - vertices[0];
    }
    
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd normal = svd.matrixV().col(d - 1);
    normal.normalize();
    
    Eigen::VectorXd facet_centroid = Eigen::VectorXd::Zero(d);
    for (const auto& v : vertices) {
        facet_centroid += v;
    }
    facet_centroid /= vertices.size();
    
    if (normal.dot(facet_centroid - polytope_centroid) < 0) {
        normal = -normal;
    }
    
    return normal;
}

inline std::vector<Facet> extract_facets_v4(
    const Polytope& poly,
    const Eigen::VectorXd& polytope_centroid
) {
    std::vector<Facet> facets;
    int d = poly.dim;
    
    std::map<std::vector<int>, std::vector<int>> facet_vertex_map;
    
    for (size_t i = 0; i < poly.vertices.size(); ++i) {
        const auto& constraints = poly.constraints.at(poly.vertices[i].id);
        int facet_constraint_count = (int)constraints.size() - (d - 1);
        auto subsets = get_combinations(constraints, facet_constraint_count);
        
        for (const auto& subset : subsets) {
            facet_vertex_map[subset].push_back(poly.vertices[i].id);
        }
    }
    
    for (const auto& [constraint_set, vertex_ids] : facet_vertex_map) {
        if (vertex_ids.size() < (size_t)d) continue;
        
        Facet f;
        f.vertex_indices = vertex_ids;
        
        std::vector<Eigen::VectorXd> positions;
        for (int vid : vertex_ids) {
            for (const auto& v : poly.vertices) {
                if (v.id == vid) {
                    positions.push_back(v.position);
                    break;
                }
            }
        }
        
        if (positions.size() != vertex_ids.size()) continue;
        
        f.vertex_positions = positions;
        f.normal = compute_facet_normal_v4(positions, polytope_centroid, d);
        
        facets.push_back(f);
    }
    
    return facets;
}

inline BucketIndex buildBucketIndex_v4(const Polytope& P, const Eigen::VectorXd& centroid) {
    std::vector<Facet> facets = extract_facets_v4(P, centroid);
    
    int m = facets.size();
    BucketIndex idx;
    idx.K = m;
    idx.buckets.resize(m);
    
    for (int i = 0; i < m; ++i) {
        int k = hashNormal(facets[i].normal, m);
        idx.buckets[k].push_back(i);
    }
    
    buildRedirectTable(idx);
    
    return idx;
}

inline bool vertexBasedFeasibility(
    const Polytope& P,
    const Eigen::VectorXd& n,
    double b
) {
    bool found_neg = false, found_pos = false;
    
    for (const auto& v : P.vertices) {
        double val = n.dot(v.position) + b;
        if (val < 0) found_neg = true;
        else if (val > 0) found_pos = true;
        
        if (found_neg && found_pos) return true;
    }
    
    return false;
}

/**
 * V4: MAXIMUM EARLY TERMINATION
 * 
 * Strategy:
 * 1. Find one vertex on F_max with g > 0 (stop as soon as found)
 * 2. Find one vertex on F_min with g < 0 (stop as soon as found)
 * 3. Return true if both conditions met
 * 
 * This minimizes vertex evaluations, especially on non-simplicial facets!
 */
inline bool bucketFeasibility_v4(
    const Polytope& P,
    const BucketIndex& idx,
    const std::vector<Facet>& facets,
    const Eigen::VectorXd& n_hat,
    double b
) {
    int d = P.dim;
    int K = idx.K;
    
    // Step 1: Hash
    int k = hashNormal(n_hat, K);
    
    // Step 2: Collect candidates
    std::unordered_set<int> candidate_set;
    for (int delta = -d; delta <= d; ++delta) {
        int j = ((k + delta) % K + K) % K;
        int target = idx.buckets[j].empty() ? idx.redirect[j] : j;
        if (target == -1) continue;
        
        for (int fi : idx.buckets[target]) {
            candidate_set.insert(fi);
        }
    }
    
    std::vector<int> candidates(candidate_set.begin(), candidate_set.end());
    
    // Step 3: Find extremal facets
    int f_max_idx = -1, f_min_idx = -1;
    double best_pos = -2.0, best_neg = 2.0;
    
    for (int fi : candidates) {
        double dot = facets[fi].normal.dot(n_hat);
        if (dot > best_pos) {
            best_pos = dot;
            f_max_idx = fi;
        }
        if (dot < best_neg) {
            best_neg = dot;
            f_min_idx = fi;
        }
    }
    
    if (f_max_idx == -1 || f_min_idx == -1) {
        return vertexBasedFeasibility(P, n_hat, b);
    }
    
    // Step 4: MAXIMUM EARLY TERMINATION
    // Scan F_max until we find g > 0
    bool found_pos = false;
    for (const auto& v_pos : facets[f_max_idx].vertex_positions) {
        double val = n_hat.dot(v_pos) + b;
        if (val > 0) {
            found_pos = true;
            break;  // STOP immediately when found!
        }
    }
    
    // If no positive found on F_max, can't intersect
    if (!found_pos) return false;
    
    // Scan F_min until we find g < 0
    bool found_neg = false;
    for (const auto& v_pos : facets[f_min_idx].vertex_positions) {
        double val = n_hat.dot(v_pos) + b;
        if (val < 0) {
            found_neg = true;
            break;  // STOP immediately when found!
        }
    }
    
    // Intersection exists iff we found both
    return found_neg && found_pos;
}

inline Eigen::VectorXd compute_polytope_centroid_v4(const Polytope& poly) {
    if (poly.vertices.empty()) {
        throw std::runtime_error("Empty polytope");
    }
    
    Eigen::VectorXd centroid = Eigen::VectorXd::Zero(poly.dim);
    for (const auto& v : poly.vertices) {
        centroid += v.position;
    }
    centroid /= poly.vertices.size();
    
    return centroid;
}
