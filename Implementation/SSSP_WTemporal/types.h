/**
 * types.h
 * Common data types and structures for SSSP algorithm implementations
 */

#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <string>
#include <set>
#include <map>
#include <utility>
#include <limits>
#include <iostream>

// Type definitions
typedef int VertexID;                        // Vertex identifier type
typedef double Weight;                       // Edge weight type
typedef std::pair<VertexID, Weight> Edge;    // Edge representation (target, weight)

// Constants
const Weight INF = std::numeric_limits<Weight>::infinity();  // Infinity value for distance

// Edge change type
enum ChangeType {
    INSERTION,
    DELETION
};

// Edge change representation
struct EdgeChange {
    VertexID source;
    VertexID target;
    Weight weight;
    ChangeType type;
    
    // Default constructor
EdgeChange() : source(0), target(0), weight(0.0), type(INSERTION) {}
    
    EdgeChange(VertexID s, VertexID t, Weight w, ChangeType ct)
        : source(s), target(t), weight(w), type(ct) {}
};

// SSSP Tree node information
struct SSSPNode {
    VertexID parent;     // Parent in SSSP tree
    Weight distance;     // Distance from source
    bool affected;       // Affected by any change
    bool affected_del;   // Affected by deletion
    
    SSSPNode()
        : parent(-1), distance(INF), affected(false), affected_del(false) {}
};

// Performance metrics
struct Metrics {
    double total_time;           // Total execution time
    double step1_time;           // Time for step 1 (identifying affected subgraph)
    double step2_time;           // Time for step 2 (updating affected subgraph)
    int affected_vertices;       // Number of affected vertices
    int iterations;              // Number of iterations in step 2
    
    Metrics()
        : total_time(0), step1_time(0), step2_time(0), affected_vertices(0), iterations(0) {}
};

#endif // TYPES_H
