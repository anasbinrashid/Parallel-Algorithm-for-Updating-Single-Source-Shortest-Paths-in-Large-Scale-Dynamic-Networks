/**
 * types.h
 * Common data types and structures for SSSP algorithm implementations
 * Updated with timestamp support for temporal networks
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
#include <chrono>

// Type definitions
typedef int VertexID;                        // Vertex identifier type
typedef double Weight;                       // Edge weight type
typedef std::pair<VertexID, Weight> Edge;    // Edge representation (target, weight)
typedef double Timestamp;                    // Edge timestamp type

// Constants
const Weight INF = std::numeric_limits<Weight>::infinity();  // Infinity value for distance

// Edge change type
enum ChangeType {
    INSERTION,
    DELETION
};

// Edge change representation with timestamp
struct EdgeChange {
    VertexID source;
    VertexID target;
    Weight weight;
    ChangeType type;
    Timestamp time;  // When this change occurs
    
    // Default constructor
    EdgeChange() : source(0), target(0), weight(0.0), type(INSERTION), time(0.0) {}
    
    // Existing constructor
    EdgeChange(VertexID s, VertexID t, Weight w, ChangeType ct, Timestamp tm = 0.0)
        : source(s), target(t), weight(w), type(ct), time(tm) {}
};

// Edge with timestamp representation
struct TimedEdge {
    VertexID source;
    VertexID target;
    Weight weight;
    Timestamp start_time;   // When the edge appears
    Timestamp end_time;     // When the edge disappears (INF if still present)
    
    TimedEdge(VertexID s, VertexID t, Weight w, Timestamp st, Timestamp et = INF)
        : source(s), target(t), weight(w), start_time(st), end_time(et) {}
        
    // Check if edge exists at a given time
    bool existsAt(Timestamp time) const {
        return (time >= start_time && (end_time == INF || time < end_time));
    }
};

// SSSP Tree node information
struct SSSPNode {
    VertexID parent;     // Parent in SSSP tree
    Weight distance;     // Distance from source
    bool affected;       // Affected by any change
    bool affected_del;   // Affected by deletion
    Timestamp time;      // Time when this SSSP tree is valid
    
    SSSPNode()
        : parent(-1), distance(INF), affected(false), affected_del(false), time(0.0) {}
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

// Timer class for performance measurements


class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    bool running;
    
public:
    // Constructor
    Timer() : running(false) {}
    
    // Start the timer
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        running = true;
    }
    
    // Stop the timer
    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
        running = false;
    }
    
    // Get elapsed time in seconds
    double getElapsedTimeInSeconds() const {
        if (running) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(current_time - start_time);
            return duration.count() / 1000000.0;
        } else {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            return duration.count() / 1000000.0;
        }
    }
    
};

#endif // TYPES_H
