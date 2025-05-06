/**
 * utils.h
 * Utility functions for timing, I/O, etc.
 */

#ifndef UTILS_H
#define UTILS_H

#include "types.h"
#include "graph.h"
#include <chrono>
#include <string>
#include <vector>
#include <random>
#include <fstream>

// Timer class for performance measurements
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    bool running;
    
public:
    // Constructor
    Timer();
    
    // Start the timer
    void start();
    
    // Stop the timer
    void stop();
    
    // Get elapsed time in seconds
    double getElapsedTimeInSeconds() const;
};

// Generate a random temporal graph
Graph generateTemporalGraph(int num_vertices, int num_initial_edges, 
                           int num_changes, double insertion_ratio = 0.5);

// Save graph to file
void saveGraphToFile(const Graph& graph, const std::string& filename);

// Save changes to file
void saveChangesToFile(const std::vector<EdgeChange>& changes, 
                       const std::string& filename);

// Load changes from file
std::vector<EdgeChange> loadChangesFromFile(const std::string& filename);

// Print SSSP tree
void printSSSP(const std::vector<SSSPNode>& sssp);

// Verify SSSP solution against Dijkstra's algorithm
bool verifySSSP(const Graph& graph, VertexID source, 
                const std::vector<SSSPNode>& sssp);

// Write metrics to file
void writeMetricsToFile(const Metrics& metrics, const std::string& filename);

// Generate visualization of SSSP tree (DOT format)
void generateSSSPVisualization(const Graph& graph, const std::vector<SSSPNode>& sssp, 
                              VertexID source, const std::string& filename);

#endif // UTILS_H
