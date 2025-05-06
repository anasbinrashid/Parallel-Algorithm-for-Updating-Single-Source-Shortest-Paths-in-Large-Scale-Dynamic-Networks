/**
 * temporal_graph.h
 * Temporal graph representation and operations
 */

#ifndef TEMPORAL_GRAPH_H
#define TEMPORAL_GRAPH_H

#include "types.h"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <queue>
#include <map>
#include <set>

// TemporalGraph class
class TemporalGraph {
private:
    int num_vertices;
    std::vector<std::vector<TimedEdge>> temporal_edges;  // All timed edges for each vertex
    std::vector<std::vector<Edge>> snapshot_edges;       // Current snapshot edges
    Timestamp current_time;                              // Current time point
    
public:
    // Constructor with number of vertices
    TemporalGraph(int n);
    
    // Constructor from file
    TemporalGraph(const std::string& filename);
    
    // Get number of vertices
    int getNumVertices() const;
    
    // Get current timestamp
    Timestamp getCurrentTime() const;
    
    // Set current time and update snapshot
    void setCurrentTime(Timestamp time);
    
    // Get adjacency list for a vertex in current snapshot
    const std::vector<Edge>& getNeighbors(VertexID v) const;
    
    // Get all temporal edges for a vertex
    const std::vector<TimedEdge>& getTemporalEdges(VertexID v) const;
    
    // Add timed edge
    void addTimedEdge(VertexID source, VertexID target, Weight weight, 
                     Timestamp start_time, Timestamp end_time = INF);
    
    // Remove timed edge
    bool removeTimedEdge(VertexID source, VertexID target, Timestamp time);
    
    // Check if edge exists at current time
    bool hasEdge(VertexID source, VertexID target) const;
    
    // Get edge weight at current time
    Weight getEdgeWeight(VertexID source, VertexID target) const;
    
    // Apply changes to the temporal graph
    void applyChanges(const std::vector<EdgeChange>& changes);
    
    // Get all time points where graph structure changes
    std::vector<Timestamp> getAllTimepoints() const;
    
    // Update snapshot to reflect current time
    void updateSnapshot();
    
    // Print graph at current time
    void print() const;
    
    // Print temporal graph with all timestamps
    void printTemporal() const;
    
    // Compute initial SSSP using Dijkstra's algorithm for current snapshot
    std::vector<SSSPNode> computeInitialSSSP(VertexID source) const;
};

#endif // TEMPORAL_GRAPH_H
