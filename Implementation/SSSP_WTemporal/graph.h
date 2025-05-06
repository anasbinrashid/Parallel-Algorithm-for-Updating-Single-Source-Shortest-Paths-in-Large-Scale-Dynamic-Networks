/**
 * graph.h
 * Graph representation and operations
 */

#ifndef GRAPH_H
#define GRAPH_H

#include "types.h"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <queue>

// Graph class
class Graph {
private:
    int num_vertices;
    std::vector<std::vector<Edge>> adjacency_list;
    
public:
    // Constructor with number of vertices
    Graph(int n);
    
    // Constructor from file
    Graph(const std::string& filename);
    
    // Get number of vertices
    int getNumVertices() const;
    
    // Get adjacency list for a vertex
    const std::vector<Edge>& getNeighbors(VertexID v) const;
    
    // Add edge
    void addEdge(VertexID source, VertexID target, Weight weight);
    
    // Remove edge
    bool removeEdge(VertexID source, VertexID target);
    
    // Check if edge exists
    bool hasEdge(VertexID source, VertexID target) const;
    
    // Get edge weight
    Weight getEdgeWeight(VertexID source, VertexID target) const;
    
    // Apply changes to the graph
    void applyChanges(const std::vector<EdgeChange>& changes);
    
    // Print graph
    void print() const;
    
    // Compute initial SSSP using Dijkstra's algorithm
    std::vector<SSSPNode> computeInitialSSSP(VertexID source) const;
};

#endif // GRAPH_H
