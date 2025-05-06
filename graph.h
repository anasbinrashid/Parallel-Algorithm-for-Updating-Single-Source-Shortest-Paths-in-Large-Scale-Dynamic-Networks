#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <iostream>
#include <limits>
#include <algorithm>
#include <queue>

// Structure representing an edge change
struct EdgeChange {
    int source;     // Source vertex
    int target;     // Target vertex
    float weight;   // Edge weight
    bool isInsert;  // true for insert, false for delete
    
    EdgeChange() : source(-1), target(-1), weight(0.0f), isInsert(true) {}
    
    EdgeChange(int s, int t, float w, bool insert) 
        : source(s), target(t), weight(w), isInsert(insert) {}
};

// Graph class using adjacency list representation
class Graph {
private:
    int numVertices;
    std::vector<std::vector<std::pair<int, float>>> adjacencyList; // (vertex, weight)
    std::unordered_map<int, int> vertexMap;    // Maps original vertex IDs to internal indices
    std::unordered_map<int, int> reverseMap;   // Maps internal indices to original vertex IDs

public:
    // Constructor
    Graph();
    
    // Copy constructor
    Graph(const Graph& other);
    
    // Load graph from edge file
    bool loadFromEdgeFile(const std::string& filename);
    
    // Add edge
    void addEdge(int u, int v, float weight);
    
    // Remove edge
    void removeEdge(int u, int v);
    
    // Check if edge exists
    bool hasEdge(int u, int v) const;
    
    // Get edge weight
    float getEdgeWeight(int u, int v) const;
    
    // Get number of vertices
    int getNumVertices() const;
    
    // Get number of edges
    int getNumEdges() const;
    
    // Get neighbors of a vertex
    const std::vector<std::pair<int, float>>& getNeighbors(int v) const;
    
    // Get all vertices
    std::vector<int> getAllVertices() const;
    
    // Map vertex ID (external to internal)
    int mapVertex(int v) const;
    
    // Reverse map (internal to external)
    int reverseMapVertex(int v) const;
    
    // Get vertex map
    const std::unordered_map<int, int>& getVertexMap() const;
    
    // Get reverse map
    const std::unordered_map<int, int>& getReverseMap() const;
    
    // Create a subgraph based on vertex partition
    Graph createSubgraph(const std::vector<int>& vertices);
    
    // Print graph
    void printGraph() const;
};

// Single-Source Shortest Path Tree structure
class SSSPTree {
public:
    int sourceVertex;                // Source vertex ID
    int numVertices;                 // Number of vertices
    std::vector<int> parent;         // Parent in SSSP tree
    std::vector<float> distance;     // Distance from source
    std::vector<bool> affectedDel;   // Affected by deletion
    std::vector<bool> affected;      // Affected by any change
    
    // Constructor
    SSSPTree(int numVerts = 0, int source = 0);
    
    // Initialize SSSP tree using Dijkstra's algorithm
    void initialize(const Graph& g, int source);
    
    // Get source vertex
    int getSource() const;
    
    // Check if tree is valid
    bool isValid() const;
    
    // Print SSSP tree
    void printTree(const Graph& g) const;
};

#endif // GRAPH_H
