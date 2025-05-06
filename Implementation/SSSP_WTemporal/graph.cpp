/**
 * graph.cpp
 * Implementation of graph operations
 */

#include "graph.h"

// Constructor with number of vertices
Graph::Graph(int n) : num_vertices(n) {
    adjacency_list.resize(n);
}

// Constructor from file
Graph::Graph(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }
    
    std::string line;
    // Read first line for number of vertices
    std::getline(file, line);
    std::istringstream iss(line);
    iss >> num_vertices;
    
    // Initialize adjacency list
    adjacency_list.resize(num_vertices);
    
    // Read edges
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        VertexID source, target;
        Weight weight;
        
        if (!(iss >> source >> target >> weight)) {
            continue;  // Skip malformed lines
        }
        
        if (source >= 0 && source < num_vertices && 
            target >= 0 && target < num_vertices) {
            addEdge(source, target, weight);
        } else {
            std::cerr << "Warning: Invalid edge: " << source << " -> " 
                      << target << " (weight: " << weight << ")" << std::endl;
        }
    }
    
    file.close();
}

// Get number of vertices
int Graph::getNumVertices() const {
    return num_vertices;
}

// Get adjacency list for a vertex
const std::vector<Edge>& Graph::getNeighbors(VertexID v) const {
    return adjacency_list[v];
}

// Add edge
void Graph::addEdge(VertexID source, VertexID target, Weight weight) {
    // Check if edge already exists
    for (auto& edge : adjacency_list[source]) {
        if (edge.first == target) {
            edge.second = weight;  // Update weight
            return;
        }
    }
    
    // Add new edge
    adjacency_list[source].push_back(Edge(target, weight));
}

// Remove edge
bool Graph::removeEdge(VertexID source, VertexID target) {
    auto& neighbors = adjacency_list[source];
    for (auto it = neighbors.begin(); it != neighbors.end(); ++it) {
        if (it->first == target) {
            neighbors.erase(it);
            return true;
        }
    }
    return false;
}

// Check if edge exists
bool Graph::hasEdge(VertexID source, VertexID target) const {
    const auto& neighbors = adjacency_list[source];
    for (const auto& edge : neighbors) {
        if (edge.first == target) {
            return true;
        }
    }
    return false;
}

// Get edge weight
Weight Graph::getEdgeWeight(VertexID source, VertexID target) const {
    const auto& neighbors = adjacency_list[source];
    for (const auto& edge : neighbors) {
        if (edge.first == target) {
            return edge.second;
        }
    }
    return INF;  // Edge doesn't exist
}

// Apply changes to the graph
void Graph::applyChanges(const std::vector<EdgeChange>& changes) {
    for (const auto& change : changes) {
        if (change.type == INSERTION) {
            addEdge(change.source, change.target, change.weight);
        } else {  // DELETION
            removeEdge(change.source, change.target);
        }
    }
}

// Print graph
void Graph::print() const {
    std::cout << "Graph with " << num_vertices << " vertices:" << std::endl;
    for (VertexID v = 0; v < num_vertices; ++v) {
        std::cout << "Vertex " << v << ": ";
        for (const auto& edge : adjacency_list[v]) {
            std::cout << "(" << edge.first << ", " << edge.second << ") ";
        }
        std::cout << std::endl;
    }
}

// Compute initial SSSP using Dijkstra's algorithm
std::vector<SSSPNode> Graph::computeInitialSSSP(VertexID source) const {
    std::vector<SSSPNode> sssp(num_vertices);
    
    // Initialize distances
    for (VertexID v = 0; v < num_vertices; ++v) {
        sssp[v].distance = INF;
        sssp[v].parent = -1;
    }
    
    // Distance to source is 0
    sssp[source].distance = 0;
    
    // Priority queue for Dijkstra's algorithm
    // Pair of (distance, vertex)
    std::priority_queue<std::pair<Weight, VertexID>, 
                        std::vector<std::pair<Weight, VertexID>>, 
                        std::greater<std::pair<Weight, VertexID>>> pq;
    
    pq.push(std::make_pair(0, source));
    
    while (!pq.empty()) {
        Weight dist = pq.top().first;
        VertexID u = pq.top().second;
        pq.pop();
        
        // Skip if we've already found a better path
        if (dist > sssp[u].distance) {
            continue;
        }
        
        // Explore neighbors
        for (const auto& edge : adjacency_list[u]) {
            VertexID v = edge.first;
            Weight weight = edge.second;
            
            // If we find a shorter path
            if (sssp[u].distance + weight < sssp[v].distance) {
                sssp[v].distance = sssp[u].distance + weight;
                sssp[v].parent = u;
                pq.push(std::make_pair(sssp[v].distance, v));
            }
        }
    }
    
    return sssp;
}
