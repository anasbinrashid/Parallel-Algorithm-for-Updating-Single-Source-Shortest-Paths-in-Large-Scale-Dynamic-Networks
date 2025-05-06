/**
 * temporal_graph.cpp
 * Implementation of temporal graph operations
 */

#include "temporal_graph.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <set>

// Constructor with number of vertices
TemporalGraph::TemporalGraph(int n) : num_vertices(n), current_time(0.0) {
    temporal_edges.resize(n);
    snapshot_edges.resize(n);
}

// Constructor from file
TemporalGraph::TemporalGraph(const std::string& filename) : current_time(0.0) {
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
    
    // Initialize arrays
    temporal_edges.resize(num_vertices);
    snapshot_edges.resize(num_vertices);
    
    // Read edges
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        VertexID source, target;
        Weight weight;
        Timestamp start_time, end_time = INF;
        
        // Try to read with timestamp(s)
        if (!(iss >> source >> target >> weight >> start_time)) {
            continue;  // Skip malformed lines
        }
        
        // Check if end time is provided
        double temp;
        if (iss >> temp) {
            end_time = temp;
        }
        
        if (source >= 0 && source < num_vertices && 
            target >= 0 && target < num_vertices) {
            addTimedEdge(source, target, weight, start_time, end_time);
        } else {
            std::cerr << "Warning: Invalid edge: " << source << " -> " 
                      << target << " (weight: " << weight << ")" << std::endl;
        }
    }
    
    file.close();
    
    // Update snapshot for time 0
    updateSnapshot();
}

// Get number of vertices
int TemporalGraph::getNumVertices() const {
    return num_vertices;
}

// Get current timestamp
Timestamp TemporalGraph::getCurrentTime() const {
    return current_time;
}

// Set current time and update snapshot
void TemporalGraph::setCurrentTime(Timestamp time) {
    current_time = time;
    updateSnapshot();
}

// Get adjacency list for a vertex in current snapshot
const std::vector<Edge>& TemporalGraph::getNeighbors(VertexID v) const {
    return snapshot_edges[v];
}

// Get all temporal edges for a vertex
const std::vector<TimedEdge>& TemporalGraph::getTemporalEdges(VertexID v) const {
    return temporal_edges[v];
}

// Add timed edge
void TemporalGraph::addTimedEdge(VertexID source, VertexID target, Weight weight, 
                                Timestamp start_time, Timestamp end_time) {
    // Add to temporal edges
    temporal_edges[source].push_back(TimedEdge(source, target, weight, start_time, end_time));
    
    // Update snapshot if edge is active at current time
    if (start_time <= current_time && (end_time == INF || current_time < end_time)) {
        snapshot_edges[source].push_back(Edge(target, weight));
    }
}

// Remove timed edge
bool TemporalGraph::removeTimedEdge(VertexID source, VertexID target, Timestamp time) {
    bool removed = false;
    
    // Set end time to current time for matching edges
    for (auto& edge : temporal_edges[source]) {
        if (edge.target == target && edge.existsAt(time)) {
            edge.end_time = time;
            removed = true;
        }
    }
    
    // Update snapshot if needed
    if (removed && time <= current_time) {
        updateSnapshot();
    }
    
    return removed;
}

// Check if edge exists at current time
bool TemporalGraph::hasEdge(VertexID source, VertexID target) const {
    for (const auto& edge : snapshot_edges[source]) {
        if (edge.first == target) {
            return true;
        }
    }
    return false;
}

// Get edge weight at current time
Weight TemporalGraph::getEdgeWeight(VertexID source, VertexID target) const {
    for (const auto& edge : snapshot_edges[source]) {
        if (edge.first == target) {
            return edge.second;
        }
    }
    return INF;  // Edge doesn't exist
}

// Apply changes to the temporal graph
void TemporalGraph::applyChanges(const std::vector<EdgeChange>& changes) {
    for (const auto& change : changes) {
        if (change.type == INSERTION) {
            addTimedEdge(change.source, change.target, change.weight, change.time);
        } else {  // DELETION
            removeTimedEdge(change.source, change.target, change.time);
        }
    }
}

// Get all time points where graph structure changes
std::vector<Timestamp> TemporalGraph::getAllTimepoints() const {
    std::set<Timestamp> timepoints;
    
    // Add all start and end times
    for (VertexID v = 0; v < num_vertices; ++v) {
        for (const auto& edge : temporal_edges[v]) {
            timepoints.insert(edge.start_time);
            if (edge.end_time != INF) {
                timepoints.insert(edge.end_time);
            }
        }
    }
    
    return std::vector<Timestamp>(timepoints.begin(), timepoints.end());
}

// Update snapshot to reflect current time
void TemporalGraph::updateSnapshot() {
    // Clear current snapshot
    for (auto& neighbors : snapshot_edges) {
        neighbors.clear();
    }
    
    // Add edges that exist at current time
    for (VertexID v = 0; v < num_vertices; ++v) {
        for (const auto& edge : temporal_edges[v]) {
            if (edge.existsAt(current_time)) {
                snapshot_edges[v].push_back(Edge(edge.target, edge.weight));
            }
        }
    }
}

// Print graph at current time
void TemporalGraph::print() const {
    std::cout << "Graph snapshot at time " << current_time << " with " 
              << num_vertices << " vertices:" << std::endl;
    
    for (VertexID v = 0; v < num_vertices; ++v) {
        std::cout << "Vertex " << v << ": ";
        for (const auto& edge : snapshot_edges[v]) {
            std::cout << "(" << edge.first << ", " << edge.second << ") ";
        }
        std::cout << std::endl;
    }
}

// Print temporal graph with all timestamps
void TemporalGraph::printTemporal() const {
    std::cout << "Temporal Graph with " << num_vertices << " vertices:" << std::endl;
    
    for (VertexID v = 0; v < num_vertices; ++v) {
        std::cout << "Vertex " << v << ":" << std::endl;
        for (const auto& edge : temporal_edges[v]) {
            std::cout << "  -> " << edge.target << " (weight: " << edge.weight 
                      << ", time: [" << edge.start_time;
            
            if (edge.end_time == INF) {
                std::cout << ", INF";
            } else {
                std::cout << ", " << edge.end_time;
            }
            
            std::cout << "]) " << std::endl;
        }
    }
}

// Compute initial SSSP using Dijkstra's algorithm for current snapshot
std::vector<SSSPNode> TemporalGraph::computeInitialSSSP(VertexID source) const {
    std::vector<SSSPNode> sssp(num_vertices);
    
    // Initialize distances
    for (VertexID v = 0; v < num_vertices; ++v) {
        sssp[v].distance = INF;
        sssp[v].parent = -1;
        sssp[v].time = current_time;
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
        for (const auto& edge : snapshot_edges[u]) {
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
