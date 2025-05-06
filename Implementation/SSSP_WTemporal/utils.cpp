/**
 * utils.cpp
 * Implementation of utility functions
 */

#include "utils.h"
#include <iostream>
#include <iomanip>
#include <set>
#include <algorithm>

// Timer methods
Timer::Timer() : running(false) {}

void Timer::start() {
    start_time = std::chrono::high_resolution_clock::now();
    running = true;
}

void Timer::stop() {
    end_time = std::chrono::high_resolution_clock::now();
    running = false;
}

double Timer::getElapsedTimeInSeconds() const {
    if (running) {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(current_time - start_time);
        return duration.count() / 1000000.0;
    } else {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000000.0;
    }
}

// Generate a random temporal graph
Graph generateTemporalGraph(int num_vertices, int num_initial_edges, 
                           int num_changes, double insertion_ratio) {
    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> vertex_dist(0, num_vertices - 1);
    std::uniform_real_distribution<> weight_dist(0.1, 10.0);  // Edge weights between 0.1 and 10.0
    std::uniform_real_distribution<> prob_dist(0.0, 1.0);     // Probability distribution
    
    // Create graph
    Graph graph(num_vertices);
    
    // Set to keep track of added edges
    std::set<std::pair<VertexID, VertexID>> added_edges;
    
    // Add initial edges
    int edges_added = 0;
    while (edges_added < num_initial_edges) {
        VertexID source = vertex_dist(gen);
        VertexID target = vertex_dist(gen);
        
        // Avoid self-loops and duplicates
        if (source != target && !added_edges.count({source, target})) {
            Weight weight = std::round(weight_dist(gen) * 10) / 10;  // Round to 1 decimal place
            graph.addEdge(source, target, weight);
            added_edges.insert({source, target});
            edges_added++;
        }
    }
    
    // Generate changes
    std::vector<EdgeChange> changes;
    
    for (int i = 0; i < num_changes; i++) {
        ChangeType type;
        
        // Determine change type based on insertion ratio
        if (prob_dist(gen) < insertion_ratio) {
            type = INSERTION;
            
            // For insertion, find an edge that doesn't exist
            VertexID source, target;
            bool found = false;
            
            while (!found) {
                source = vertex_dist(gen);
                target = vertex_dist(gen);
                
                if (source != target && !graph.hasEdge(source, target)) {
                    Weight weight = std::round(weight_dist(gen) * 10) / 10;
                    changes.push_back(EdgeChange(source, target, weight, type));
                    found = true;
                }
            }
        } else {
            type = DELETION;
            
            // For deletion, find an existing edge
            if (added_edges.empty()) {
                // No edges to delete, switch to insertion
                type = INSERTION;
                VertexID source = vertex_dist(gen);
                VertexID target = vertex_dist(gen);
                
                if (source != target && !graph.hasEdge(source, target)) {
                    Weight weight = std::round(weight_dist(gen) * 10) / 10;
                    changes.push_back(EdgeChange(source, target, weight, type));
                }
            } else {
                // Select a random existing edge
                int edge_idx = std::uniform_int_distribution<>(0, added_edges.size() - 1)(gen);
                auto it = added_edges.begin();
                std::advance(it, edge_idx);
                
                VertexID source = it->first;
                VertexID target = it->second;
                
                // Delete edge
                changes.push_back(EdgeChange(source, target, 0.0, type));
                added_edges.erase(it);
            }
        }
    }
    
    return graph;
}

// Save graph to file
void saveGraphToFile(const Graph& graph, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    file << graph.getNumVertices() << std::endl;
    
    for (VertexID v = 0; v < graph.getNumVertices(); ++v) {
        for (const auto& edge : graph.getNeighbors(v)) {
            file << v << " " << edge.first << " " << edge.second << std::endl;
        }
    }
    
    file.close();
}

// Save changes to file
void saveChangesToFile(const std::vector<EdgeChange>& changes, 
                       const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    file << changes.size() << std::endl;
    
    for (const auto& change : changes) {
        file << change.source << " " << change.target << " " << change.weight << " "
             << (change.type == INSERTION ? "+" : "-") << std::endl;
    }
    
    file.close();
}

// Load changes from file
std::vector<EdgeChange> loadChangesFromFile(const std::string& filename) {
    std::vector<EdgeChange> changes;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return changes;
    }
    
    std::string line;
    // Read number of changes
    std::getline(file, line);
    int num_changes = std::stoi(line);
    
    // Read changes
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        VertexID source, target;
        Weight weight;
        char type_char;
        
        if (!(iss >> source >> target >> weight >> type_char)) {
            continue;  // Skip malformed lines
        }
        
        ChangeType type = (type_char == '+') ? INSERTION : DELETION;
        changes.push_back(EdgeChange(source, target, weight, type));
    }
    
    file.close();
    return changes;
}

// Print SSSP tree
void printSSSP(const std::vector<SSSPNode>& sssp) {
    std::cout << "SSSP Tree:" << std::endl;
    std::cout << std::setw(10) << "Vertex" << std::setw(10) << "Parent" 
              << std::setw(10) << "Distance" << std::endl;
    
    for (size_t v = 0; v < sssp.size(); ++v) {
        std::cout << std::setw(10) << v << std::setw(10) << sssp[v].parent;
        
        if (sssp[v].distance == INF) {
            std::cout << std::setw(10) << "INF" << std::endl;
        } else {
            std::cout << std::setw(10) << sssp[v].distance << std::endl;
        }
    }
}

// Verify SSSP solution against Dijkstra's algorithm
bool verifySSSP(const Graph& graph, VertexID source, 
                const std::vector<SSSPNode>& sssp) {
    // Compute SSSP using Dijkstra's algorithm
    std::vector<SSSPNode> dijkstra_sssp = graph.computeInitialSSSP(source);
    
    // Compare distances
    for (size_t v = 0; v < sssp.size(); ++v) {
        if (sssp[v].distance != dijkstra_sssp[v].distance) {
            std::cerr << "Verification failed for vertex " << v << ": "
                      << "Expected distance " << dijkstra_sssp[v].distance
                      << ", got " << sssp[v].distance << std::endl;
            return false;
        }
    }
    
    return true;
}

// Write metrics to file
void writeMetricsToFile(const Metrics& metrics, const std::string& filename) {
    std::ofstream file(filename, std::ios::app);  // Append mode
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    file << metrics.total_time << "," 
         << metrics.step1_time << "," 
         << metrics.step2_time << "," 
         << metrics.affected_vertices << ","
         << metrics.iterations << std::endl;
    
    file.close();
}

// Generate visualization of SSSP tree (DOT format)
void generateSSSPVisualization(const Graph& graph, const std::vector<SSSPNode>& sssp, 
                              VertexID source, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    file << "digraph SSSP {" << std::endl;
    file << "  graph [rankdir=LR];" << std::endl;
    file << "  node [shape=circle];" << std::endl;
    
    // Source node with special styling
    file << "  " << source << " [fillcolor=green, style=filled, label=\"" << source << " (0.0)\"];" << std::endl;
    
    // Unreachable nodes
    for (VertexID v = 0; v < graph.getNumVertices(); ++v) {
        if (sssp[v].distance == INF) {
            file << "  " << v << " [fillcolor=red, style=filled, label=\"" << v << " (INF)\"];" << std::endl;
        } else if (v != source) {
            file << "  " << v << " [label=\"" << v << " (" << sssp[v].distance << ")\"];" << std::endl;
        }
    }
    
    // SSSP tree edges
    for (VertexID v = 0; v < graph.getNumVertices(); ++v) {
        if (v != source && sssp[v].parent != -1) {
            file << "  " << sssp[v].parent << " -> " << v 
                 << " [color=blue, penwidth=2.0, label=\"" << graph.getEdgeWeight(sssp[v].parent, v) << "\"];" << std::endl;
        }
    }
    
    // Non-tree edges
    for (VertexID v = 0; v < graph.getNumVertices(); ++v) {
        for (const auto& edge : graph.getNeighbors(v)) {
            VertexID u = edge.first;
            Weight w = edge.second;
            
            // Skip SSSP tree edges
            if (sssp[u].parent == v || sssp[v].parent == u) {
                continue;
            }
            
            file << "  " << v << " -> " << u << " [color=gray, label=\"" << w << "\"];" << std::endl;
        }
    }
    
    file << "}" << std::endl;
    file.close();
}
