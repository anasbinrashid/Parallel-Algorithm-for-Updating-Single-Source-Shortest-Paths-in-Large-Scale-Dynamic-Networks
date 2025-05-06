/**
 * utils_temporal.cpp
 * Implementation of utility functions for temporal graphs
 */

#include "utils_temporal.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <set>

// In utils_temporal.h, ensure this is defined:
/*
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
};*/

// Save temporal graph to file
void saveTemporalGraphToFile(const TemporalGraph& graph, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    file << graph.getNumVertices() << std::endl;
    
    for (VertexID v = 0; v < graph.getNumVertices(); ++v) {
        for (const auto& edge : graph.getTemporalEdges(v)) {
            file << edge.source << " " << edge.target << " " << edge.weight << " " 
                 << edge.start_time;
            
            if (edge.end_time != INF) {
                file << " " << edge.end_time;
            }
            
            file << std::endl;
        }
    }
    
    file.close();
}

// Save edge changes to file
void saveEdgeChangesToFile(const std::vector<EdgeChange>& changes, 
                         const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    file << changes.size() << std::endl;
    
    for (const auto& change : changes) {
        file << change.source << " " << change.target << " " << change.weight << " "
             << (change.type == INSERTION ? "+" : "-") << " " << change.time << std::endl;
    }
    
    file.close();
}

// Load edge changes from file
std::vector<EdgeChange> loadEdgeChangesFromFile(const std::string& filename) {
    std::vector<EdgeChange> changes;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return changes;
    }
    
    std::string line;
    // Read number of changes
    std::getline(file, line);
    // We can use this to pre-reserve space in the vector
    int num_changes = std::stoi(line);
    changes.reserve(num_changes);
    
    // Read changes
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        VertexID source, target;
        Weight weight;
        char type_char;
        Timestamp time;
        
        if (!(iss >> source >> target >> weight >> type_char >> time)) {
            continue;  // Skip malformed lines
        }
        
        ChangeType type = (type_char == '+') ? INSERTION : DELETION;
        changes.push_back(EdgeChange(source, target, weight, type, time));
    }
    
    file.close();
    return changes;
}

// Verify SSSP solution against Dijkstra's algorithm for temporal graph
bool verifyTemporalSSSP(const TemporalGraph& graph, VertexID source, 
                       const std::vector<SSSPNode>& sssp) {
    // Compute SSSP using Dijkstra's algorithm
    std::vector<SSSPNode> dijkstra_sssp = graph.computeInitialSSSP(source);
    
    // Compare distances
    for (size_t v = 0; v < sssp.size(); ++v) {
        if (std::abs(sssp[v].distance - dijkstra_sssp[v].distance) > 1e-6 && 
            !(sssp[v].distance == INF && dijkstra_sssp[v].distance == INF)) {
            std::cerr << "Verification failed for vertex " << v << ": "
                      << "Expected distance " << dijkstra_sssp[v].distance
                      << ", got " << sssp[v].distance << std::endl;
            return false;
        }
    }
    
    return true;
}

// Generate visualization of SSSP tree (DOT format)
void generateSSSPVisualization(const TemporalGraph& graph, const std::vector<SSSPNode>& sssp, 
                             VertexID source, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    Timestamp time = sssp[0].time;  // Assumes all nodes have the same timestamp
    
    file << "digraph SSSP {" << std::endl;
    file << "  graph [rankdir=LR];" << std::endl;
    file << "  node [shape=circle];" << std::endl;
    file << "  label=\"SSSP Tree at time " << time << "\";" << std::endl;
    
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

// Generate temporal SSSP visualization
void generateTemporalSSSPVisualization(const TemporalGraph& graph, 
                                     const std::vector<SSSPNode>& sssp, 
                                     VertexID source, Timestamp time,
                                     const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    file << "digraph SSSP {" << std::endl;
    file << "  graph [rankdir=LR];" << std::endl;
    file << "  node [shape=circle];" << std::endl;
    file << "  label=\"SSSP Tree at time " << time << "\";" << std::endl;
    
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

// Generate temporal evolution animation script (for GraphViz)
void generateTemporalEvolutionAnimation(const TemporalGraph& original_graph,
                                      const std::map<Timestamp, std::vector<SSSPNode>>& sssp_trees,
                                      VertexID source,
                                      const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    // Make a non-const copy that we can modify
    TemporalGraph graph = original_graph;
    
    file << "#!/bin/bash" << std::endl;
    file << "# Script to generate a temporal evolution animation of SSSP trees" << std::endl;
    file << std::endl;
    
    // Generate DOT files for each timepoint
    for (const auto& entry : sssp_trees) {
        Timestamp time = entry.first;
        std::string dot_filename = "sssp_time_" + std::to_string(time) + ".dot";
        std::string png_filename = "sssp_time_" + std::to_string(time) + ".png";
        
        file << "# Generate visualization for time " << time << std::endl;
        file << "cat > " << dot_filename << " << 'EOT'" << std::endl;
        file << "digraph SSSP {" << std::endl;
        file << "  graph [rankdir=LR];" << std::endl;
        file << "  node [shape=circle];" << std::endl;
        file << "  label=\"SSSP Tree at time " << time << "\";" << std::endl;
        
        // Set graph to this timepoint
        graph.setCurrentTime(time);
        
        // Source node with special styling
        file << "  " << source << " [fillcolor=green, style=filled, label=\"" << source << " (0.0)\"];" << std::endl;
        
        // Other nodes
        const auto& sssp = entry.second;
        for (VertexID v = 0; v < graph.getNumVertices(); ++v) {
            if (v != source) {
                if (sssp[v].distance == INF) {
                    file << "  " << v << " [fillcolor=red, style=filled, label=\"" << v << " (INF)\"];" << std::endl;
                } else {
                    file << "  " << v << " [label=\"" << v << " (" << sssp[v].distance << ")\"];" << std::endl;
                }
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
        file << "EOT" << std::endl;
        
        // Convert DOT to PNG
        file << "dot -Tpng " << dot_filename << " -o " << png_filename << std::endl;
        file << std::endl;
    }
    
    // Create GIF animation
    file << "# Create GIF animation" << std::endl;
    file << "convert -delay 100 -loop 0 sssp_time_*.png sssp_evolution.gif" << std::endl;
    file << std::endl;
    
    // Clean up
    file << "# Clean up temporary files" << std::endl;
    file << "# rm sssp_time_*.dot" << std::endl;
    
    file.close();
    
    // Make the script executable
    std::string chmod_cmd = "chmod +x " + filename;
    int result = system(chmod_cmd.c_str());
    (void)result; // Explicitly ignore the return value
    
    std::cout << "Animation script generated in " << filename << std::endl;
    std::cout << "Run './" << filename << "' to generate the animation" << std::endl;
}

// Generate a random temporal graph for testing
TemporalGraph generateRandomTemporalGraph(int num_vertices, int num_initial_edges, 
                                       int num_timepoints, int changes_per_timepoint) {
    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> vertex_dist(0, num_vertices - 1);
    std::uniform_real_distribution<> weight_dist(0.1, 10.0);
    
    // Create graph
    TemporalGraph graph(num_vertices);
    
    // Set to keep track of added edges
    std::set<std::pair<VertexID, VertexID>> added_edges;
    
    // Add initial edges at time 0
    int edges_added = 0;
    while (edges_added < num_initial_edges) {
        VertexID u = vertex_dist(gen);
        VertexID v = vertex_dist(gen);
        
        // Avoid self-loops and duplicates
        if (u != v && !added_edges.count({u, v})) {
            Weight weight = std::round(weight_dist(gen) * 10) / 10;  // Round to 1 decimal place
            graph.addTimedEdge(u, v, weight, 0.0);
            added_edges.insert({u, v});
            edges_added++;
        }
    }
    
    // Generate changes for each timepoint
    for (int t = 1; t <= num_timepoints; t++) {
        Timestamp time = static_cast<double>(t);
        
        // Different ratios for each timepoint
        double insertion_ratio;
        if (t <= num_timepoints / 3) {
            insertion_ratio = 0.8;  // Mostly insertions in early timepoints
        } else if (t <= 2 * num_timepoints / 3) {
            insertion_ratio = 0.5;  // Balanced changes in middle timepoints
        } else {
            insertion_ratio = 0.2;  // Mostly deletions in late timepoints
        }
        
        // Generate changes for this timepoint
        for (int i = 0; i < changes_per_timepoint; i++) {
            if (std::uniform_real_distribution<>(0.0, 1.0)(gen) < insertion_ratio) {
                // Insertion
                VertexID u, v;
                bool found = false;
                
                // Try to find an edge that doesn't exist
                int attempts = 0;
                while (!found && attempts < 100) {
                    u = vertex_dist(gen);
                    v = vertex_dist(gen);
                    
                    if (u != v && !added_edges.count({u, v})) {
                        Weight weight = std::round(weight_dist(gen) * 10) / 10;
                        graph.addTimedEdge(u, v, weight, time);
                        added_edges.insert({u, v});
                        found = true;
                    }
                    
                    attempts++;
                }
            } else {
                // Deletion
                if (!added_edges.empty()) {
                    // Select a random existing edge
                    int edge_idx = std::uniform_int_distribution<>(0, added_edges.size() - 1)(gen);
                    auto it = added_edges.begin();
                    std::advance(it, edge_idx);
                    
                    VertexID u = it->first;
                    VertexID v = it->second;
                    
                    // Delete edge
                    graph.removeTimedEdge(u, v, time);
                    added_edges.erase(it);
                }
            }
        }
    }
    
    return graph;
}
