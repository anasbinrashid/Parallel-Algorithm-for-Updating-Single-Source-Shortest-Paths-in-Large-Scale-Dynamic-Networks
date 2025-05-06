/**
 * sssp_temporal.cpp
 * Implementation of temporal SSSP update algorithm
 */

#include "sssp_temporal.h"
#include <omp.h>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <queue>
#include <chrono>
#include "utils_temporal.h"
#include "types.h"

// Timer class for measuring execution time
/*
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    bool running;

public:
    Timer() : running(false) {}

    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        running = true;
    }

    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
        running = false;
    }

    double getElapsedTimeInSeconds() const {
        if (running) {
            auto current_time = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double>(current_time - start_time).count();
        }
        return std::chrono::duration<double>(end_time - start_time).count();
    }
};*/


// Constructor
SSSPTemporal::SSSPTemporal(TemporalGraph& g, VertexID src, int threads)
    : graph(g), source(src), num_threads(threads) {
    // Set number of threads for OpenMP
    omp_set_num_threads(num_threads);
}

// Initialize SSSP tree at time 0
void SSSPTemporal::initialize() {
    // Set graph time to 0
    graph.setCurrentTime(0.0);
    
    // Compute initial SSSP
    std::vector<SSSPNode> initial_sssp = graph.computeInitialSSSP(source);
    
    // Store SSSP tree for time 0
    sssp_trees[0.0] = initial_sssp;
}

// Process changes to identify affected vertices (Step 1)
void SSSPTemporal::processChanges(const std::vector<EdgeChange>& changes, Timestamp time) {
    // Separate changes into deletions and insertions
    std::vector<EdgeChange> deletions;
    std::vector<EdgeChange> insertions;
    
    for (const auto& change : changes) {
        if (change.time == time) {
            if (change.type == DELETION) {
                deletions.push_back(change);
            } else {
                insertions.push_back(change);
            }
        }
    }
    
    // Process deletions first
    processDeletions(deletions, time);
    
    // Then process insertions
    processInsertions(insertions, time);
}

// Process deletions
void SSSPTemporal::processDeletions(const std::vector<EdgeChange>& deletions, Timestamp time) {
    // Find the closest earlier time that has an SSSP tree
    Timestamp prev_time = 0.0;
    for (const auto& entry : sssp_trees) {
        if (entry.first < time && entry.first > prev_time) {
            prev_time = entry.first;
        }
    }
    
    // Copy SSSP tree from previous time as starting point
    sssp_trees[time] = sssp_trees[prev_time];
    auto& sssp_tree = sssp_trees[time];
    
    // Update time in each node
    for (auto& node : sssp_tree) {
        node.time = time;
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < deletions.size(); ++i) {
        const auto& change = deletions[i];
        VertexID u = change.source;
        VertexID v = change.target;
        
        // Check if the edge is part of the SSSP tree
        if (sssp_tree[v].parent == u || sssp_tree[u].parent == v) {
            VertexID affected_vertex;
            
            if (sssp_tree[v].parent == u) {
                affected_vertex = v;  // v is child of u
            } else {
                affected_vertex = u;  // u is child of v
            }
            
            // Mark as affected by deletion
            sssp_tree[affected_vertex].affected_del = true;
            sssp_tree[affected_vertex].affected = true;
            
            // Disconnect from parent
            sssp_tree[affected_vertex].parent = -1;
            sssp_tree[affected_vertex].distance = INF;
        }
    }
}

// Process insertions
void SSSPTemporal::processInsertions(const std::vector<EdgeChange>& insertions, Timestamp time) {
    auto& sssp_tree = sssp_trees[time];
    
    #pragma omp parallel for
    for (size_t i = 0; i < insertions.size(); ++i) {
        const auto& change = insertions[i];
        VertexID u = change.source;
        VertexID v = change.target;
        Weight weight = change.weight;
        
        // Determine vertices with known and unknown distances
        VertexID known_vertex, unknown_vertex;
        bool update_possible = false;
        
        // If both vertices have known distances, check which path is shorter
        if (sssp_tree[u].distance != INF && sssp_tree[v].distance != INF) {
            // Check if path through new edge is shorter for either vertex
            if (sssp_tree[u].distance + weight < sssp_tree[v].distance) {
                known_vertex = u;
                unknown_vertex = v;
                update_possible = true;
            } else if (sssp_tree[v].distance + weight < sssp_tree[u].distance) {
                known_vertex = v;
                unknown_vertex = u;
                update_possible = true;
            }
        } 
        // If only one vertex has a known distance, it can potentially update the other
        else if (sssp_tree[u].distance != INF) {
            known_vertex = u;
            unknown_vertex = v;
            update_possible = true;
        } else if (sssp_tree[v].distance != INF) {
            known_vertex = v;
            unknown_vertex = u;
            update_possible = true;
        }
        
        // Update if possible
        if (update_possible) {
            Weight new_distance = sssp_tree[known_vertex].distance + weight;
            
            // Only update if it improves the distance
            if (new_distance < sssp_tree[unknown_vertex].distance) {
                #pragma omp critical
                {
                    if (new_distance < sssp_tree[unknown_vertex].distance) {
                        sssp_tree[unknown_vertex].distance = new_distance;
                        sssp_tree[unknown_vertex].parent = known_vertex;
                        sssp_tree[unknown_vertex].affected = true;
                    }
                }
            }
        }
    }
}

// Update affected vertices (Step 2)
void SSSPTemporal::updateAffectedVertices(Timestamp time) {
    // First process deletion-affected subtrees
    processAffectedSubtrees(time);
    
    // Then update distances iteratively
    updateDistances(time);
}

// Process deletion-affected subtrees
void SSSPTemporal::processAffectedSubtrees(Timestamp time) {
    auto& sssp_tree = sssp_trees[time];
    bool affected_vertices_exist = true;
    
    while (affected_vertices_exist) {
        affected_vertices_exist = false;
        
        #pragma omp parallel for schedule(dynamic)
        for (VertexID v = 0; v < graph.getNumVertices(); ++v) {
            if (sssp_tree[v].affected_del) {
                sssp_tree[v].affected_del = false;
                
                // Process all children
                for (VertexID c = 0; c < graph.getNumVertices(); ++c) {
                    if (sssp_tree[c].parent == v) {
                        sssp_tree[c].distance = INF;
                        sssp_tree[c].parent = -1;
                        sssp_tree[c].affected_del = true;
                        sssp_tree[c].affected = true;
                        affected_vertices_exist = true;
                    }
                }
            }
        }
        
        // Synchronize threads after each iteration
        #pragma omp barrier
    }
}

// Update distances of affected vertices
void SSSPTemporal::updateDistances(Timestamp time) {
    auto& sssp_tree = sssp_trees[time];
    bool affected_vertices_exist = true;
    int iterations = 0;
    
    while (affected_vertices_exist) {
        affected_vertices_exist = false;
        iterations++;
        
        #pragma omp parallel for schedule(dynamic)
        for (VertexID v = 0; v < graph.getNumVertices(); ++v) {
            if (sssp_tree[v].affected) {
                // Reset affected flag
                sssp_tree[v].affected = false;
                
                // Check all neighbors in current graph snapshot
                for (const auto& edge : graph.getNeighbors(v)) {
                    VertexID n = edge.first;
                    Weight weight = edge.second;
                    
                    // If neighbor can provide a shorter path to v
                    if (sssp_tree[n].distance != INF && 
                        sssp_tree[n].distance + weight < sssp_tree[v].distance) {
                        sssp_tree[v].distance = sssp_tree[n].distance + weight;
                        sssp_tree[v].parent = n;
                        sssp_tree[v].affected = true;
                        affected_vertices_exist = true;
                    }
                    
                    // If v can provide a shorter path to neighbor
                    if (sssp_tree[v].distance != INF && 
                        sssp_tree[v].distance + weight < sssp_tree[n].distance) {
                        #pragma omp critical
                        {
                            if (sssp_tree[v].distance + weight < sssp_tree[n].distance) {
                                sssp_tree[n].distance = sssp_tree[v].distance + weight;
                                sssp_tree[n].parent = v;
                                sssp_tree[n].affected = true;
                                affected_vertices_exist = true;
                            }
                        }
                    }
                }
            }
        }
        
        // Synchronize threads after each iteration
        #pragma omp barrier
    }
}

// Update SSSP tree for a specific timestamp
Metrics SSSPTemporal::updateAtTime(Timestamp time) {
    Metrics metrics;
    Timer timer;
    
    // Update graph to the specified time
    graph.setCurrentTime(time);
    
    // Start timing
    timer.start();
    
    // Find all changes that happened at this time
    std::vector<EdgeChange> changes_at_time;
    
    // Step 1: Process changes to identify affected vertices
    Timer step1_timer;
    step1_timer.start();
    processChanges(changes_at_time, time);
    step1_timer.stop();
    metrics.step1_time = step1_timer.getElapsedTimeInSeconds();
    
    // Step 2: Update affected vertices
    Timer step2_timer;
    step2_timer.start();
    updateAffectedVertices(time);
    step2_timer.stop();
    metrics.step2_time = step2_timer.getElapsedTimeInSeconds();
    
    // Count affected vertices
    int affected_count = 0;
    for (const auto& node : sssp_trees[time]) {
        if (node.affected || node.affected_del) {
            affected_count++;
        }
    }
    metrics.affected_vertices = affected_count;
    
    // Stop timing
    timer.stop();
    metrics.total_time = timer.getElapsedTimeInSeconds();
    
    return metrics;
}

// Update SSSP tree with changes
Metrics SSSPTemporal::update(const std::vector<EdgeChange>& changes) {
    Metrics metrics;
    Timer timer;
    
    // Apply changes to the graph
    graph.applyChanges(changes);
    
    // Get all timepoints where changes occur
    std::set<Timestamp> timepoints;
    for (const auto& change : changes) {
        timepoints.insert(change.time);
    }
    
    // Start timing
    timer.start();
    
    // For each timepoint, update the SSSP tree
    Timer step1_timer, step2_timer;
    int total_affected = 0;
    
    for (Timestamp time : timepoints) {
        // Set graph time to current timepoint
        graph.setCurrentTime(time);
        
        // Find changes at this timepoint
        std::vector<EdgeChange> changes_at_time;
        for (const auto& change : changes) {
            if (change.time == time) {
                changes_at_time.push_back(change);
            }
        }
        
        // Step 1: Process changes to identify affected vertices
        step1_timer.start();
        processChanges(changes_at_time, time);
        step1_timer.stop();
        
        // Step 2: Update affected vertices
        step2_timer.start();
        updateAffectedVertices(time);
        step2_timer.stop();
        
        // Count affected vertices
        int affected_count = 0;
        for (const auto& node : sssp_trees[time]) {
            if (node.affected || node.affected_del) {
                affected_count++;
            }
        }
        total_affected += affected_count;
    }
    
    // Stop timing
    timer.stop();
    
    // Populate metrics
    metrics.total_time = timer.getElapsedTimeInSeconds();
    metrics.step1_time = step1_timer.getElapsedTimeInSeconds();
    metrics.step2_time = step2_timer.getElapsedTimeInSeconds();
    metrics.affected_vertices = total_affected;
    
    return metrics;
}

// Update SSSP trees for all timepoints
std::vector<Metrics> SSSPTemporal::updateAllTimepoints() {
    // Get all timepoints in the temporal graph
    std::vector<Timestamp> timepoints = graph.getAllTimepoints();
    std::vector<Metrics> all_metrics;
    
    // Sort timepoints in ascending order
    std::sort(timepoints.begin(), timepoints.end());
    
    // Initialize SSSP tree at time 0 if not already done
    if (sssp_trees.empty()) {
        initialize();
    }
    
    // Update SSSP tree for each timepoint
    for (Timestamp time : timepoints) {
        if (time > 0 && sssp_trees.find(time) == sssp_trees.end()) {
            Metrics metrics = updateAtTime(time);
            all_metrics.push_back(metrics);
        }
    }
    
    return all_metrics;
}

// Get the SSSP tree at a specific time
const std::vector<SSSPNode>& SSSPTemporal::getSSSPTreeAtTime(Timestamp time) const {
    auto it = sssp_trees.find(time);
    if (it == sssp_trees.end()) {
        throw std::runtime_error("No SSSP tree available for time " + std::to_string(time));
    }
    return it->second;
}

// Get all SSSP trees
const std::map<Timestamp, std::vector<SSSPNode>>& SSSPTemporal::getAllSSSPTrees() const {
    return sssp_trees;
}

// Print the SSSP tree at a specific time
void SSSPTemporal::printTree(Timestamp time) const {
    auto it = sssp_trees.find(time);
    if (it == sssp_trees.end()) {
        std::cerr << "No SSSP tree available for time " << time << std::endl;
        return;
    }
    
    std::cout << "SSSP Tree at time " << time << ":" << std::endl;
    std::cout << std::setw(10) << "Vertex" << std::setw(10) << "Parent" 
              << std::setw(10) << "Distance" << std::endl;
    
    const auto& sssp_tree = it->second;
    for (size_t v = 0; v < sssp_tree.size(); ++v) {
        std::cout << std::setw(10) << v << std::setw(10) << sssp_tree[v].parent;
        
        if (sssp_tree[v].distance == INF) {
            std::cout << std::setw(10) << "INF" << std::endl;
        } else {
            std::cout << std::setw(10) << sssp_tree[v].distance << std::endl;
        }
    }
}

// Print all SSSP trees
void SSSPTemporal::printAllTrees() const {
    for (const auto& entry : sssp_trees) {
        printTree(entry.first);
        std::cout << std::endl;
    }
}
