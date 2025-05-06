/**
 * sssp_openmp.cpp
 * Implementation of OpenMP-based SSSP update algorithm
 */

#include "sssp_openmp.h"
#include "utils.h"
#include <omp.h>
#include <algorithm>
#include <iostream>
#include <queue>
#include <set>

// Constructor
SSSPOpenMP::SSSPOpenMP(Graph& g, VertexID src, int threads)
    : graph(g), source(src), num_threads(threads) {
    // Set number of threads for OpenMP
    omp_set_num_threads(num_threads);
}

// Initialize SSSP tree
void SSSPOpenMP::initialize() {
    sssp_tree = graph.computeInitialSSSP(source);
}

// Reset SSSP tree to correct state
void SSSPOpenMP::resetToCorrectTree() {
    sssp_tree = graph.computeInitialSSSP(source);
}

// Process changes to identify affected vertices (Step 1)
void SSSPOpenMP::processChanges(const std::vector<EdgeChange>& changes) {
    // Flag to track if we need a complete recalculation
    bool need_complete_recalc = false;
    
    // Separate changes into deletions and insertions
    std::vector<EdgeChange> deletions;
    std::vector<EdgeChange> insertions;
    
    for (const auto& change : changes) {
        if (change.type == DELETION) {
            deletions.push_back(change);
        } else {
            insertions.push_back(change);
        }
    }
    
    // Process deletions first
    if (!deletions.empty()) {
        processDeletions(deletions);
    }
    
    // Process insertions
    if (!insertions.empty()) {
        processInsertions(insertions);
    }
    
    // If we have both types of changes, we'll need more processing
    if (!deletions.empty() && !insertions.empty()) {
        need_complete_recalc = true;
    }
    
    // If we detected a need for complete recalculation
    if (need_complete_recalc) {
        // We'll do a complete recalculation from scratch
        // This ensures correctness when complex edge changes happen
        recalculateSSP();
    }
}

// Process deletions
void SSSPOpenMP::processDeletions(const std::vector<EdgeChange>& deletions) {
    // First identify directly affected vertices
    std::vector<VertexID> directly_affected;
    
    for (const auto& change : deletions) {
        VertexID u = change.source;
        VertexID v = change.target;
        
        // Check if the edge is part of the SSSP tree
        if (sssp_tree[v].parent == u) {
            directly_affected.push_back(v);  // v is child of u
        } else if (sssp_tree[u].parent == v) {
            directly_affected.push_back(u);  // u is child of v
        }
    }
    
    // Mark directly affected vertices
    for (VertexID v : directly_affected) {
        sssp_tree[v].affected_del = true;
        sssp_tree[v].affected = true;
        sssp_tree[v].parent = -1;
        sssp_tree[v].distance = INF;
    }
    
    // Now recursively mark all descendants of affected vertices
    markDescendantsAsAffected(directly_affected);
}

// Recursively mark all descendants of affected vertices
void SSSPOpenMP::markDescendantsAsAffected(const std::vector<VertexID>& affected_vertices) {
    if (affected_vertices.empty()) {
        return;
    }
    
    // Convert affected_vertices to a set for faster lookup
    std::set<VertexID> affected_set(affected_vertices.begin(), affected_vertices.end());
    
    // Find all vertices whose parent is in the affected set
    std::vector<VertexID> next_level_affected;
    
    for (VertexID v = 0; v < graph.getNumVertices(); ++v) {
        if (sssp_tree[v].distance != INF) {  // Skip already unreachable vertices
            VertexID parent = sssp_tree[v].parent;
            
            if (parent != -1 && affected_set.find(parent) != affected_set.end()) {
                sssp_tree[v].distance = INF;
                sssp_tree[v].parent = -1;
                sssp_tree[v].affected_del = true;
                sssp_tree[v].affected = true;
                next_level_affected.push_back(v);
            }
        }
    }
    
    // Recursively mark next level
    if (!next_level_affected.empty()) {
        markDescendantsAsAffected(next_level_affected);
    }
}

// Recalculate SSP from scratch
void SSSPOpenMP::recalculateSSP() {
    // This is a simple way to ensure correctness
    // Recompute the entire SSSP tree using Dijkstra's algorithm
    sssp_tree = graph.computeInitialSSSP(source);
}

// Process insertions
void SSSPOpenMP::processInsertions(const std::vector<EdgeChange>& insertions) {
    bool complex_insertion = false;
    
    // First, check if any insertion potentially creates a shortcut in the graph
    for (const auto& change : insertions) {
        VertexID u = change.source;
        VertexID v = change.target;
        Weight weight = change.weight;
        
        // Both vertices must be reachable to detect a shortcut
        if (sssp_tree[u].distance != INF && sssp_tree[v].distance != INF) {
            // If this insertion creates a shorter path between already reachable vertices
            if (sssp_tree[u].distance + weight < sssp_tree[v].distance ||
                sssp_tree[v].distance + weight < sssp_tree[u].distance) {
                complex_insertion = true;
                break;
            }
        }
    }
    
    // If we detected a complex insertion, just recalculate the whole tree
    if (complex_insertion) {
        recalculateSSP();
        return;
    }
    
    // For simpler cases, we can just mark vertices as affected
    for (const auto& change : insertions) {
        VertexID u = change.source;
        VertexID v = change.target;
        Weight weight = change.weight;
        
        // If the source vertex is reachable, the target might be affected
        if (sssp_tree[u].distance != INF) {
            Weight potential_dist = sssp_tree[u].distance + weight;
            if (potential_dist < sssp_tree[v].distance) {
                sssp_tree[v].distance = potential_dist;
                sssp_tree[v].parent = u;
                sssp_tree[v].affected = true;
            }
        }
        
        // If the target vertex is reachable, the source might be affected
        if (sssp_tree[v].distance != INF) {
            Weight potential_dist = sssp_tree[v].distance + weight;
            if (potential_dist < sssp_tree[u].distance) {
                sssp_tree[u].distance = potential_dist;
                sssp_tree[u].parent = v;
                sssp_tree[u].affected = true;
            }
        }
    }
}

// Update affected vertices (Step 2)
void SSSPOpenMP::updateAffectedVertices() {
    // Update distances iteratively
    updateDistances();
}

// Update distances of affected vertices
void SSSPOpenMP::updateDistances() {
    bool affected_vertices_exist = true;
    int iterations = 0;
    const int max_iterations = graph.getNumVertices() * 2;  // Prevent infinite loops
    
    while (affected_vertices_exist && iterations < max_iterations) {
        affected_vertices_exist = false;
        iterations++;
        
        // Create a copy of the current SSSP tree
        std::vector<SSSPNode> new_sssp = sssp_tree;
        
        // Process all vertices in parallel
        #pragma omp parallel for schedule(dynamic)
        for (VertexID v = 0; v < graph.getNumVertices(); ++v) {
            if (sssp_tree[v].affected || iterations == 1) {
                // Reset affected flag
                new_sssp[v].affected = false;
                
                // If this vertex is reachable
                if (sssp_tree[v].distance != INF) {
                    // Check all outgoing edges
                    for (const auto& edge : graph.getNeighbors(v)) {
                        VertexID n = edge.first;
                        Weight weight = edge.second;
                        
                        // If we can provide a shorter path
                        Weight new_dist = sssp_tree[v].distance + weight;
                        if (new_dist < new_sssp[n].distance) {
                            #pragma omp critical
                            {
                                if (new_dist < new_sssp[n].distance) {
                                    new_sssp[n].distance = new_dist;
                                    new_sssp[n].parent = v;
                                    new_sssp[n].affected = true;
                                    affected_vertices_exist = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Update SSSP tree
        sssp_tree = new_sssp;
    }
}

// Update SSSP tree with changes
Metrics SSSPOpenMP::update(const std::vector<EdgeChange>& changes) {
    Metrics metrics;
    Timer timer;
    
    // Apply changes to the graph
    graph.applyChanges(changes);
    
    // Start timing
    timer.start();
    
    // Step 1: Process changes to identify affected vertices
    Timer step1_timer;
    step1_timer.start();
    processChanges(changes);
    step1_timer.stop();
    metrics.step1_time = step1_timer.getElapsedTimeInSeconds();
    
    // Step 2: Update affected vertices
    Timer step2_timer;
    step2_timer.start();
    updateAffectedVertices();
    step2_timer.stop();
    metrics.step2_time = step2_timer.getElapsedTimeInSeconds();
    
    // Count affected vertices
    int affected_count = 0;
    for (const auto& node : sssp_tree) {
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

// Get the updated SSSP tree
const std::vector<SSSPNode>& SSSPOpenMP::getSSSPTree() const {
    return sssp_tree;
}

// Print the SSSP tree
void SSSPOpenMP::printTree() const {
    printSSSP(sssp_tree);
}
