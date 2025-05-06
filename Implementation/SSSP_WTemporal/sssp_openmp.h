/**
 * sssp_openmp.h
 * OpenMP implementation of SSSP update algorithm
 */

#ifndef SSSP_OPENMP_H
#define SSSP_OPENMP_H

#include "types.h"
#include "graph.h"
#include <vector>

class SSSPOpenMP {
private:
    Graph& graph;
    VertexID source;
    std::vector<SSSPNode> sssp_tree;
    int num_threads;
    
    // Process changes to identify affected vertices (Step 1)
    void processChanges(const std::vector<EdgeChange>& changes);
    
    // Process deletions
    void processDeletions(const std::vector<EdgeChange>& deletions);
    
    // Recursively mark all descendants of affected vertices
    void markDescendantsAsAffected(const std::vector<VertexID>& affected_vertices);
    
    // Process insertions
    void processInsertions(const std::vector<EdgeChange>& insertions);
    
    // Recalculate SSP from scratch using Dijkstra's algorithm
    void recalculateSSP();
    
    // Update affected vertices (Step 2)
    void updateAffectedVertices();
    
    // Update distances of affected vertices
    void updateDistances();
    
public:
    // Constructor
    SSSPOpenMP(Graph& g, VertexID src, int threads = 4);
    
    // Initialize SSSP tree
    void initialize();
    
    // Reset SSSP tree to correct state using Dijkstra's algorithm
    void resetToCorrectTree();
    
    // Update SSSP tree with changes
    Metrics update(const std::vector<EdgeChange>& changes);
    
    // Get the updated SSSP tree
    const std::vector<SSSPNode>& getSSSPTree() const;
    
    // Print the SSSP tree
    void printTree() const;
};

#endif // SSSP_OPENMP_H
