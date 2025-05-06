#ifndef SEQUENTIAL_SSSP_H
#define SEQUENTIAL_SSSP_H

#include "graph.h"
#include <vector>

// Sequential implementation of SSSP updating algorithm
class SequentialSSSP {
public:
    // Update SSSP tree when edges change
    static void updateSSSP(const Graph& g, SSSPTree& tree, const std::vector<EdgeChange>& changes);

private:
    // Step 1: Identify vertices affected by changes
    static void identifyAffectedVertices(const Graph& g, SSSPTree& tree, const std::vector<EdgeChange>& changes);
    
    // Step 2: Update affected subgraphs
    static void updateAffectedSubgraphs(const Graph& g, SSSPTree& tree);
    
    // Process edge deletion
    static void processEdgeDeletion(const Graph& g, SSSPTree& tree, const EdgeChange& edge);
    
    // Process edge insertion
    static void processEdgeInsertion(const Graph& g, SSSPTree& tree, const EdgeChange& edge);
    
    // Update disconnected subtree
    static void updateDisconnectedSubtree(const Graph& g, SSSPTree& tree, int vertex);
};

#endif // SEQUENTIAL_SSSP_H
