/**
 * sssp_temporal.h
 * OpenMP implementation of SSSP update algorithm for temporal graphs
 */

#ifndef SSSP_TEMPORAL_H
#define SSSP_TEMPORAL_H

#include "types.h"
#include "temporal_graph.h"
#include <vector>
#include <map>

class SSSPTemporal {
private:
    TemporalGraph& graph;
    VertexID source;
    std::map<Timestamp, std::vector<SSSPNode>> sssp_trees;  // SSSP trees at different timestamps
    int num_threads;
    
    // Process changes to identify affected vertices (Step 1)
    void processChanges(const std::vector<EdgeChange>& changes, Timestamp time);
    
    // Process deletions
    void processDeletions(const std::vector<EdgeChange>& deletions, Timestamp time);
    
    // Process insertions
    void processInsertions(const std::vector<EdgeChange>& insertions, Timestamp time);
    
    // Update affected vertices (Step 2)
    void updateAffectedVertices(Timestamp time);
    
    // Process deletion-affected subtrees
    void processAffectedSubtrees(Timestamp time);
    
    // Update distances of affected vertices
    void updateDistances(Timestamp time);
    
public:
    // Constructor
    SSSPTemporal(TemporalGraph& g, VertexID src, int threads = 4);
    
    // Initialize SSSP tree at time 0
    void initialize();
    
    // Update SSSP tree for a specific timestamp
    Metrics updateAtTime(Timestamp time);
    
    // Update SSSP tree with changes
    Metrics update(const std::vector<EdgeChange>& changes);
    
    // Update SSSP trees for all timepoints
    std::vector<Metrics> updateAllTimepoints();
    
    // Get the SSSP tree at a specific time
    const std::vector<SSSPNode>& getSSSPTreeAtTime(Timestamp time) const;
    
    // Get all SSSP trees
    const std::map<Timestamp, std::vector<SSSPNode>>& getAllSSSPTrees() const;
    
    // Print the SSSP tree at a specific time
    void printTree(Timestamp time) const;
    
    // Print all SSSP trees
    void printAllTrees() const;
};

#endif // SSSP_TEMPORAL_H
