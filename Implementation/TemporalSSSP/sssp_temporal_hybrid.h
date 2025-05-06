/**
 * sssp_temporal_hybrid.h
 * Hybrid OpenMP+MPI+METIS implementation of SSSP update algorithm for temporal graphs
 */

#ifndef SSSP_TEMPORAL_HYBRID_H
#define SSSP_TEMPORAL_HYBRID_H

#include "types.h"
#include "temporal_graph.h"
#include <vector>
#include <map>
#include <set>
#include <mpi.h>

class SSSPTemporalHybrid {
private:
    TemporalGraph& graph;
    VertexID source;
    std::map<Timestamp, std::vector<SSSPNode>> sssp_trees;  // SSSP trees at different timestamps
    int num_threads;  // OpenMP threads per process
    int rank;         // MPI rank
    int size;         // Number of MPI processes
    
    // Partition information
    std::vector<int> vertex_to_partition;  // Maps vertex to its partition ID
    std::vector<std::vector<VertexID>> partition_vertices;  // Lists vertices in each partition
    std::vector<std::set<VertexID>> boundary_vertices;  // Boundary vertices for each partition
    
    // Partition the graph using METIS (or a simple partitioning if METIS fails)
    void partitionGraph();
    
    // Simple partitioning method (fallback if METIS fails)
    void simplePartition();
    
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
    
    // Synchronize SSSP tree across processes for a specific time
    void synchronizeSSPTree(Timestamp time);
    
    // Get affected vertices in partition at a specific time
    std::vector<VertexID> getAffectedVerticesInPartition(Timestamp time);
    
    // Create safe access to SSSP tree at time (initialize if not exists)
    std::vector<SSSPNode>& getOrCreateSSSPTree(Timestamp time);
    
public:
    // Constructor
    SSSPTemporalHybrid(TemporalGraph& g, VertexID src, int threads = 4);
    
    // Destructor
    ~SSSPTemporalHybrid();
    
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

#endif // SSSP_TEMPORAL_HYBRID_H
