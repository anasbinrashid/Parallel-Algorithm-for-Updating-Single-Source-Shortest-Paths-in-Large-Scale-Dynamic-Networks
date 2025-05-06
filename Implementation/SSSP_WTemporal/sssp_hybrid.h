/**
 * sssp_hybrid.h
 * Hybrid OpenMP+MPI+METIS implementation of SSSP update algorithm
 */

#ifndef SSSP_HYBRID_H
#define SSSP_HYBRID_H

#include "types.h"
#include "graph.h"
#include <vector>
#include <map>
#include <mpi.h>

class SSSPHybrid {
private:
    Graph& graph;
    VertexID source;
    std::vector<SSSPNode> sssp_tree;
    int num_threads;  // OpenMP threads per process
    int rank;         // MPI rank
    int size;         // Number of MPI processes
    
    // Partition information
    std::vector<int> vertex_to_partition;  // Maps vertex to its partition ID
    std::vector<std::vector<VertexID>> partition_vertices;  // Lists vertices in each partition
    std::vector<std::set<VertexID>> boundary_vertices;  // Boundary vertices for each partition
    
    // Partition the graph using METIS
    void partitionGraph();
    
    // Process changes to identify affected vertices (Step 1)
    void processChanges(const std::vector<EdgeChange>& changes);
    
    // Process deletions
    void processDeletions(const std::vector<EdgeChange>& deletions);
    
    // Process insertions
    void processInsertions(const std::vector<EdgeChange>& insertions);
    
    // Update affected vertices (Step 2)
    void updateAffectedVertices();
    
    // Process deletion-affected subtrees
    void processAffectedSubtrees();
    
    // Update distances of affected vertices
    void updateDistances();
    
    // Synchronize SSSP tree across processes
    void synchronizeSSPTree();
    
    // Get affected vertices in partition
    std::vector<VertexID> getAffectedVerticesInPartition();
    
public:
    // Constructor
    SSSPHybrid(Graph& g, VertexID src, int threads = 4);
    
    // Destructor
    ~SSSPHybrid();
    
    // Initialize SSSP tree
    void initialize();
    
    // Update SSSP tree with changes
    Metrics update(const std::vector<EdgeChange>& changes);
    
    // Get the updated SSSP tree
    const std::vector<SSSPNode>& getSSSPTree() const;
    
    // Print the SSSP tree
    void printTree() const;
};

#endif // SSSP_HYBRID_H
