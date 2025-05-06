#ifndef PARALLEL_SSSP_H
#define PARALLEL_SSSP_H

#include "graph.h"
#include "metis_wrapper.h"
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <unordered_map>
#include <atomic>
#include <functional>
#include <numeric>  

// Structure for exchanging boundary vertex information between MPI processes
struct BoundaryInfo {
    int vertexID;   // Global vertex ID
    float distance; // Distance from source
    int parent;     // Parent vertex ID (global)
    
    BoundaryInfo() : vertexID(-1), distance(0.0f), parent(-1) {}
    BoundaryInfo(int id, float dist, int p) : vertexID(id), distance(dist), parent(p) {}
};

// Parallel implementation of SSSP updating algorithm using MPI and OpenMP
class ParallelSSSP {
public:
    // Constructor
    ParallelSSSP(int rank, int size);
    
    // Initialize with graph, source vertex, and partitioning info
    void initialize(const Graph& globalGraph, int sourceVertex, int numThreads, int asyncLevel = 1);
    
    // Update SSSP when edges change
    void updateSSSP(const std::vector<EdgeChange>& changes);
    
    // Get results after computation
    void gatherResults(Graph& originalGraph, SSSPTree& globalTree);
    
    // Get local SSSP tree
    const SSSPTree& getLocalTree() const;
    
    // Get local graph
    const Graph& getLocalGraph() const;
    
    // Set asynchrony level for OpenMP updates
    void setAsynchronyLevel(int level);
    
    // Set maximum iterations
    void setMaxIterations(int max);

private:
    int rank;               // MPI rank
    int size;               // MPI size
    int sourceVertex;       // Global source vertex
    int numThreads;         // Number of OpenMP threads
    int asyncLevel;         // Level of asynchrony for updates
    int maxIterations;      // Maximum number of iterations
    bool verbose;           // Verbose output
    Graph localGraph;       // Local subgraph
    SSSPTree localTree;     // Local SSSP tree
    std::vector<int> partitions;     // Partitioning information
    std::vector<int> ghostVertices;  // Ghost (boundary) vertices
    std::unordered_map<int, int> globalToLocalMap; // Maps global vertex IDs to local IDs
    std::unordered_map<int, int> localToGlobalMap; // Maps local vertex IDs to global IDs
    
    // Map global vertex ID to local vertex ID
    int mapGlobalToLocal(int globalID) const;
    
    // Map local vertex ID to global vertex ID
    int mapLocalToGlobal(int localID) const;
    
    // Step 1: Identify vertices affected by changes (parallel)
    void identifyAffectedVertices(const std::vector<EdgeChange>& changes);
    
    // Step 2: Update affected subgraphs (parallel)
    void updateAffectedSubgraphs();
    
    // Process edge deletion
    void processEdgeDeletion(const EdgeChange& edge);
    
    // Process edge insertion
    void processEdgeInsertion(const EdgeChange& edge);
    
    // Update disconnected subtree
    void updateDisconnectedSubtree(int vertex);
    
    // Exchange boundary vertex information with other processes
    void exchangeBoundaryInfo();
    
    // Create MPI data type for BoundaryInfo
    MPI_Datatype createBoundaryInfoType();
    
    // Helper to log messages with rank info
    void log(const std::string& message);
};

#endif // PARALLEL_SSSP_H
