#ifndef METIS_WRAPPER_H
#define METIS_WRAPPER_H

#include "graph.h"
#include <metis.h>
#include <vector>
#include <unordered_set>

// Wrapper class for METIS graph partitioning
class MetisWrapper {
public:
    // Partition a graph into 'nparts' parts
    static std::vector<int> partitionGraph(const Graph& g, int nparts);
    
    // Get ghost vertices for a partition
    static std::vector<int> getGhostVertices(const Graph& g, const std::vector<int>& partitions, int partID);
    
    // Get vertices in a partition
    static std::vector<int> getPartitionVertices(const Graph& g, const std::vector<int>& partitions, int partID);
    
    // Create a local subgraph for a partition (including ghost vertices)
    static Graph createLocalGraph(const Graph& g, const std::vector<int>& partitions, int partID);

private:
    // Convert graph to CSR format for METIS
    static void graphToCSR(const Graph& g, std::vector<idx_t>& xadj, std::vector<idx_t>& adjncy, 
                          std::vector<idx_t>& adjwgt);
};

#endif // METIS_WRAPPER_H
