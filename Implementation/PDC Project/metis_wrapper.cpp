#include "metis_wrapper.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>

std::vector<int> MetisWrapper::partitionGraph(const Graph& g, int nparts) {
    int numVertices = g.getNumVertices();
    
    if (numVertices == 0) {
        std::cerr << "Error: Cannot partition empty graph" << std::endl;
        return std::vector<int>();
    }
    
    // Ensure at least one vertex per partition
    nparts = std::min(nparts, numVertices); 
    
    // Convert graph to CSR format for METIS
    std::vector<idx_t> xadj, adjncy, adjwgt;
    graphToCSR(g, xadj, adjncy, adjwgt);
    
    // METIS parameters
    idx_t nvtxs = numVertices;
    idx_t ncon = 1;  // Number of balancing constraints
    idx_t objval;    // Stores the edge-cut or communication volume
    
    // Allocate memory for partition vector
    std::vector<idx_t> metisPartition(numVertices);
    
    // Options for METIS partitioning
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;    // Minimize edge-cut
    options[METIS_OPTION_NUMBERING] = 0;                  // C-style numbering (0-based)
    options[METIS_OPTION_SEED] = 1;                       // Fixed seed for reproducibility
    options[METIS_OPTION_MINCONN] = 1;                    // Minimize maximum connectivity
    options[METIS_OPTION_CONTIG] = 0;                     // Don't force contiguous partitions - key change
    
    // Use METIS_PartGraphKway for all graphs - simplifies code and ensures consistent approach
    int ret = METIS_PartGraphKway(
        &nvtxs,      // Number of vertices
        &ncon,       // Number of constraints
        xadj.data(), // Adjacency structure
        adjncy.data(), // Adjacency list
        NULL,        // Vertex weights (NULL = all 1's)
        NULL,        // Size of vertices for computing communication volume
        adjwgt.data(), // Edge weights
        &nparts,     // Number of parts
        NULL,        // Target partition weights
        NULL,        // Constraints weights
        options,     // Options
        &objval,     // Output: edge-cut or communication volume
        metisPartition.data() // Output: partition vector
    );
    
    if (ret != METIS_OK) {
        std::cerr << "METIS partitioning failed with error code " << ret << std::endl;
        
        // Retry with default options only
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_NUMBERING] = 0;
        
        ret = METIS_PartGraphKway(
            &nvtxs, &ncon, xadj.data(), adjncy.data(), 
            NULL, NULL, adjwgt.data(), &nparts, 
            NULL, NULL, options, &objval, metisPartition.data()
        );
        
        if (ret != METIS_OK) {
            std::cerr << "METIS fallback partitioning failed with error code " << ret << std::endl;
            // Return basic sequential partition
            std::vector<int> simplePartition(numVertices);
            for (int i = 0; i < numVertices; i++) {
                simplePartition[i] = i % nparts;
            }
            return simplePartition;
        }
    }
    
    // Convert idx_t to int
    std::vector<int> partition(numVertices);
    for (int i = 0; i < numVertices; i++) {
        partition[i] = static_cast<int>(metisPartition[i]);
    }
    
    // Count vertices per partition for logging
    std::vector<int> partSizes(nparts, 0);
    for (int p : partition) {
        if (p >= 0 && p < nparts) {
            partSizes[p]++;
        }
    }
    
    // Calculate edge-cut to verify METIS results
    int edgeCut = 0;
    for (int i = 0; i < numVertices; i++) {
        int originalVertex = g.reverseMapVertex(i);
        int partID = partition[i];
        
        for (const auto& edge : g.getNeighbors(originalVertex)) {
            int neighborIdx = edge.first;
            if (neighborIdx < partition.size() && partition[neighborIdx] != partID) {
                edgeCut++;
            }
        }
    }
    
    std::cout << "Graph partitioned into " << nparts << " parts" << std::endl;
    std::cout << "Manual edge-cut calculation: " << edgeCut << std::endl;
    std::cout << "METIS reported edge-cut: " << objval << std::endl;
    std::cout << "Partition sizes: ";
    for (int i = 0; i < nparts; i++) {
        std::cout << "P" << i << "=" << partSizes[i] << " ";
    }
    std::cout << std::endl;
    
    return partition;
}

void MetisWrapper::graphToCSR(const Graph& g, std::vector<idx_t>& xadj, std::vector<idx_t>& adjncy, std::vector<idx_t>& adjwgt) {
    int numVertices = g.getNumVertices();
    
    // Initialize xadj with index of where each vertex's adjacency list begins
    xadj.resize(numVertices + 1);
    xadj[0] = 0;
    
    // Count total number of edges for pre-allocating
    size_t totalEdges = 0;
    for (int i = 0; i < numVertices; i++) {
        int originalVertex = g.reverseMapVertex(i);
        totalEdges += g.getNeighbors(originalVertex).size();
    }
    
    // Pre-allocate memory for adjncy and adjwgt
    adjncy.resize(totalEdges);
    adjwgt.resize(totalEdges);
    
    // Fill adjacency list and weights
    size_t edgeIndex = 0;
    for (int i = 0; i < numVertices; i++) {
        int originalVertex = g.reverseMapVertex(i);
        const auto& neighbors = g.getNeighbors(originalVertex);
        
        for (const auto& edge : neighbors) {
            // Ensure we're using proper vertex indices for METIS
            int mappedNeighbor = edge.first;
            
            // Bounds check to prevent segfaults
            if (mappedNeighbor >= 0 && mappedNeighbor < numVertices) {
                adjncy[edgeIndex] = mappedNeighbor;
                
                // Convert float weight to int (METIS requires int weights)
                // Scale by a reasonable factor to prevent overflow
                float scaledWeight = edge.second * 100.0f;
                adjwgt[edgeIndex] = static_cast<idx_t>(std::max(1.0f, std::round(scaledWeight)));
                
                edgeIndex++;
            }
        }
        
        xadj[i + 1] = edgeIndex;
    }
    
    // Resize in case we filtered out any edges
    if (edgeIndex < totalEdges) {
        adjncy.resize(edgeIndex);
        adjwgt.resize(edgeIndex);
    }
    
    // Validate that the CSR representation is proper
    for (size_t i = 0; i < numVertices; i++) {
        if (xadj[i] > xadj[i+1]) {
            std::cerr << "Error: Invalid CSR format - xadj not monotonically increasing at vertex " << i << std::endl;
            // Fix the issue
            xadj[i+1] = xadj[i];
        }
    }
    
    for (size_t i = 0; i < edgeIndex; i++) {
        if (adjncy[i] >= numVertices) {
            std::cerr << "Error: Invalid CSR format - adjncy index " << i << " out of range: " << adjncy[i] << std::endl;
            // Fix by assigning a valid vertex index
            adjncy[i] = 0;
        }
        if (adjwgt[i] <= 0) {
            // Ensure minimum weight of 1 for METIS
            adjwgt[i] = 1;
        }
    }
}

std::vector<int> MetisWrapper::getGhostVertices(const Graph& g, const std::vector<int>& partitions, int partID) {
    if (partitions.empty() || partID < 0) {
        return std::vector<int>();
    }
    
    std::unordered_set<int> ghostSet;
    int numVertices = g.getNumVertices();
    
    // Track connection importance
    std::unordered_map<int, int> connectionCount;
    std::unordered_map<int, float> connectionWeight;
    
    // Iterate through vertices in this partition
    for (int i = 0; i < numVertices; i++) {
        if (i < partitions.size() && partitions[i] == partID) {
            int originalVertex = g.reverseMapVertex(i);
            
            // Check neighbors
            for (const auto& edge : g.getNeighbors(originalVertex)) {
                int neighborIdx = edge.first;
                
                // If neighbor is in a different partition, it's a ghost vertex
                if (neighborIdx < partitions.size() && partitions[neighborIdx] != partID) {
                    // Track connection strength
                    connectionCount[neighborIdx]++;
                    connectionWeight[neighborIdx] += 1.0f / (edge.second + 0.1f);  // Higher weight for shorter edges
                    ghostSet.insert(neighborIdx);
                }
            }
        }
    }
    
    // If we have too many ghost vertices, prioritize the most important ones
    if (ghostSet.size() > 0.25 * numVertices) {
        // Score each ghost vertex by connection strength
        std::vector<std::pair<int, float>> rankedGhosts;
        for (int v : ghostSet) {
            float score = connectionCount[v] * connectionWeight[v];
            rankedGhosts.push_back({v, score});
        }
        
        // Sort by importance score (highest first)
        std::sort(rankedGhosts.begin(), rankedGhosts.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Limit to maximum 25% of total vertices
        int maxGhosts = std::max(10, static_cast<int>(0.25 * numVertices));
        
        std::vector<int> filteredGhosts;
        filteredGhosts.reserve(std::min(maxGhosts, static_cast<int>(rankedGhosts.size())));
        
        for (size_t i = 0; i < std::min(static_cast<size_t>(maxGhosts), rankedGhosts.size()); i++) {
            filteredGhosts.push_back(rankedGhosts[i].first);
        }
        
        std::cout << "Limited ghost vertices from " << ghostSet.size() << " to " << filteredGhosts.size() 
                  << " for partition " << partID << std::endl;
        
        return filteredGhosts;
    }
    
    // Convert set to vector
    std::vector<int> ghostVertices(ghostSet.begin(), ghostSet.end());
    return ghostVertices;
}

std::vector<int> MetisWrapper::getPartitionVertices(const Graph& g, const std::vector<int>& partitions, int partID) {
    if (partitions.empty() || partID < 0) {
        return std::vector<int>();
    }
    
    std::vector<int> vertices;
    int numVertices = std::min(g.getNumVertices(), static_cast<int>(partitions.size()));
    
    for (int i = 0; i < numVertices; i++) {
        if (partitions[i] == partID) {
            vertices.push_back(i);
        }
    }
    
    return vertices;
}

Graph MetisWrapper::createLocalGraph(const Graph& g, const std::vector<int>& partitions, int partID) {
    if (partitions.empty() || partID < 0) {
        std::cerr << "Error: Invalid partition information" << std::endl;
        return Graph();
    }
    
    // Get vertices in this partition and ghost vertices
    std::vector<int> localVertices = getPartitionVertices(g, partitions, partID);
    std::vector<int> ghostVertices = getGhostVertices(g, partitions, partID);
    
    if (localVertices.empty()) {
        std::cerr << "Warning: Partition " << partID << " is empty" << std::endl;
    }
    
    // Create lookup set for faster vertex membership checking
    std::unordered_set<int> allVerticesSet;
    for (int v : localVertices) {
        allVerticesSet.insert(v);
    }
    for (int v : ghostVertices) {
        allVerticesSet.insert(v);
    }
    
    // Create subgraph
    Graph localGraph;
    
    // First add all local vertices (always include all of these)
    for (int v : localVertices) {
        int originalVertex = g.reverseMapVertex(v);
        
        for (const auto& edge : g.getNeighbors(originalVertex)) {
            int targetIdx = edge.first;
            
            // Add edge if target is also in the local graph
            if (allVerticesSet.count(targetIdx) > 0) {
                int targetOriginal = g.reverseMapVertex(targetIdx);
                localGraph.addEdge(originalVertex, targetOriginal, edge.second);
            }
        }
    }
    
    // Then add ghost vertices, but only their edges to local vertices
    // This prevents creating unnecessary connections between ghost vertices
    for (int v : ghostVertices) {
        int originalVertex = g.reverseMapVertex(v);
        bool hasLocalConnection = false;
        
        for (const auto& edge : g.getNeighbors(originalVertex)) {
            int targetIdx = edge.first;
            
            // Only add edge if target is in the local partition (not just any ghost)
            if (targetIdx < partitions.size() && partitions[targetIdx] == partID) {
                int targetOriginal = g.reverseMapVertex(targetIdx);
                localGraph.addEdge(originalVertex, targetOriginal, edge.second);
                hasLocalConnection = true;
            }
        }
        
        // If ghost vertex has no connections to local partition, skip it entirely
        if (!hasLocalConnection) {
            allVerticesSet.erase(v);
        }
    }
    
    // Count how many ghost vertices were actually kept after filtering
    int keptGhostCount = 0;
    for (int v : ghostVertices) {
        if (allVerticesSet.count(v) > 0) {
            keptGhostCount++;
        }
    }
    
    std::cout << "Created optimized local graph for partition " << partID 
              << " with " << localGraph.getNumVertices() << " vertices ("
              << localVertices.size() << " local, " << keptGhostCount << " essential ghost) and "
              << localGraph.getNumEdges() << " edges" << std::endl;
    
    return localGraph;
}
