#include "metis_wrapper.h"
#include <iostream>

std::vector<idx_t> partition_graph(const Graph& graph, int num_parts) {
    int n = graph.get_num_vertices();
    
    
    
    if (num_parts <= 1) {
        std::vector<idx_t> part(n, 0);  // Assign all vertices to partition 0
        std::cout << "Single MPI process: all vertices assigned to one partition." << std::endl;
        return part;
    }
    
    std::vector<idx_t> xadj(n + 1, 0);
    std::vector<idx_t> adjncy;
    std::vector<idx_t> eweights;
    
    // Count edges for each vertex to build CSR format
    for (int i = 0; i < n; i++) {
        const auto& neighbors = graph.get_neighbors(i);
        xadj[i + 1] = xadj[i] + neighbors.size();
        
        for (const auto& edge : neighbors) {
            adjncy.push_back(edge.first);
            eweights.push_back(static_cast<idx_t>(edge.second * 1));  // Convert to integer weight
        }
    }
    
    // Prepare METIS parameters
    idx_t nvtxs = n;
    idx_t ncon = 1;  // Number of balancing constraints
    idx_t nparts = num_parts;
    idx_t objval = 0;  // Stores edge-cut or total communication volume
    
    std::vector<idx_t> part(n);
    
    // Set options
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;  // C-style (0-based) numbering
    
    // Call METIS to partition the graph
    int ret = METIS_PartGraphKway(
        &nvtxs,     // Number of vertices
        &ncon,      // Number of balancing constraints
        xadj.data(),// Adjacency structure: index to adjncy
        adjncy.data(), // Adjacency structure: neighbors
        NULL,       // Vertex weights (NULL = all 1)
        NULL,       // Size of vertices for total communication volume
        eweights.data(), // Edge weights
        &nparts,    // Number of parts to partition into
        NULL,       // Target weight for each partition and constraint
        NULL,       // Allowed load imbalance
        options,    // Options array
        &objval,    // Output: the edge-cut or total communication volume
        part.data() // Output: partition vector
    );
    
    if (ret != METIS_OK) {
        std::cerr << "METIS partitioning failed with error: " << ret << std::endl;
        exit(1);
    }
    
    std::cout << "METIS partitioning successful. Edge cut: " << objval << std::endl;
    
    return part;
}
