/**
 * sssp_hybrid.cpp
 * Implementation of Hybrid OpenMP+MPI+METIS SSSP update algorithm
 */

#include "sssp_hybrid.h"
#include "utils.h"
#include <omp.h>
#include <metis.h>
#include <algorithm>
#include <iostream>
#include <queue>
#include <set>
#include <cmath> // For std::abs

// Constructor
SSSPHybrid::SSSPHybrid(Graph& g, VertexID src, int threads)
    : graph(g), source(src), num_threads(threads) {
    
    // Initialize MPI
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int provided;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
    }
    
    // Get rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Set number of threads for OpenMP
    omp_set_num_threads(num_threads);
    
    // Ensure all processes have correct initialization before partitioning
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Partition the graph
    partitionGraph();
}

// Destructor
SSSPHybrid::~SSSPHybrid() {
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }
}

// Partition the graph using METIS - with limits on number of partitions
void SSSPHybrid::partitionGraph() {
    int num_vertices = graph.getNumVertices();
    
    // Limit the number of partitions to at most 6 regardless of MPI size
    // This helps ensure stability when scaling to 1-10 processes
    int effective_partitions = std::min(6, size);
    
    // Initialize data structures for all processes
    vertex_to_partition.resize(num_vertices, 0);
    
    // Only perform partitioning on the master process
    if (rank == 0) {
        // Convert graph to CSR format for METIS
        std::vector<idx_t> xadj;
        std::vector<idx_t> adjncy;
        std::vector<idx_t> adjwgt;
        
        xadj.push_back(0);
        for (VertexID v = 0; v < num_vertices; ++v) {
            const auto& neighbors = graph.getNeighbors(v);
            for (const auto& edge : neighbors) {
                adjncy.push_back(edge.first);
                // Convert weights to integers for METIS (multiply by 10 to preserve 1 decimal place)
                adjwgt.push_back(static_cast<idx_t>(edge.second * 10));
            }
            xadj.push_back(xadj.back() + neighbors.size());
        }
        
        // Prepare METIS input
        idx_t nvtxs = num_vertices;
        idx_t ncon = 1;  // Number of balancing constraints
        idx_t nparts = effective_partitions;  // Limited number of partitions
        
        // Weights and partition information
        idx_t objval = 0;  // Stores the edge-cut
        std::vector<idx_t> part(num_vertices);  // Stores the partition assignment
        
        // METIS options
        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_NUMBERING] = 0;  // Use C-style numbering (starting from 0)
        
        // Call METIS to partition the graph
        int ret = METIS_PartGraphKway(
            &nvtxs,         // Number of vertices
            &ncon,          // Number of balancing constraints
            xadj.data(),    // Adjacency structure
            adjncy.data(),  // Adjacency information
            NULL,           // Vertex weights
            NULL,           // Vertex sizes
            adjwgt.data(),  // Edge weights
            &nparts,        // Number of partitions
            NULL,           // Target partition weights
            NULL,           // Allowed imbalance
            options,        // Options
            &objval,        // Output: Objective value (edge-cut)
            part.data()     // Output: Partition information
        );
        
        if (ret != METIS_OK) {
            std::cerr << "METIS partitioning failed with error code " << ret << " - using simple partitioning" << std::endl;
            // Fall back to simple partitioning
            for (VertexID v = 0; v < num_vertices; ++v) {
                part[v] = v % effective_partitions;
            }
        }
        
        // Initialize partition data structures
        partition_vertices.resize(effective_partitions);
        
        // Assign vertices to partitions
        for (VertexID v = 0; v < num_vertices; ++v) {
            // Map original partition to rank (for effective partitions < size)
            int mapped_part = part[v];
            vertex_to_partition[v] = mapped_part;
            
            // Safety check
            if (mapped_part >= 0 && mapped_part < effective_partitions) {
                partition_vertices[mapped_part].push_back(v);
            }
        }
        
        // Identify boundary vertices
        boundary_vertices.resize(effective_partitions);
        for (VertexID v = 0; v < num_vertices; ++v) {
            int p = vertex_to_partition[v];
            const auto& neighbors = graph.getNeighbors(v);
            
            for (const auto& edge : neighbors) {
                VertexID u = edge.first;
                int q = vertex_to_partition[u];
                
                if (p != q) {
                    boundary_vertices[p].insert(v);
                    break;
                }
            }
        }
    }
    
    // Broadcast partition information to all processes
    MPI_Bcast(vertex_to_partition.data(), num_vertices, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Broadcast the effective number of partitions
    MPI_Bcast(&effective_partitions, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Synchronize partition_vertices
    std::vector<int> partition_sizes(effective_partitions, 0);
    if (rank == 0) {
        for (int p = 0; p < effective_partitions; ++p) {
            partition_sizes[p] = partition_vertices[p].size();
        }
    }
    
    MPI_Bcast(partition_sizes.data(), effective_partitions, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        partition_vertices.resize(effective_partitions);
        for (int p = 0; p < effective_partitions; ++p) {
            partition_vertices[p].resize(partition_sizes[p]);
        }
    }
    
    for (int p = 0; p < effective_partitions; ++p) {
        if (partition_sizes[p] > 0) {
            MPI_Bcast(partition_vertices[p].data(), partition_sizes[p], MPI_INT, 0, MPI_COMM_WORLD);
        }
    }
    
    // Synchronize boundary_vertices - only if this process's rank is within the effective partitions
    std::vector<int> boundary_sizes(effective_partitions, 0);
    if (rank == 0) {
        for (int p = 0; p < effective_partitions; ++p) {
            boundary_sizes[p] = boundary_vertices[p].size();
        }
    }
    
    MPI_Bcast(boundary_sizes.data(), effective_partitions, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        boundary_vertices.resize(effective_partitions);
    }
    
    for (int p = 0; p < effective_partitions; ++p) {
        if (boundary_sizes[p] > 0) {
            std::vector<VertexID> boundary_array;
            if (rank == 0) {
                boundary_array.assign(boundary_vertices[p].begin(), boundary_vertices[p].end());
            } else {
                boundary_array.resize(boundary_sizes[p]);
            }
            
            MPI_Bcast(boundary_array.data(), boundary_sizes[p], MPI_INT, 0, MPI_COMM_WORLD);
            
            if (rank != 0) {
                boundary_vertices[p].clear();
                boundary_vertices[p].insert(boundary_array.begin(), boundary_array.end());
            }
        }
    }
    
    // Ensure only processes with valid partition data participate in computation
    if (rank >= effective_partitions) {
        // Create an empty partition for this process
        if (rank < partition_vertices.size()) {
            partition_vertices[rank].clear();
        } else if (partition_vertices.size() <= rank) {
            partition_vertices.resize(rank + 1);
        }
    }
    
    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);
}

// Initialize SSSP tree
void SSSPHybrid::initialize() {
    int num_vertices = graph.getNumVertices();
    
    // Resize and initialize sssp_tree on all processes
    sssp_tree.resize(num_vertices);
    for (int i = 0; i < num_vertices; i++) {
        sssp_tree[i].parent = -1;
        sssp_tree[i].distance = INF;
        sssp_tree[i].affected = false;
        sssp_tree[i].affected_del = false;
    }
    
    // Special handling for source vertex
    if (rank == 0) {
        sssp_tree[source].distance = 0.0;
    }
    
    // Only master process computes initial SSSP
    if (rank == 0) {
        sssp_tree = graph.computeInitialSSSP(source);
    }
    
    // Broadcast initial SSSP tree to all processes
    synchronizeSSPTree();
}

// Process changes to identify affected vertices (Step 1)
void SSSPHybrid::processChanges(const std::vector<EdgeChange>& changes) {
    // Skip if this rank doesn't participate in computation
    if (rank >= size || partition_vertices.empty() || (rank < partition_vertices.size() && partition_vertices[rank].empty())) {
        return;
    }
    
    // Ensure sssp_tree is sized correctly
    int num_vertices = graph.getNumVertices();
    if (sssp_tree.size() != num_vertices) {
        sssp_tree.resize(num_vertices);
        for (int i = 0; i < num_vertices; i++) {
            sssp_tree[i].parent = -1;
            sssp_tree[i].distance = INF;
            sssp_tree[i].affected = false;
            sssp_tree[i].affected_del = false;
        }
    }
    
    // Each process handles changes for its own partition
    std::vector<EdgeChange> local_deletions;
    std::vector<EdgeChange> local_insertions;
    
    // Find changes relevant to this process
    for (const auto& change : changes) {
        if (change.source >= num_vertices || change.target >= num_vertices) {
            continue; // Skip invalid vertices
        }
        
        VertexID u = change.source;
        VertexID v = change.target;
        
        // Check if either endpoint is in this process's partition
        if (vertex_to_partition[u] == rank || vertex_to_partition[v] == rank) {
            if (change.type == DELETION) {
                local_deletions.push_back(change);
            } else {
                local_insertions.push_back(change);
            }
        }
    }
    
    // Process deletions first
    processDeletions(local_deletions);
    
    // Then process insertions
    processInsertions(local_insertions);
}

// Process deletions
void SSSPHybrid::processDeletions(const std::vector<EdgeChange>& deletions) {
    int num_vertices = graph.getNumVertices();
    
    #pragma omp parallel for
    for (size_t i = 0; i < deletions.size(); ++i) {
        const auto& change = deletions[i];
        VertexID u = change.source;
        VertexID v = change.target;
        
        // Skip invalid vertices
        if (u >= num_vertices || v >= num_vertices) {
            continue;
        }
        
        // Check if the edge is part of the SSSP tree
        if (sssp_tree[v].parent == u || sssp_tree[u].parent == v) {
            VertexID affected_vertex;
            
            if (sssp_tree[v].parent == u) {
                affected_vertex = v;  // v is child of u
            } else {
                affected_vertex = u;  // u is child of v
            }
            
            // Mark as affected by deletion
            #pragma omp critical
            {
                sssp_tree[affected_vertex].affected_del = true;
                sssp_tree[affected_vertex].affected = true;
                
                // Disconnect from parent
                sssp_tree[affected_vertex].parent = -1;
                sssp_tree[affected_vertex].distance = INF;
            }
        }
    }
}

// Process insertions
void SSSPHybrid::processInsertions(const std::vector<EdgeChange>& insertions) {
    int num_vertices = graph.getNumVertices();
    
    #pragma omp parallel for
    for (size_t i = 0; i < insertions.size(); ++i) {
        const auto& change = insertions[i];
        VertexID u = change.source;
        VertexID v = change.target;
        Weight weight = change.weight;
        
        // Skip invalid vertices
        if (u >= num_vertices || v >= num_vertices) {
            continue;
        }
        
        // Determine vertices with known and unknown distances
        VertexID known_vertex = -1, unknown_vertex = -1;
        bool update_possible = false;
        
        // If both vertices have known distances, check which path is shorter
        if (sssp_tree[u].distance != INF && sssp_tree[v].distance != INF) {
            // Check if path through new edge is shorter for either vertex
            if (sssp_tree[u].distance + weight < sssp_tree[v].distance) {
                known_vertex = u;
                unknown_vertex = v;
                update_possible = true;
            } else if (sssp_tree[v].distance + weight < sssp_tree[u].distance) {
                known_vertex = v;
                unknown_vertex = u;
                update_possible = true;
            }
        } 
        // If only one vertex has a known distance, it can potentially update the other
        else if (sssp_tree[u].distance != INF) {
            known_vertex = u;
            unknown_vertex = v;
            update_possible = true;
        } else if (sssp_tree[v].distance != INF) {
            known_vertex = v;
            unknown_vertex = u;
            update_possible = true;
        }
        
        // Update if possible
        if (update_possible && known_vertex != -1 && unknown_vertex != -1) {
            Weight new_distance = sssp_tree[known_vertex].distance + weight;
            
            // Only update if it improves the distance
            if (new_distance < sssp_tree[unknown_vertex].distance) {
                #pragma omp critical
                {
                    if (new_distance < sssp_tree[unknown_vertex].distance) {
                        sssp_tree[unknown_vertex].distance = new_distance;
                        sssp_tree[unknown_vertex].parent = known_vertex;
                        sssp_tree[unknown_vertex].affected = true;
                    }
                }
            }
        }
    }
}

// Update affected vertices (Step 2)
void SSSPHybrid::updateAffectedVertices() {
    // Skip if this rank doesn't participate in computation
    if (rank >= size || partition_vertices.empty() || 
        (rank < partition_vertices.size() && partition_vertices[rank].empty())) {
        return;
    }
    
    // First process deletion-affected subtrees
    processAffectedSubtrees();
    
    // Then update distances iteratively
    updateDistances();
}

// Process deletion-affected subtrees
void SSSPHybrid::processAffectedSubtrees() {
    // Skip if this rank is out of range
    if (rank >= partition_vertices.size()) {
        return;
    }
    
    // Get vertices in this partition
    const auto& local_vertices = partition_vertices[rank];
    if (local_vertices.empty()) {
        return;
    }
    
    int num_vertices = graph.getNumVertices();
    bool global_affected_exist = true;
    int iteration = 0;
    
    // Maximum number of iterations to prevent infinite loops
    const int MAX_ITERATIONS = 100;
    
    while (global_affected_exist && iteration < MAX_ITERATIONS) {
        iteration++;
        bool local_affected_exist = false;
        
        #pragma omp parallel for schedule(dynamic) reduction(||:local_affected_exist)
        for (size_t i = 0; i < local_vertices.size(); ++i) {
            VertexID v = local_vertices[i];
            
            // Skip invalid vertices
            if (v >= num_vertices) {
                continue;
            }
            
            if (sssp_tree[v].affected_del) {
                // Reset affected_del flag for this vertex
                #pragma omp critical
                {
                    sssp_tree[v].affected_del = false;
                }
                
                // Process all children in this partition
                for (VertexID c : local_vertices) {
                    if (c >= num_vertices) {
                        continue;
                    }
                    
                    if (sssp_tree[c].parent == v) {
                        #pragma omp critical
                        {
                            sssp_tree[c].distance = INF;
                            sssp_tree[c].parent = -1;
                            sssp_tree[c].affected_del = true;
                            sssp_tree[c].affected = true;
                        }
                        local_affected_exist = true;
                    }
                }
            }
        }
        
        // Synchronize across all processes
        try {
            synchronizeSSPTree();
        } catch (const std::exception& e) {
            std::cerr << "Exception in synchronizeSSPTree during processAffectedSubtrees: " 
                      << e.what() << std::endl;
            break;
        }
        
        // Check if any process still has affected vertices
        int local_flag = local_affected_exist ? 1 : 0;
        int global_flag = 0;
        
        MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        global_affected_exist = (global_flag != 0);
    }
}

// Update distances of affected vertices
void SSSPHybrid::updateDistances() {
    // Skip if this rank is out of range
    if (rank >= partition_vertices.size()) {
        return;
    }
    
    // Get vertices in this partition
    const auto& local_vertices = partition_vertices[rank];
    if (local_vertices.empty()) {
        return;
    }
    
    int num_vertices = graph.getNumVertices();
    bool global_affected_exist = true;
    int iterations = 0;
    
    // Maximum number of iterations to prevent infinite loops
    const int MAX_ITERATIONS = 100;
    
    while (global_affected_exist && iterations < MAX_ITERATIONS) {
        iterations++;
        bool local_affected_exist = false;
        
        #pragma omp parallel for schedule(dynamic) reduction(||:local_affected_exist)
        for (size_t i = 0; i < local_vertices.size(); ++i) {
            VertexID v = local_vertices[i];
            
            // Skip invalid vertices
            if (v >= num_vertices) {
                continue;
            }
            
            if (sssp_tree[v].affected) {
                // Reset affected flag for this vertex
                #pragma omp critical
                {
                    sssp_tree[v].affected = false;
                }
                
                // Check all neighbors
                for (const auto& edge : graph.getNeighbors(v)) {
                    VertexID n = edge.first;
                    Weight weight = edge.second;
                    
                    // Skip invalid neighbors
                    if (n >= num_vertices) {
                        continue;
                    }
                    
                    // If neighbor can provide a shorter path to v
                    if (sssp_tree[n].distance != INF && 
                        sssp_tree[n].distance + weight < sssp_tree[v].distance) {
                        #pragma omp critical
                        {
                            sssp_tree[v].distance = sssp_tree[n].distance + weight;
                            sssp_tree[v].parent = n;
                            sssp_tree[v].affected = true;
                        }
                        local_affected_exist = true;
                    }
                    
                    // If v can provide a shorter path to neighbor
                    if (sssp_tree[v].distance != INF && 
                        sssp_tree[v].distance + weight < sssp_tree[n].distance) {
                        #pragma omp critical
                        {
                            if (sssp_tree[v].distance + weight < sssp_tree[n].distance) {
                                sssp_tree[n].distance = sssp_tree[v].distance + weight;
                                sssp_tree[n].parent = v;
                                sssp_tree[n].affected = true;
                                local_affected_exist = true;
                            }
                        }
                    }
                }
            }
        }
        
        // Synchronize across all processes
        try {
            synchronizeSSPTree();
        } catch (const std::exception& e) {
            std::cerr << "Exception in synchronizeSSPTree during updateDistances: " 
                      << e.what() << std::endl;
            break;
        }
        
        // Check if any process still has affected vertices
        int local_flag = local_affected_exist ? 1 : 0;
        int global_flag = 0;
        
        MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        global_affected_exist = (global_flag != 0);
    }
}

// Completely rewritten synchronizeSSPTree function to avoid segmentation faults
// Additional changes to sssp_hybrid.cpp

// 1. Make the synchronizeSSPTree function more robust
void SSSPHybrid::synchronizeSSPTree() {
    int num_vertices = graph.getNumVertices();
    
    // Initialize or resize tree on all processes
    if (sssp_tree.size() != num_vertices) {
        sssp_tree.resize(num_vertices);
        for (int i = 0; i < num_vertices; i++) {
            sssp_tree[i].parent = -1;
            sssp_tree[i].distance = INF;
            sssp_tree[i].affected = false;
            sssp_tree[i].affected_del = false;
        }
    }
    
    // Use simpler, more robust approach avoiding complex MPI operations
    if (rank == 0) {
        // Root collects all data
        std::vector<SSSPNode> global_tree = sssp_tree;
        
        // Receive from all other processes and merge
        for (int p = 1; p < size; p++) {
            std::vector<int> parents(num_vertices, -1);
            std::vector<double> distances(num_vertices, INF);
            std::vector<int> affected(num_vertices, 0);
            std::vector<int> affected_del(num_vertices, 0);
            
            // Receive data from each process
            MPI_Status status;
            MPI_Recv(parents.data(), num_vertices, MPI_INT, p, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(distances.data(), num_vertices, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(affected.data(), num_vertices, MPI_INT, p, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(affected_del.data(), num_vertices, MPI_INT, p, 3, MPI_COMM_WORLD, &status);
            
            // Merge data - find minimum distances
            for (int v = 0; v < num_vertices; v++) {
                // Update if this process has a better distance
                if (distances[v] < global_tree[v].distance) {
                    global_tree[v].distance = distances[v];
                    global_tree[v].parent = parents[v];
                }
                
                // Merge affected flags (OR operation)
                global_tree[v].affected |= (affected[v] != 0);
                global_tree[v].affected_del |= (affected_del[v] != 0);
            }
        }
        
        // Now broadcast the merged tree back to all processes
        std::vector<int> parents(num_vertices);
        std::vector<double> distances(num_vertices);
        std::vector<int> affected(num_vertices);
        std::vector<int> affected_del(num_vertices);
        
        // Extract data
        for (int v = 0; v < num_vertices; v++) {
            parents[v] = global_tree[v].parent;
            distances[v] = global_tree[v].distance;
            affected[v] = global_tree[v].affected ? 1 : 0;
            affected_del[v] = global_tree[v].affected_del ? 1 : 0;
        }
        
        // Broadcast to all processes
        MPI_Bcast(parents.data(), num_vertices, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(distances.data(), num_vertices, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(affected.data(), num_vertices, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(affected_del.data(), num_vertices, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Update root's tree (already done for root)
    } else {
        // Non-root processes send their data
        std::vector<int> parents(num_vertices);
        std::vector<double> distances(num_vertices);
        std::vector<int> affected(num_vertices);
        std::vector<int> affected_del(num_vertices);
        
        // Pack data
        for (int v = 0; v < num_vertices; v++) {
            parents[v] = sssp_tree[v].parent;
            distances[v] = sssp_tree[v].distance;
            affected[v] = sssp_tree[v].affected ? 1 : 0;
            affected_del[v] = sssp_tree[v].affected_del ? 1 : 0;
        }
        
        // Send to root
        MPI_Send(parents.data(), num_vertices, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(distances.data(), num_vertices, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        MPI_Send(affected.data(), num_vertices, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(affected_del.data(), num_vertices, MPI_INT, 0, 3, MPI_COMM_WORLD);
        
        // Receive broadcast from root
        MPI_Bcast(parents.data(), num_vertices, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(distances.data(), num_vertices, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(affected.data(), num_vertices, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(affected_del.data(), num_vertices, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Update local tree
        for (int v = 0; v < num_vertices; v++) {
            sssp_tree[v].parent = parents[v];
            sssp_tree[v].distance = distances[v];
            sssp_tree[v].affected = (affected[v] != 0);
            sssp_tree[v].affected_del = (affected_del[v] != 0);
        }
    }
    
    // Final barrier to ensure all processes are synchronized
    MPI_Barrier(MPI_COMM_WORLD);
}

// 2. Fix the broadcast of changes in main_hybrid.cpp

// Replace the current broadcast code in main_hybrid.cpp with:

// Get affected vertices in partition
std::vector<VertexID> SSSPHybrid::getAffectedVerticesInPartition() {
    std::vector<VertexID> affected_vertices;
    
    // Skip if this rank is out of range
    if (rank >= partition_vertices.size()) {
        return affected_vertices;
    }
    
    // Get vertices in this partition
    const auto& local_vertices = partition_vertices[rank];
    int num_vertices = graph.getNumVertices();
    
    for (VertexID v : local_vertices) {
        if (v < num_vertices && (sssp_tree[v].affected || sssp_tree[v].affected_del)) {
            affected_vertices.push_back(v);
        }
    }
    
    return affected_vertices;
}

// Update SSSP tree with changes
Metrics SSSPHybrid::update(const std::vector<EdgeChange>& changes) {
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
    
    // Count affected vertices in this partition
    std::vector<VertexID> affected_vertices = getAffectedVerticesInPartition();
    int local_affected_count = affected_vertices.size();
    int global_affected_count = 0;
    
    MPI_Reduce(&local_affected_count, &global_affected_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        metrics.affected_vertices = global_affected_count;
    }
    
    // Stop timing
    timer.stop();
    metrics.total_time = timer.getElapsedTimeInSeconds();
    
    return metrics;
}

// Get the updated SSSP tree
const std::vector<SSSPNode>& SSSPHybrid::getSSSPTree() const {
    return sssp_tree;
}

// Print the SSSP tree
void SSSPHybrid::printTree() const {
    if (rank == 0) {
        printSSSP(sssp_tree);
    }
}
