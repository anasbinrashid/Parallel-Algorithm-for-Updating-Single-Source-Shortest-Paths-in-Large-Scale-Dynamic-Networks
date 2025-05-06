/**
 * sssp_temporal_hybrid.cpp
 * Implementation of hybrid OpenMP+MPI+METIS SSSP update algorithm for temporal graphs
 * Fixed to avoid segmentation faults in MPI communication
 */

#include "sssp_temporal_hybrid.h"
#include <omp.h>
#ifdef USE_METIS
#include <metis.h>
#endif
#include <algorithm>
#include <iostream>
#include <queue>
#include <numeric>
#include <cstring>
#include<iomanip>

// Safe access to SSSP tree at time (initialize if not exists)
std::vector<SSSPNode>& SSSPTemporalHybrid::getOrCreateSSSPTree(Timestamp time) {
    // Check if tree exists at this time
    if (sssp_trees.find(time) == sssp_trees.end()) {
        // Find closest earlier time
        Timestamp prev_time = 0.0;
        for (const auto& entry : sssp_trees) {
            if (entry.first < time && entry.first > prev_time) {
                prev_time = entry.first;
            }
        }
        
        // If we have a previous tree, copy it; otherwise initialize a new one
        if (sssp_trees.find(prev_time) != sssp_trees.end()) {
            sssp_trees[time] = sssp_trees[prev_time];
            
            // Update time in each node
            for (auto& node : sssp_trees[time]) {
                node.time = time;
            }
        } else {
            // Initialize a new SSSP tree
            sssp_trees[time].resize(graph.getNumVertices());
            for (auto& node : sssp_trees[time]) {
                node.time = time;
            }
        }
    }
    
    return sssp_trees[time];
}

// Constructor
SSSPTemporalHybrid::SSSPTemporalHybrid(TemporalGraph& g, VertexID src, int threads)
    : graph(g), source(src), num_threads(threads) {
    
    // Initialize MPI
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(NULL, NULL);
    }
    
    // Get rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Set number of threads for OpenMP
    omp_set_num_threads(num_threads);
    
    // Initialize partition information vectors with safe sizes
    int num_vertices = graph.getNumVertices();
    vertex_to_partition.resize(num_vertices, 0);  // Default all vertices to partition 0
    partition_vertices.resize(size);
    boundary_vertices.resize(size);
    
    // Simple initial partition (will be refined by partitionGraph)
    simplePartition();
}

// Destructor
SSSPTemporalHybrid::~SSSPTemporalHybrid() {
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }
}

// Simple partitioning method (fallback or initial partition)
void SSSPTemporalHybrid::simplePartition() {
    int num_vertices = graph.getNumVertices();
    
    // Clear existing partition data
    vertex_to_partition.assign(num_vertices, 0);
    for (int p = 0; p < size; ++p) {
        partition_vertices[p].clear();
        boundary_vertices[p].clear();
    }
    
    // Simple round-robin assignment of vertices to partitions
    for (VertexID v = 0; v < num_vertices; ++v) {
        int p = v % size;
        vertex_to_partition[v] = p;
        partition_vertices[p].push_back(v);
    }
    
    // Identify boundary vertices
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
    
    if (rank == 0) {
        std::cout << "Using simple partitioning (round-robin)" << std::endl;
    }
}

// Partition the graph using METIS
void SSSPTemporalHybrid::partitionGraph() {
    int num_vertices = graph.getNumVertices();
    
    // Start with simple partitioning to ensure we have valid partition data
    simplePartition();
    
#ifdef USE_METIS
    // Only perform METIS partitioning on the master process if we have a graph with enough vertices
    if (rank == 0 && num_vertices >= size * 2) {
        // Set graph time to 0 for initial partitioning
        graph.setCurrentTime(0.0);
        
        // Convert graph to CSR format for METIS
        std::vector<idx_t> xadj;
        std::vector<idx_t> adjncy;
        std::vector<idx_t> adjwgt;
        
        // Initialize arrays with enough capacity
        xadj.reserve(num_vertices + 1);
        xadj.push_back(0);
        
        // Count total edges for pre-allocation
        int total_edges = 0;
        for (VertexID v = 0; v < num_vertices; ++v) {
            total_edges += graph.getNeighbors(v).size();
        }
        
        adjncy.reserve(total_edges);
        adjwgt.reserve(total_edges);
        
        // Fill CSR format arrays
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
        idx_t nparts = size;  // Number of partitions equals number of processes
        
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
            std::cerr << "METIS partitioning failed with error code " << ret << std::endl;
            std::cerr << "Using simple partitioning as fallback" << std::endl;
            simplePartition();
        } else {
            // METIS successful, use its partitioning
            std::cout << "Using METIS partitioning (edge-cut: " << objval << ")" << std::endl;
            
            // Clear existing partition data
            for (int p = 0; p < size; ++p) {
                partition_vertices[p].clear();
                boundary_vertices[p].clear();
            }
            
            // Assign vertices to partitions
            for (VertexID v = 0; v < num_vertices; ++v) {
                vertex_to_partition[v] = part[v];
                partition_vertices[part[v]].push_back(v);
            }
            
            // Identify boundary vertices
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
    }
#endif
    
    // Broadcast partition information to all processes
    MPI_Bcast(vertex_to_partition.data(), num_vertices, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Synchronize partition_vertices
    std::vector<int> partition_sizes(size);
    
    if (rank == 0) {
        for (int p = 0; p < size; ++p) {
            partition_sizes[p] = partition_vertices[p].size();
        }
    }
    
    MPI_Bcast(partition_sizes.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        // Resize partition_vertices based on received sizes
        for (int p = 0; p < size; ++p) {
            partition_vertices[p].resize(partition_sizes[p]);
        }
    }
    
    // Broadcast each partition's vertices
    for (int p = 0; p < size; ++p) {
        if (partition_sizes[p] > 0) {
            MPI_Bcast(partition_vertices[p].data(), partition_sizes[p], MPI_INT, 0, MPI_COMM_WORLD);
        }
    }
    
    // Synchronize boundary_vertices
    std::vector<int> boundary_sizes(size);
    
    if (rank == 0) {
        for (int p = 0; p < size; ++p) {
            boundary_sizes[p] = boundary_vertices[p].size();
        }
    }
    
    MPI_Bcast(boundary_sizes.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    
    for (int p = 0; p < size; ++p) {
        std::vector<VertexID> boundary_array;
        
        if (rank == 0) {
            boundary_array.assign(boundary_vertices[p].begin(), boundary_vertices[p].end());
        } else {
            boundary_array.resize(boundary_sizes[p]);
        }
        
        if (boundary_sizes[p] > 0) {
            MPI_Bcast(boundary_array.data(), boundary_sizes[p], MPI_INT, 0, MPI_COMM_WORLD);
        }
        
        if (rank != 0) {
            boundary_vertices[p].clear();
            boundary_vertices[p].insert(boundary_array.begin(), boundary_array.end());
        }
    }
}

// Initialize SSSP tree at time 0
void SSSPTemporalHybrid::initialize() {
    // Set graph time to 0
    graph.setCurrentTime(0.0);
    
    // Initialize an empty SSSP tree vector on all processes
    std::vector<SSSPNode> initial_sssp(graph.getNumVertices());
    
    // Only master process computes initial SSSP
    if (rank == 0) {
        std::cout << "Computing initial SSSP tree on rank 0..." << std::endl;
        initial_sssp = graph.computeInitialSSSP(source);
    }
    
    // Store SSSP tree for time 0 (will be updated by synchronization)
    sssp_trees[0.0] = initial_sssp;
    
    // Broadcast initial SSSP tree to all processes
    synchronizeSSPTree(0.0);
    
    if (rank == 0) {
        std::cout << "Initial SSSP tree synchronized across all processes" << std::endl;
    }
}

// Process changes to identify affected vertices (Step 1)
void SSSPTemporalHybrid::processChanges(const std::vector<EdgeChange>& changes, Timestamp time) {
    // Each process handles changes for its own partition
    std::vector<EdgeChange> local_deletions;
    std::vector<EdgeChange> local_insertions;
    
    for (const auto& change : changes) {
        if (change.time == time) {
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
    }
    
    // Get or create SSSP tree for this time
    auto& sssp_tree = getOrCreateSSSPTree(time);
    
    // Process deletions first
    processDeletions(local_deletions, time);
    
    // Then process insertions
    processInsertions(local_insertions, time);
}

// Process deletions
void SSSPTemporalHybrid::processDeletions(const std::vector<EdgeChange>& deletions, Timestamp time) {
    auto& sssp_tree = getOrCreateSSSPTree(time);
    
    #pragma omp parallel for
    for (size_t i = 0; i < deletions.size(); ++i) {
        const auto& change = deletions[i];
        VertexID u = change.source;
        VertexID v = change.target;
        
        // Check if the edge is part of the SSSP tree
        if (sssp_tree[v].parent == u || sssp_tree[u].parent == v) {
            VertexID affected_vertex;
            
            if (sssp_tree[v].parent == u) {
                affected_vertex = v;  // v is child of u
            } else {
                affected_vertex = u;  // u is child of v
            }
            
            // Mark as affected by deletion
            sssp_tree[affected_vertex].affected_del = true;
            sssp_tree[affected_vertex].affected = true;
            
            // Disconnect from parent
            sssp_tree[affected_vertex].parent = -1;
            sssp_tree[affected_vertex].distance = INF;
        }
    }
}

// Process insertions
void SSSPTemporalHybrid::processInsertions(const std::vector<EdgeChange>& insertions, Timestamp time) {
    auto& sssp_tree = getOrCreateSSSPTree(time);
    
    #pragma omp parallel for
    for (size_t i = 0; i < insertions.size(); ++i) {
        const auto& change = insertions[i];
        VertexID u = change.source;
        VertexID v = change.target;
        Weight weight = change.weight;
        
        // Determine vertices with known and unknown distances
        VertexID known_vertex, unknown_vertex;
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
        if (update_possible) {
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
void SSSPTemporalHybrid::updateAffectedVertices(Timestamp time) {
    // First process deletion-affected subtrees
    processAffectedSubtrees(time);
    
    // Then update distances iteratively
    updateDistances(time);
}

// Process deletion-affected subtrees
void SSSPTemporalHybrid::processAffectedSubtrees(Timestamp time) {
    // Get vertices in this partition
    const auto& local_vertices = partition_vertices[rank];
    auto& sssp_tree = getOrCreateSSSPTree(time);
    
    bool global_affected_exist = true;
    int iteration = 0;
    
    while (global_affected_exist && iteration < 100) { // Limit iterations to avoid infinite loops
        iteration++;
        bool local_affected_exist = false;
        
        #pragma omp parallel for schedule(dynamic) reduction(||:local_affected_exist)
        for (size_t i = 0; i < local_vertices.size(); ++i) {
            VertexID v = local_vertices[i];
            
            if (v < sssp_tree.size() && sssp_tree[v].affected_del) {
                sssp_tree[v].affected_del = false;
                
                // Process all children in this partition
                for (size_t j = 0; j < local_vertices.size(); ++j) {
                    VertexID c = local_vertices[j];
                    if (c < sssp_tree.size() && sssp_tree[c].parent == v) {
                        sssp_tree[c].distance = INF;
                        sssp_tree[c].parent = -1;
                        sssp_tree[c].affected_del = true;
                        sssp_tree[c].affected = true;
                        local_affected_exist = true;
                    }
                }
            }
        }
        
        // Synchronize across all processes
        synchronizeSSPTree(time);
        
        // Check if any process still has affected vertices
        int local_flag = local_affected_exist ? 1 : 0;
        int global_flag = 0;
        
        MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        global_affected_exist = (global_flag != 0);
    }
}

// Update distances of affected vertices
void SSSPTemporalHybrid::updateDistances(Timestamp time) {
    // Get vertices in this partition
    const auto& local_vertices = partition_vertices[rank];
    auto& sssp_tree = getOrCreateSSSPTree(time);
    
    bool global_affected_exist = true;
    int iterations = 0;
    
    while (global_affected_exist && iterations < 100) { // Limit iterations to avoid infinite loops
        iterations++;
        bool local_affected_exist = false;
        
        #pragma omp parallel for schedule(dynamic) reduction(||:local_affected_exist)
        for (size_t i = 0; i < local_vertices.size(); ++i) {
            VertexID v = local_vertices[i];
            
            if (v < sssp_tree.size() && sssp_tree[v].affected) {
                // Reset affected flag
                sssp_tree[v].affected = false;
                
                // Check all neighbors in current graph snapshot
                for (const auto& edge : graph.getNeighbors(v)) {
                    VertexID n = edge.first;
                    Weight weight = edge.second;
                    
                    if (n >= sssp_tree.size()) continue;
                    
                    // If neighbor can provide a shorter path to v
                    if (sssp_tree[n].distance != INF && 
                        sssp_tree[n].distance + weight < sssp_tree[v].distance) {
                        sssp_tree[v].distance = sssp_tree[n].distance + weight;
                        sssp_tree[v].parent = n;
                        sssp_tree[v].affected = true;
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
        synchronizeSSPTree(time);
        
        // Check if any process still has affected vertices
        int local_flag = local_affected_exist ? 1 : 0;
        int global_flag = 0;
        
        MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        global_affected_exist = (global_flag != 0);
    }
}

// Synchronize SSSP tree across processes for a specific time
void SSSPTemporalHybrid::synchronizeSSPTree(Timestamp time) {
    int num_vertices = graph.getNumVertices();
    
    // Ensure all processes have an SSSP tree for this time with proper size
    auto& sssp_tree = getOrCreateSSSPTree(time);
    
    // Ensure tree has correct size
    if (sssp_tree.size() != num_vertices) {
        sssp_tree.resize(num_vertices);
    }
    
    // Create buffer for node data (easier to handle than structs with MPI)
    // Format for each vertex: [parent, distance, affected, affected_del]
    std::vector<double> send_buffer(num_vertices * 4);
    std::vector<double> recv_buffer(num_vertices * 4);
    
    // Pack SSSP data into buffer
    for (VertexID v = 0; v < num_vertices; ++v) {
        send_buffer[v*4]     = static_cast<double>(sssp_tree[v].parent);
        send_buffer[v*4 + 1] = sssp_tree[v].distance;
        send_buffer[v*4 + 2] = sssp_tree[v].affected ? 1.0 : 0.0;
        send_buffer[v*4 + 3] = sssp_tree[v].affected_del ? 1.0 : 0.0;
    }
    
    // Allreduce to get minimum distances and combine flags
    MPI_Allreduce(send_buffer.data(), recv_buffer.data(), num_vertices*4, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    
    // Unpack data
    for (VertexID v = 0; v < num_vertices; ++v) {
        double parent_val = recv_buffer[v*4];
        double dist_val = recv_buffer[v*4 + 1];
        
        // Only update if we got a better distance
        if (dist_val < sssp_tree[v].distance) {
            sssp_tree[v].parent = static_cast<VertexID>(parent_val);
            sssp_tree[v].distance = dist_val;
        }
        
        // OR the affected flags
        sssp_tree[v].affected |= (recv_buffer[v*4 + 2] > 0.5);
        sssp_tree[v].affected_del |= (recv_buffer[v*4 + 3] > 0.5);
    }
    
    // Make sure all processes are in sync
    MPI_Barrier(MPI_COMM_WORLD);
}

// Get affected vertices in partition at a specific time
std::vector<VertexID> SSSPTemporalHybrid::getAffectedVerticesInPartition(Timestamp time) {
    const auto& local_vertices = partition_vertices[rank];
    std::vector<VertexID> affected_vertices;
    
    if (sssp_trees.find(time) != sssp_trees.end()) {
        const auto& sssp_tree = sssp_trees[time];
        
        for (VertexID v : local_vertices) {
            if (v < sssp_tree.size() && (sssp_tree[v].affected || sssp_tree[v].affected_del)) {
                affected_vertices.push_back(v);
            }
        }
    }
    
    return affected_vertices;
}

// Update SSSP tree for a specific timestamp
Metrics SSSPTemporalHybrid::updateAtTime(Timestamp time) {
    Metrics metrics;
    Timer timer;
    
    // Update graph to the specified time
    graph.setCurrentTime(time);
    
    // Start timing
    timer.start();
    
    // Find all changes that happened at this time
    std::vector<EdgeChange> changes_at_time;
    
    // Step 1: Process changes to identify affected vertices
    Timer step1_timer;
    step1_timer.start();
    processChanges(changes_at_time, time);
    step1_timer.stop();
    metrics.step1_time = step1_timer.getElapsedTimeInSeconds();
    
    // Step 2: Update affected vertices
    Timer step2_timer;
    step2_timer.start();
    updateAffectedVertices(time);
    step2_timer.stop();
    metrics.step2_time = step2_timer.getElapsedTimeInSeconds();
    
    // Count affected vertices
    int local_affected_count = getAffectedVerticesInPartition(time).size();
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

// Update SSSP tree with changes
Metrics SSSPTemporalHybrid::update(const std::vector<EdgeChange>& changes) {
    Metrics metrics;
    Timer timer;
    
    // Apply changes to the graph
    graph.applyChanges(changes);
    
    // Get all timepoints where changes occur
    std::set<Timestamp> timepoints;
    for (const auto& change : changes) {
        timepoints.insert(change.time);
    }
    
    // Start timing
    timer.start();
    
    // For each timepoint, update the SSSP tree
    Timer step1_timer, step2_timer;
    int total_affected = 0;
    
    for (Timestamp time : timepoints) {
        // Set graph time to current timepoint
        graph.setCurrentTime(time);
        
        // Find changes at this timepoint
        std::vector<EdgeChange> changes_at_time;
        for (const auto& change : changes) {
            if (change.time == time) {
                changes_at_time.push_back(change);
            }
        }
        
        // Step 1: Process changes to identify affected vertices
        step1_timer.start();
        processChanges(changes_at_time, time);
        step1_timer.stop();
        
        // Step 2: Update affected vertices
        step2_timer.start();
        updateAffectedVertices(time);
        step2_timer.stop();
        
        // Count affected vertices
        int local_affected_count = getAffectedVerticesInPartition(time).size();
        int global_affected_count = 0;
        
        MPI_Reduce(&local_affected_count, &global_affected_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            total_affected += global_affected_count;
        }
    }
    
    // Stop timing
    timer.stop();
    
    // Broadcast total affected vertices to all processes
    MPI_Bcast(&total_affected, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Populate metrics
    metrics.total_time = timer.getElapsedTimeInSeconds();
    metrics.step1_time = step1_timer.getElapsedTimeInSeconds();
    metrics.step2_time = step2_timer.getElapsedTimeInSeconds();
    metrics.affected_vertices = total_affected;
    
    return metrics;
}

// Update SSSP trees for all timepoints
std::vector<Metrics> SSSPTemporalHybrid::updateAllTimepoints() {
    // Get all timepoints in the temporal graph
    std::vector<Timestamp> timepoints = graph.getAllTimepoints();
    std::vector<Metrics> all_metrics;
    
    // Sort timepoints in ascending order
    std::sort(timepoints.begin(), timepoints.end());
    
    // Initialize SSSP tree at time 0 if not already done
    if (sssp_trees.empty()) {
        initialize();
    }
    
    // Update SSSP tree for each timepoint
    for (Timestamp time : timepoints) {
        if (time > 0 && sssp_trees.find(time) == sssp_trees.end()) {
            Metrics metrics = updateAtTime(time);
            all_metrics.push_back(metrics);
        }
    }
    
    return all_metrics;
}

// Get the SSSP tree at a specific time
const std::vector<SSSPNode>& SSSPTemporalHybrid::getSSSPTreeAtTime(Timestamp time) const {
    auto it = sssp_trees.find(time);
    if (it == sssp_trees.end()) {
        // Find closest earlier time
        Timestamp closest_time = 0.0;
        for (const auto& entry : sssp_trees) {
            if (entry.first <= time && entry.first > closest_time) {
                closest_time = entry.first;
            }
        }
        
        // If we have any tree, return the closest one
        if (!sssp_trees.empty()) {
            it = sssp_trees.find(closest_time);
            if (it != sssp_trees.end()) {
                return it->second;
            }
        }
        
        // No tree found, throw exception
        throw std::runtime_error("No SSSP tree available for time " + std::to_string(time));
    }
    return it->second;
}

// Get all SSSP trees
const std::map<Timestamp, std::vector<SSSPNode>>& SSSPTemporalHybrid::getAllSSSPTrees() const {
    return sssp_trees;
}

// Print the SSSP tree at a specific time
void SSSPTemporalHybrid::printTree(Timestamp time) const {
    if (rank == 0) {
        try {
            const auto& sssp_tree = getSSSPTreeAtTime(time);
            
            std::cout << "SSSP Tree at time " << time << ":" << std::endl;
            std::cout << std::setw(10) << "Vertex" << std::setw(10) << "Parent" 
                      << std::setw(15) << "Distance" << std::endl;
            
            for (size_t v = 0; v < sssp_tree.size(); ++v) {
                std::cout << std::setw(10) << v << std::setw(10) << sssp_tree[v].parent;
                
                if (sssp_tree[v].distance == INF) {
                    std::cout << std::setw(15) << "INF" << std::endl;
                } else {
                    std::cout << std::setw(15) << sssp_tree[v].distance << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cout << "Error printing tree: " << e.what() << std::endl;
        }
    }
}

// Print all SSSP trees
void SSSPTemporalHybrid::printAllTrees() const {
    if (rank == 0) {
        for (const auto& entry : sssp_trees) {
            printTree(entry.first);
            std::cout << std::endl;
        }
    }
}
