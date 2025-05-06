#include "parallel_sssp.h"
#include <iostream>
#include <queue>
#include <limits>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <unordered_set>
#include <numeric>  

ParallelSSSP::ParallelSSSP(int rank, int size) : 
    rank(rank), 
    size(size), 
    sourceVertex(-1), 
    numThreads(1), 
    asyncLevel(1),
    maxIterations(100),
    verbose(false) {
    
    // Set default number of OpenMP threads
    omp_set_num_threads(numThreads);
}

void ParallelSSSP::initialize(const Graph& globalGraph, int sourceVertex, int numThreads, int asyncLevel) {
    this->sourceVertex = sourceVertex;
    this->numThreads = numThreads;
    this->asyncLevel = asyncLevel;
    
    // Set number of OpenMP threads
    omp_set_num_threads(numThreads);
    
    // Determine optimal number of partitions based on graph size and available processes
    int graphSize = globalGraph.getNumVertices();
    int optimalPartitions = std::min(size, (graphSize > 5000) ? size : 2);
    
    log("Initializing with " + std::to_string(optimalPartitions) + " partitions");
    
    // Partition the graph - only rank 0 does this to ensure consistency
    if (rank == 0) {
        partitions = MetisWrapper::partitionGraph(globalGraph, optimalPartitions);
        
        // Distribute partitioning information to all other processes
        for (int i = 1; i < size; i++) {
            MPI_Send(partitions.data(), globalGraph.getNumVertices(), MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        // Receive partitioning information from rank 0
        partitions.resize(globalGraph.getNumVertices());
        MPI_Recv(partitions.data(), globalGraph.getNumVertices(), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // Create local subgraph for this process
    int partID = rank % optimalPartitions;
    localGraph = MetisWrapper::createLocalGraph(globalGraph, partitions, partID);
    
    // Get ghost vertices
    ghostVertices = MetisWrapper::getGhostVertices(globalGraph, partitions, partID);
    
    log("Local graph has " + std::to_string(localGraph.getNumVertices()) + " vertices and " 
        + std::to_string(localGraph.getNumEdges()) + " edges, with " 
        + std::to_string(ghostVertices.size()) + " ghost vertices");
    
    // Create mapping between global and local vertex IDs
    for (int localID = 0; localID < localGraph.getNumVertices(); localID++) {
        int globalID = localGraph.reverseMapVertex(localID);
        globalToLocalMap[globalID] = localID;
        localToGlobalMap[localID] = globalID;
    }
    
    // Initialize local SSSP tree
    int localSourceID = mapGlobalToLocal(sourceVertex);
    if (localSourceID == -1) {
        // Source vertex is not in this partition
        log("Source vertex " + std::to_string(sourceVertex) + " is not in this partition. Finding closest vertex.");
        
        // Find the first valid vertex in this partition
        localSourceID = -1;
        for (const auto& pair : localToGlobalMap) {
            if (localSourceID == -1 || pair.first < localSourceID) {
                localSourceID = pair.first;
            }
        }
        
        if (localSourceID == -1 && !localToGlobalMap.empty()) {
            // Just take the first vertex if nothing better
            localSourceID = localToGlobalMap.begin()->first;
        }
        
        if (localSourceID == -1) {
            // Empty partition - use 0 but ensure the tree is properly initialized
            localSourceID = 0;
        }
    }
    
    log("Using local source vertex: " + std::to_string(localSourceID));
    
    localTree = SSSPTree(localGraph.getNumVertices(), localSourceID);
    localTree.initialize(localGraph, localGraph.reverseMapVertex(localSourceID));
    
    // Exchange boundary information to ensure consistency
    try {
        exchangeBoundaryInfo();
    } catch (const std::exception& e) {
        log("Warning: Exception during initial boundary exchange: " + std::string(e.what()));
        // Continue anyway - we'll try to recover
    }
}

void ParallelSSSP::updateSSSP(const std::vector<EdgeChange>& changes) {
    if (changes.empty()) {
        log("No changes to process");
        return;
    }
    
    log("Processing " + std::to_string(changes.size()) + " edge changes");
    
    // Filter changes to only include those relevant to this partition
    std::vector<EdgeChange> localChanges;
    for (const EdgeChange& change : changes) {
        int srcLocal = mapGlobalToLocal(change.source);
        int destLocal = mapGlobalToLocal(change.target);
        
        // Include if either source or target is in this partition
        if (srcLocal != -1 || destLocal != -1) {
            localChanges.push_back(change);
        }
    }
    
    log("Local changes: " + std::to_string(localChanges.size()));
    
    // Process in batches to avoid overwhelming communication
    const int BATCH_SIZE = 500;
    int totalBatches = (localChanges.size() + BATCH_SIZE - 1) / BATCH_SIZE;
    
    for (int batch = 0; batch < totalBatches; batch++) {
        int startIdx = batch * BATCH_SIZE;
        int endIdx = std::min(startIdx + BATCH_SIZE, static_cast<int>(localChanges.size()));
        
        std::vector<EdgeChange> batchChanges(localChanges.begin() + startIdx, 
                                            localChanges.begin() + endIdx);
        
        log("Processing batch " + std::to_string(batch+1) + "/" + std::to_string(totalBatches) + 
            " with " + std::to_string(batchChanges.size()) + " changes");
        
        // Step 1: Identify vertices affected by changes
        try {
            // Process deletions first, then insertions
            std::vector<EdgeChange> deletions, insertions;
            for (const EdgeChange& edge : batchChanges) {
                if (edge.isInsert) {
                    insertions.push_back(edge);
                } else {
                    deletions.push_back(edge);
                }
            }
            
            // Process deletions first
            #pragma omp parallel for schedule(dynamic, 64)
            for (size_t i = 0; i < deletions.size(); i++) {
                processEdgeDeletion(deletions[i]);
            }
            
            // Then process insertions
            #pragma omp parallel for schedule(dynamic, 64)
            for (size_t i = 0; i < insertions.size(); i++) {
                processEdgeInsertion(insertions[i]);
            }
            
            // Update the local graph structure
            for (const EdgeChange& edge : batchChanges) {
                int localU = mapGlobalToLocal(edge.source);
                int localV = mapGlobalToLocal(edge.target);
                
                // Skip if neither vertex exists locally
                if (localU == -1 && localV == -1) {
                    continue;
                }
                
                int originalU = edge.source;
                int originalV = edge.target;
                
                if (edge.isInsert) {
                    // Add edge to local graph if needed
                    if (localU != -1 && localV != -1) {
                        try {
                            localGraph.addEdge(originalU, originalV, edge.weight);
                        } catch (const std::exception& e) {
                            log("Warning: Failed to add edge: " + std::string(e.what()));
                        }
                    }
                } else {
                    // Remove edge from local graph if it exists
                    if (localU != -1 && localV != -1) {
                        try {
                            localGraph.removeEdge(originalU, originalV);
                        } catch (const std::exception& e) {
                            log("Warning: Failed to remove edge: " + std::string(e.what()));
                        }
                    }
                }
            }
            
            // Exchange boundary information
            if (batch % 2 == 0 || batch == totalBatches - 1) {
                exchangeBoundaryInfo();
            }
            
            // Step 2: Update affected subgraphs using a priority-based approach
            updateAffectedSubgraphs();
            
            // Final exchange at end of batch
            if (batch == totalBatches - 1) {
                exchangeBoundaryInfo();
            }
        } catch (const std::exception& e) {
            log("Warning: Exception during batch processing: " + std::string(e.what()));
            // Continue with next batch
        }
    }
    
    log("SSSP update complete");
}

void ParallelSSSP::gatherResults(Graph& originalGraph, SSSPTree& globalTree) {
    // Only gather results on rank 0
    if (rank == 0) {
        log("Gathering results from all processes");
        
        // Initialize global tree
        globalTree = SSSPTree(originalGraph.getNumVertices(), originalGraph.mapVertex(sourceVertex));
        
        // Create MPI datatype for BoundaryInfo
        MPI_Datatype boundaryInfoType = createBoundaryInfoType();
        
        // First, initialize distances to infinity
        for (int i = 0; i < globalTree.numVertices; i++) {
            globalTree.distance[i] = std::numeric_limits<float>::infinity();
        }
        
        // Gather local tree information from all processes
        for (int i = 0; i < size; i++) {
            if (i == 0) {
                // Copy local tree of rank 0 to global tree
                for (int v = 0; v < localTree.numVertices; v++) {
                    int globalID = mapLocalToGlobal(v);
                    int globalMapped = originalGraph.mapVertex(globalID);
                    
                    if (globalMapped != -1) {
                        int parentGlobalID = (localTree.parent[v] != -1) ? mapLocalToGlobal(localTree.parent[v]) : -1;
                        int parentGlobalMapped = (parentGlobalID != -1) ? originalGraph.mapVertex(parentGlobalID) : -1;
                        
                        // Update only if this distance is better than existing
                        if (localTree.distance[v] < globalTree.distance[globalMapped]) {
                            globalTree.distance[globalMapped] = localTree.distance[v];
                            globalTree.parent[globalMapped] = parentGlobalMapped;
                        }
                    }
                }
            } else {
                // Receive information from other processes
                int localSize;
                MPI_Status status;
                MPI_Recv(&localSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
                
                if (localSize > 0) {
                    std::vector<BoundaryInfo> remoteInfo(localSize);
                    MPI_Recv(remoteInfo.data(), localSize, boundaryInfoType, i, 0, MPI_COMM_WORLD, &status);
                    
                    // Update global tree with remote information
                    for (const auto& info : remoteInfo) {
                        int globalMapped = originalGraph.mapVertex(info.vertexID);
                        if (globalMapped != -1) {
                            int parentGlobalMapped = (info.parent != -1) ? originalGraph.mapVertex(info.parent) : -1;
                            
                            // Update only if this is a better path
                            if (info.distance < globalTree.distance[globalMapped]) {
                                globalTree.distance[globalMapped] = info.distance;
                                globalTree.parent[globalMapped] = parentGlobalMapped;
                            }
                        }
                    }
                }
            }
        }
        
        MPI_Type_free(&boundaryInfoType);
        log("Results gathered successfully");
    } else {
        // Send local tree information to rank 0
        MPI_Datatype boundaryInfoType = createBoundaryInfoType();
        
        // Prepare tree information to send - filter out infinite distances
        std::vector<BoundaryInfo> treeInfo;
        for (int v = 0; v < localTree.numVertices; v++) {
            if (!std::isinf(localTree.distance[v])) {
                int globalID = mapLocalToGlobal(v);
                int parentGlobalID = (localTree.parent[v] != -1) ? mapLocalToGlobal(localTree.parent[v]) : -1;
                
                treeInfo.emplace_back(globalID, localTree.distance[v], parentGlobalID);
            }
        }
        
        // Send size and data
        int localSize = treeInfo.size();
        MPI_Send(&localSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        
        if (localSize > 0) {
            MPI_Send(treeInfo.data(), localSize, boundaryInfoType, 0, 0, MPI_COMM_WORLD);
        }
        
        MPI_Type_free(&boundaryInfoType);
        log("Sent " + std::to_string(localSize) + " vertices to rank 0");
    }
}

const SSSPTree& ParallelSSSP::getLocalTree() const {
    return localTree;
}

const Graph& ParallelSSSP::getLocalGraph() const {
    return localGraph;
}

void ParallelSSSP::setAsynchronyLevel(int level) {
    asyncLevel = level;
    log("Set asynchrony level to " + std::to_string(level));
}

void ParallelSSSP::setMaxIterations(int max) {
    maxIterations = max;
    log("Set maximum iterations to " + std::to_string(max));
}

int ParallelSSSP::mapGlobalToLocal(int globalID) const {
    auto it = globalToLocalMap.find(globalID);
    if (it != globalToLocalMap.end()) {
        return it->second;
    }
    return -1; // Global ID not found in local graph
}

int ParallelSSSP::mapLocalToGlobal(int localID) const {
    auto it = localToGlobalMap.find(localID);
    if (it != localToGlobalMap.end()) {
        return it->second;
    }
    return -1; // Local ID not found
}

void ParallelSSSP::identifyAffectedVertices(const std::vector<EdgeChange>& changes) {
    log("Identifying affected vertices from " + std::to_string(changes.size()) + " changes");
    
    // Track affected vertices to reduce unnecessary computation
    std::unordered_set<int> uniqueAffectedVertices;
    
    // Separate deletions and insertions - process deletions first
    std::vector<EdgeChange> deletions, insertions;
    for (const EdgeChange& edge : changes) {
        if (edge.isInsert) {
            insertions.push_back(edge);
        } else {
            deletions.push_back(edge);
        }
    }
    
    // Process deletions first
    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < deletions.size(); i++) {
        const EdgeChange& edge = deletions[i];
        processEdgeDeletion(edge);
    }
    
    // Then process insertions
    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < insertions.size(); i++) {
        const EdgeChange& edge = insertions[i];
        processEdgeInsertion(edge);
    }
    
    // Process any local graph structural changes
    for (const EdgeChange& edge : changes) {
        int localU = mapGlobalToLocal(edge.source);
        int localV = mapGlobalToLocal(edge.target);
        
        // Skip if neither vertex exists locally
        if (localU == -1 && localV == -1) {
            continue;
        }
        
        int originalU = edge.source;
        int originalV = edge.target;
        
        if (edge.isInsert) {
            // Add edge to local graph if needed
            if (localU != -1 && localV != -1) {
                try {
                    localGraph.addEdge(originalU, originalV, edge.weight);
                } catch (const std::exception& e) {
                    log("Warning: Failed to add edge (" + std::to_string(originalU) + 
                        "," + std::to_string(originalV) + "): " + std::string(e.what()));
                }
            }
        } else {
            // Remove edge from local graph if it exists
            if (localU != -1 && localV != -1) {
                try {
                    localGraph.removeEdge(originalU, originalV);
                } catch (const std::exception& e) {
                    log("Warning: Failed to remove edge (" + std::to_string(originalU) + 
                        "," + std::to_string(originalV) + "): " + std::string(e.what()));
                }
            }
        }
    }
}

void ParallelSSSP::processEdgeDeletion(const EdgeChange& edge) {
    int localU = mapGlobalToLocal(edge.source);
    int localV = mapGlobalToLocal(edge.target);
    
    // Skip if neither vertex exists locally
    if (localU == -1 && localV == -1) {
        return;
    }
    
    // At least one vertex is local
    bool isTreeEdge = false;
    int child = -1;
    
    #pragma omp critical(tree_update)
    {
        // Check if this edge is part of the SSSP tree
        if (localU != -1 && localV != -1) {
            if (localTree.parent[localV] == localU) {
                isTreeEdge = true;
                child = localV;
            } else if (localTree.parent[localU] == localV) {
                isTreeEdge = true;
                child = localU;
            }
        }
        
        if (isTreeEdge && child != -1) {
            // Mark the child as affected by deletion
            localTree.distance[child] = std::numeric_limits<float>::infinity();
            localTree.parent[child] = -1;
            localTree.affectedDel[child] = true;
            localTree.affected[child] = true;
        }
    }
}

void ParallelSSSP::processEdgeInsertion(const EdgeChange& edge) {
    int localU = mapGlobalToLocal(edge.source);
    int localV = mapGlobalToLocal(edge.target);
    
    // Skip if neither vertex exists locally
    if (localU == -1 && localV == -1) {
        return;
    }
    
    // At least one vertex is local - need to check if this edge provides a better path
    float distU = (localU != -1) ? localTree.distance[localU] : std::numeric_limits<float>::infinity();
    float distV = (localV != -1) ? localTree.distance[localV] : std::numeric_limits<float>::infinity();
    
    // Consider both directions
    if (localU != -1 && !std::isinf(distU)) {
        float newDist = distU + edge.weight;
        if (localV != -1 && newDist < distV) {
            #pragma omp critical(tree_update)
            {
                // Double-check after lock
                if (newDist < localTree.distance[localV]) {
                    localTree.distance[localV] = newDist;
                    localTree.parent[localV] = localU;
                    localTree.affected[localV] = true;
                }
            }
        }
    }
    
    if (localV != -1 && !std::isinf(distV)) {
        float newDist = distV + edge.weight;
        if (localU != -1 && newDist < distU) {
            #pragma omp critical(tree_update)
            {
                // Double-check after lock
                if (newDist < localTree.distance[localU]) {
                    localTree.distance[localU] = newDist;
                    localTree.parent[localU] = localV;
                    localTree.affected[localU] = true;
                }
            }
        }
    }
}

void ParallelSSSP::updateAffectedSubgraphs() {
    // First, update vertices affected by deletion (mark all descendants as affected)
    log("Updating subtrees of deletion-affected vertices");
    
    std::atomic<bool> hasDeleteAffected(true);
    int deleteIterations = 0;
    
    // Process deletions first to mark all vertices in disconnected subtrees
    while (hasDeleteAffected.load() && deleteIterations++ < maxIterations) {
        hasDeleteAffected.store(false);
        
        #pragma omp parallel
        {
            bool threadHasAffected = false;
            
            // Use dynamic scheduling for better load balancing
            #pragma omp for schedule(dynamic, 64)
            for (int v = 0; v < localTree.numVertices; v++) {
                if (localTree.affectedDel[v]) {
                    // Reset this flag
                    localTree.affectedDel[v] = false;
                    threadHasAffected = true;
                    
                    // Update the disconnected subtree
                    updateDisconnectedSubtree(v);
                }
            }
            
            if (threadHasAffected) {
                hasDeleteAffected.store(true);
            }
        }
        
        // Only exchange boundary information periodically to reduce overhead
        if (hasDeleteAffected.load() && (deleteIterations % 5 == 0)) {
            try {
                exchangeBoundaryInfo();
            } catch (const std::exception& e) {
                log("Warning: Exception during deletion boundary exchange: " + std::string(e.what()));
            }
        }
    }
    
    if (deleteIterations >= maxIterations) {
        log("Warning: Maximum iterations reached while processing deletions. Algorithm may not have converged.");
    }
    
    // Then update all affected vertices using a priority-based approach
    log("Updating all affected vertices");
    
    // Structure to prioritize processing vertices with lower distances first
    typedef std::pair<float, int> DistanceVertex;
    
    // Global priority queue for affected vertices
    std::priority_queue<DistanceVertex, std::vector<DistanceVertex>, std::greater<DistanceVertex>> globalQueue;
    
    // Initialize the priority queue with all affected vertices
    #pragma omp parallel
    {
        // Thread-local queue to reduce contention
        std::vector<DistanceVertex> localQueue;
        
        #pragma omp for schedule(dynamic, 64)
        for (int v = 0; v < localTree.numVertices; v++) {
            if (localTree.affected[v]) {
                localQueue.emplace_back(localTree.distance[v], v);
                localTree.affected[v] = false;  // Reset flag
            }
        }
        
        // Add to global queue with single critical region
        #pragma omp critical(queue_update)
        {
            for (const auto& item : localQueue) {
                if (!std::isinf(item.first)) {  // Only add vertices with finite distances
                    globalQueue.push(item);
                }
            }
        }
    }
    
    // Now process the vertices in priority order
    int iterations = 0;
    bool changes = true;
    
    while (changes && iterations++ < maxIterations && !globalQueue.empty()) {
        changes = false;
        
        // Process a batch of vertices from the queue
        const int PROCESS_BATCH = 128;
        std::vector<DistanceVertex> currentBatch;
        
        for (int i = 0; i < PROCESS_BATCH && !globalQueue.empty(); i++) {
            currentBatch.push_back(globalQueue.top());
            globalQueue.pop();
        }
        
        if (currentBatch.empty()) {
            break;
        }
        
        // Process this batch in parallel
        std::vector<DistanceVertex> newAffected;
        
        #pragma omp parallel
        {
            std::vector<DistanceVertex> threadAffected;
            
            #pragma omp for schedule(dynamic, 8)
            for (size_t i = 0; i < currentBatch.size(); i++) {
                int currentV = currentBatch[i].second;
                float currentDist = currentBatch[i].first;
                
                // Skip if distance has changed since enqueuing
                if (std::abs(currentDist - localTree.distance[currentV]) > 1e-6) {
                    continue;
                }
                
                // Skip vertices with infinite distance
                if (std::isinf(currentDist)) {
                    continue;
                }
                
                int originalV = localGraph.reverseMapVertex(currentV);
                if (originalV == -1) {
                    continue;  // Skip if vertex doesn't exist in the local graph
                }
                
                // Process all neighbors
                for (const auto& edge : localGraph.getNeighbors(originalV)) {
                    int n = edge.first;
                    float weight = edge.second;
                    
                    // Try to relax the edge
                    float newDist = currentDist + weight;
                    
                    if (newDist < localTree.distance[n]) {
                        #pragma omp critical(tree_update)
                        {
                            // Double-check after lock
                            if (newDist < localTree.distance[n]) {
                                localTree.distance[n] = newDist;
                                localTree.parent[n] = currentV;
                                threadAffected.emplace_back(newDist, n);
                            }
                        }
                    }
                }
            }
            
            // Collect affected vertices
            #pragma omp critical(queue_update)
            {
                newAffected.insert(newAffected.end(), threadAffected.begin(), threadAffected.end());
            }
        }
        
        // Add new affected vertices to the queue
        for (const auto& item : newAffected) {
            globalQueue.push(item);
            changes = true;
        }
        
        // Exchange boundary information periodically
        if (changes && (iterations % 10 == 0 || iterations == 1)) {
            try {
                exchangeBoundaryInfo();
                
                // Check if ghost vertices need to be added to processing queue
                #pragma omp parallel
                {
                    std::vector<DistanceVertex> ghostQueue;
                    
                    #pragma omp for schedule(dynamic)
                    for (size_t i = 0; i < ghostVertices.size(); i++) {
                        int localID = mapGlobalToLocal(ghostVertices[i]);
                        if (localID != -1 && localTree.affected[localID]) {
                            if (!std::isinf(localTree.distance[localID])) {
                                ghostQueue.emplace_back(localTree.distance[localID], localID);
                            }
                            localTree.affected[localID] = false;
                        }
                    }
                    
                    #pragma omp critical(queue_update)
                    {
                        for (const auto& item : ghostQueue) {
                            globalQueue.push(item);
                        }
                    }
                }
            } catch (const std::exception& e) {
                log("Warning: Exception during boundary exchange: " + std::string(e.what()));
            }
        }
    }
    
    if (iterations >= maxIterations) {
        log("Warning: Maximum iteration count reached (" + std::to_string(iterations) + 
            "). Algorithm may not have converged.");
    } else {
        log("Converged after " + std::to_string(iterations) + " iterations");
    }
}

void ParallelSSSP::updateDisconnectedSubtree(int vertex) {
    // More efficient implementation using stack instead of recursion
    std::vector<int> stack;
    stack.push_back(vertex);
    
    while (!stack.empty()) {
        int v = stack.back();
        stack.pop_back();
        
        std::vector<int> children;
        
        // First collect all children to reduce critical section time
        #pragma omp critical(tree_read)
        {
            for (int child = 0; child < localTree.numVertices; child++) {
                if (child != v && localTree.parent[child] == v) {
                    children.push_back(child);
                }
            }
        }
        
        // Then update each child
        for (int child : children) {
            #pragma omp critical(tree_update)
            {
                localTree.distance[child] = std::numeric_limits<float>::infinity();
                localTree.parent[child] = -1;
                localTree.affectedDel[child] = true;
                localTree.affected[child] = true;
            }
            
            // Add child to stack
            stack.push_back(child);
        }
    }
}

void ParallelSSSP::exchangeBoundaryInfo() {
    MPI_Datatype boundaryInfoType = createBoundaryInfoType();
    
    // Prepare boundary information to send - only send ghost vertices with finite distances
    std::vector<BoundaryInfo> boundaryInfoToSend;
    
    for (int globalID : ghostVertices) {
        int localID = mapGlobalToLocal(globalID);
        if (localID != -1 && !std::isinf(localTree.distance[localID])) {
            int parentLocalID = localTree.parent[localID];
            int parentGlobalID = (parentLocalID != -1) ? mapLocalToGlobal(parentLocalID) : -1;
            
            boundaryInfoToSend.emplace_back(
                globalID,
                localTree.distance[localID],
                parentGlobalID
            );
        }
    }
    
    // Send boundary information size
    int sendSize = boundaryInfoToSend.size();
    std::vector<int> recvSizes(size);
    MPI_Allgather(&sendSize, 1, MPI_INT, recvSizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Calculate displacements for MPI_Allgatherv
    std::vector<int> displacements(size, 0);
    for (int i = 1; i < size; i++) {
        displacements[i] = displacements[i-1] + recvSizes[i-1];
    }
    
    // Total boundary information to receive
    int totalRecvSize = std::accumulate(recvSizes.begin(), recvSizes.end(), 0);
    
    if (totalRecvSize > 0) {
        std::vector<BoundaryInfo> allBoundaryInfo(totalRecvSize);
        
        // Gather boundary information from all processes
        MPI_Allgatherv(
            boundaryInfoToSend.data(), sendSize, boundaryInfoType,
            allBoundaryInfo.data(), recvSizes.data(), displacements.data(), boundaryInfoType,
            MPI_COMM_WORLD
        );
        
        // Update local tree with received boundary information
        int updatedCount = 0;
        
        #pragma omp parallel
        {
            int threadUpdated = 0;
            std::vector<std::pair<int, BoundaryInfo>> threadUpdates;
            
            #pragma omp for schedule(dynamic, 64)
            for (int i = 0; i < totalRecvSize; i++) {
                const auto& info = allBoundaryInfo[i];
                int localID = mapGlobalToLocal(info.vertexID);
                
                if (localID != -1 && info.distance < localTree.distance[localID]) {
                    // Store update instead of applying immediately
                    threadUpdates.push_back({localID, info});
                    threadUpdated++;
                }
            }
            
            // Apply updates with single critical section per thread
            if (threadUpdated > 0) {
                #pragma omp critical(tree_update)
                {
                    for (const auto& update : threadUpdates) {
                        int localID = update.first;
                        const BoundaryInfo& info = update.second;
                        
                        // Double-check after acquiring lock
                        if (info.distance < localTree.distance[localID]) {
                            localTree.distance[localID] = info.distance;
                            
                            if (info.parent != -1) {
                                int parentLocalID = mapGlobalToLocal(info.parent);
                                if (parentLocalID != -1) {
                                    localTree.parent[localID] = parentLocalID;
                                }
                            }
                            
                            localTree.affected[localID] = true;
                            updatedCount++;
                        }
                    }
                }
            }
            
            #pragma omp atomic
            updatedCount += threadUpdated;
        }
        
        log("Exchange: sent " + std::to_string(sendSize) + 
            ", received " + std::to_string(totalRecvSize) + 
            ", updated " + std::to_string(updatedCount) + " vertices");
    } else {
        log("No boundary information to exchange");
    }
    
    MPI_Type_free(&boundaryInfoType);
}

MPI_Datatype ParallelSSSP::createBoundaryInfoType() {
    MPI_Datatype boundaryInfoType;
    
    // Define the structure layout
    MPI_Datatype types[3] = {MPI_INT, MPI_FLOAT, MPI_INT};
    int blocklengths[3] = {1, 1, 1};
    
    // Calculate displacements
    MPI_Aint displacements[3];
    BoundaryInfo dummyInfo;
    
    MPI_Aint base_address;
    MPI_Get_address(&dummyInfo, &base_address);
    MPI_Get_address(&dummyInfo.vertexID, &displacements[0]);
    MPI_Get_address(&dummyInfo.distance, &displacements[1]);
    MPI_Get_address(&dummyInfo.parent, &displacements[2]);
    
    // Make relative to base address
    displacements[0] = MPI_Aint_diff(displacements[0], base_address);
    displacements[1] = MPI_Aint_diff(displacements[1], base_address);
    displacements[2] = MPI_Aint_diff(displacements[2], base_address);
    
    // Create and commit the MPI datatype
    MPI_Type_create_struct(3, blocklengths, displacements, types, &boundaryInfoType);
    MPI_Type_commit(&boundaryInfoType);
    
    return boundaryInfoType;
}

void ParallelSSSP::log(const std::string& message) {
    if (verbose) {
        std::stringstream ss;
        ss << "[Rank " << rank << "] " << message;
        std::cerr << ss.str() << std::endl;
    }
}
