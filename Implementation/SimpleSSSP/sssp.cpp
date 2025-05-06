#include "sssp.h"
#include "metis_wrapper.h"
#include <omp.h>
#include <queue>
#include <algorithm>
#include <iostream>
#include <unordered_set>

// Priority queue for Dijkstra's algorithm
typedef std::pair<Weight, Vertex> WeightVertexPair;
typedef std::priority_queue<WeightVertexPair, std::vector<WeightVertexPair>, std::greater<WeightVertexPair>> MinPriorityQueue;

SSSPResult calculate_initial_sssp_parallel(const Graph& graph, Vertex source, int rank, int num_procs) {
    int n = graph.get_num_vertices();
    std::vector<Weight> distances(n, INF);
    std::vector<Vertex> parent(n, -1);
    std::vector<bool> affected(n, false);
    std::vector<bool> affected_del(n, false);
    
    if (graph.get_partition(source) == rank || rank == 0) {
        distances[source] = 0;
        MinPriorityQueue pq;
        pq.push(std::make_pair(0, source));
        std::vector<bool> visited(n, false);
        
        while (!pq.empty()) {
            Vertex u = pq.top().second;
            Weight dist_u = pq.top().first;
            pq.pop();
            
            if (visited[u]) continue;
            visited[u] = true;
            
            for (const auto& edge : graph.get_neighbors(u)) {
                Vertex v = edge.first;
                Weight weight = edge.second;
                if (!visited[v] && dist_u + weight < distances[v]) {
                    distances[v] = dist_u + weight;
                    parent[v] = u;
                    pq.push(std::make_pair(distances[v], v));
                }
            }
        }
    }
    
    struct DistParent {
        Weight dist;
        Vertex par;
    };
    std::vector<DistParent> dist_parent_buffer(n);
    
    int source_rank = graph.get_partition(source);
    if (source_rank == -1) source_rank = 0;
    
    if (rank == source_rank) {
        for (int i = 0; i < n; i++) {
            dist_parent_buffer[i].dist = distances[i];
            dist_parent_buffer[i].par = parent[i];
        }
    }
    
    MPI_Datatype dist_parent_type;
    int blocklengths[2] = {1, 1};
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_INT};
    MPI_Aint offsets[2];
    DistParent temp;
    MPI_Aint base_address;
    MPI_Get_address(&temp, &base_address);
    MPI_Get_address(&temp.dist, &offsets[0]);
    MPI_Get_address(&temp.par, &offsets[1]);
    for (int i = 0; i < 2; i++) {
        offsets[i] = MPI_Aint_diff(offsets[i], base_address);
    }
    MPI_Type_create_struct(2, blocklengths, offsets, types, &dist_parent_type);
    MPI_Type_commit(&dist_parent_type);
    
    MPI_Bcast(dist_parent_buffer.data(), n, dist_parent_type, source_rank, MPI_COMM_WORLD);
    
    if (rank != source_rank) {
        for (int i = 0; i < n; i++) {
            distances[i] = dist_parent_buffer[i].dist;
            parent[i] = dist_parent_buffer[i].par;
        }
    }
    
    MPI_Type_free(&dist_parent_type);
    
    return {distances, parent, affected, affected_del};
}

void check_new_edges(const Graph& graph, SSSPResult& sssp, const std::vector<EdgeChange>& changes) {
    bool changes_made = false;
    for (const EdgeChange& change : changes) {
        if (change.operation == '+') {
            Vertex u = change.source;
            Vertex v = change.target;
            Weight weight = change.weight;
            
            if (sssp.distances[u] != INF && sssp.distances[u] + weight < sssp.distances[v]) {
                sssp.distances[v] = sssp.distances[u] + weight;
                sssp.parent[v] = u;
                sssp.affected[v] = true;
                changes_made = true;
            }
            if (sssp.distances[v] != INF && sssp.distances[v] + weight < sssp.distances[u]) {
                sssp.distances[u] = sssp.distances[v] + weight;
                sssp.parent[u] = v;
                sssp.affected[u] = true;
                changes_made = true;
            }
        }
    }
    
    if (changes_made) {
        std::vector<Vertex> affected_vertices;
        for (size_t i = 0; i < sssp.affected.size(); i++) {
            if (sssp.affected[i]) {
                affected_vertices.push_back(i);
            }
        }
        
        std::vector<bool> processed(sssp.distances.size(), false);
        for (Vertex v : affected_vertices) {
            if (processed[v]) continue;
            processed[v] = true;
            for (const auto& edge : graph.get_neighbors(v)) {
                Vertex neighbor = edge.first;
                Weight weight = edge.second;
                if (sssp.distances[v] + weight < sssp.distances[neighbor]) {
                    sssp.distances[neighbor] = sssp.distances[v] + weight;
                    sssp.parent[neighbor] = v;
                    sssp.affected[neighbor] = true;
                    if (!processed[neighbor]) {
                        affected_vertices.push_back(neighbor);
                    }
                }
            }
        }
    }
}

SSSPResult update_sssp_parallel(const Graph& graph, const SSSPResult& current_sssp, 
                             const std::vector<EdgeChange>& changes, Vertex source, 
                             int rank, int num_procs, int num_threads) {
    SSSPResult updated_sssp = current_sssp;
    size_t n = graph.get_num_vertices();
    if (n > updated_sssp.distances.size()) {
        updated_sssp.distances.resize(n, INF);
        updated_sssp.parent.resize(n, -1);
        updated_sssp.affected.resize(n, false);
        updated_sssp.affected_del.resize(n, false);
    }
    
    std::fill(updated_sssp.affected.begin(), updated_sssp.affected.end(), false);
    std::fill(updated_sssp.affected_del.begin(), updated_sssp.affected_del.end(), false);
    
    process_changed_edges_local(graph, current_sssp, updated_sssp, changes);
    check_new_edges(graph, updated_sssp, changes);
    
    synchronize_boundaries(graph, updated_sssp, rank, num_procs);
    update_affected_vertices_local(graph, updated_sssp, num_threads);
    
    bool global_changes = true;
    while (global_changes) {
        synchronize_boundaries(graph, updated_sssp, rank, num_procs);
        bool local_changes = false;
        for (Vertex v : graph.get_local_vertices()) {
            if (v >= 0 && static_cast<size_t>(v) < updated_sssp.affected.size() && updated_sssp.affected[v]) {
                local_changes = true;
                break;
            }
        }
        
        MPI_Allreduce(&local_changes, &global_changes, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        if (global_changes) {
            update_affected_vertices_local(graph, updated_sssp, num_threads);
        }
    }
    
    return updated_sssp;
}

void process_changed_edges_local(const Graph& graph, const SSSPResult& current_sssp, 
                               SSSPResult& updated_sssp, const std::vector<EdgeChange>& changes) {
    for (const EdgeChange& change : changes) {
        Vertex u = change.source;
        Vertex v = change.target;
        Weight weight = change.weight;
        char op = change.operation;
        
        bool own_u = graph.is_local_vertex(u);
        bool own_v = graph.is_local_vertex(v);
        
        if (!own_u && !own_v) continue;
        
        if (op == '+') {
            if (u < updated_sssp.distances.size() && v < updated_sssp.distances.size()) {
                if (updated_sssp.distances[u] != INF && updated_sssp.distances[u] + weight < updated_sssp.distances[v]) {
                    updated_sssp.distances[v] = updated_sssp.distances[u] + weight;
                    updated_sssp.parent[v] = u;
                    updated_sssp.affected[v] = true;
                } else if (updated_sssp.distances[v] != INF && updated_sssp.distances[v] + weight < updated_sssp.distances[u]) {
                    updated_sssp.distances[u] = updated_sssp.distances[v] + weight;
                    updated_sssp.parent[u] = v;
                    updated_sssp.affected[u] = true;
                } else {
                    if (updated_sssp.distances[u] != INF) updated_sssp.affected[u] = true;
                    if (updated_sssp.distances[v] != INF) updated_sssp.affected[v] = true;
                }
            }
        } else if (op == '-') {
            if (u < updated_sssp.distances.size() && v < updated_sssp.distances.size() && 
                u < updated_sssp.parent.size() && v < updated_sssp.parent.size()) {
                if (updated_sssp.parent[v] == u) {
                    updated_sssp.distances[v] = INF;
                    updated_sssp.parent[v] = -1;
                    updated_sssp.affected_del[v] = true;
                    updated_sssp.affected[v] = true;
                } else if (updated_sssp.parent[u] == v) {
                    updated_sssp.distances[u] = INF;
                    updated_sssp.parent[u] = -1;
                    updated_sssp.affected_del[u] = true;
                    updated_sssp.affected[u] = true;
                } else {
                    updated_sssp.affected[u] = true;
                    updated_sssp.affected[v] = true;
                }
            }
        }
    }
}

void update_affected_vertices_local(const Graph& graph, SSSPResult& updated_sssp, int num_threads) {
    size_t n = updated_sssp.distances.size();
    bool has_del_affected = true;
    
    // First phase: handle deleted edges and their affected subtrees
    while (has_del_affected) {
        has_del_affected = false;
        
        #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
        for (size_t i = 0; i < graph.get_local_vertices().size(); i++) {
            Vertex v = graph.get_local_vertices()[i];
            if (v >= 0 && static_cast<size_t>(v) < updated_sssp.affected_del.size() && updated_sssp.affected_del[v]) {
                updated_sssp.affected_del[v] = false;
                updated_sssp.affected[v] = true;
                
                // Find children affected by the deletion
                for (size_t c = 0; c < n; c++) {
                    if (c < updated_sssp.parent.size() && updated_sssp.parent[c] == v) {
                        if (c < updated_sssp.distances.size())
                            updated_sssp.distances[c] = INF;
                        if (c < updated_sssp.affected_del.size())
                            updated_sssp.affected_del[c] = true;
                        if (c < updated_sssp.affected.size())
                            updated_sssp.affected[c] = true;
                        
                        #pragma omp critical
                        {
                            has_del_affected = true;
                        }
                    }
                }
            }
        }
        
        // Synchronize has_del_affected across all processes
        bool global_has_del_affected;
        MPI_Allreduce(&has_del_affected, &global_has_del_affected, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        has_del_affected = global_has_del_affected;
    }
    
    // Second phase: update distances for affected vertices
    bool changes = true;
    int iteration_count = 0;
    const int MAX_ITERATIONS = 100;
    
    while (changes && iteration_count < MAX_ITERATIONS) {
        changes = false;
        iteration_count++;
        
        #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
        for (size_t i = 0; i < graph.get_local_vertices().size(); i++) {
            Vertex v = graph.get_local_vertices()[i];
            if (v >= 0 && static_cast<size_t>(v) < updated_sssp.affected.size()) {
                bool v_was_affected = updated_sssp.affected[v];
                
                if (v_was_affected) {
                    updated_sssp.affected[v] = false;
                }
                
                // Update distances through neighbors
                for (const auto& edge : graph.get_neighbors(v)) {
                    Vertex neighbor = edge.first;
                    Weight weight = edge.second;
                    
                    // Bounds checking
                    if (neighbor >= 0 && static_cast<size_t>(neighbor) < updated_sssp.distances.size() &&
                        v < updated_sssp.distances.size()) {
                        
                        // Update neighbor if we found a shorter path
                        if (updated_sssp.distances[v] != INF && 
                            updated_sssp.distances[v] + weight < updated_sssp.distances[neighbor]) {
                            #pragma omp critical
                            {
                                updated_sssp.distances[neighbor] = updated_sssp.distances[v] + weight;
                                updated_sssp.parent[neighbor] = v;
                                updated_sssp.affected[neighbor] = true;
                                changes = true;
                            }
                        }
                        
                        // Update ourselves if we found a shorter path
                        if (updated_sssp.distances[neighbor] != INF && 
                            updated_sssp.distances[neighbor] + weight < updated_sssp.distances[v]) {
                            #pragma omp critical
                            {
                                updated_sssp.distances[v] = updated_sssp.distances[neighbor] + weight;
                                updated_sssp.parent[v] = neighbor;
                                updated_sssp.affected[v] = true;
                                changes = true;
                            }
                        }
                    }
                }
            }
        }
        
        // Synchronize changes across all processes
        bool global_changes;
        MPI_Allreduce(&changes, &global_changes, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        changes = global_changes;
    }
    
    if (iteration_count >= MAX_ITERATIONS) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cerr << "Warning: Maximum iterations reached in update_affected_vertices_local" << std::endl;
        }
    }
}

void synchronize_boundaries(const Graph& graph, SSSPResult& sssp, int rank, int num_procs) {
    // Prepare data to send to other processes
    std::vector<std::vector<BoundaryData>> send_buffers(num_procs);
    
    // Collect data from local vertices that have neighbors in other partitions
    for (Vertex v : graph.get_local_vertices()) {
        for (const auto& edge : graph.get_neighbors(v)) {
            Vertex neighbor = edge.first;
            int neighbor_part = graph.get_partition(neighbor);
            if (neighbor_part != rank && neighbor_part >= 0 && neighbor_part < num_procs) {
                send_buffers[neighbor_part].push_back({v, sssp.distances[v], sssp.parent[v]});
            }
        }
    }
    
    // Count number of items to send to each process
    std::vector<int> send_counts(num_procs, 0);
    std::vector<int> recv_counts(num_procs, 0);
    for (int p = 0; p < num_procs; p++) {
        send_counts[p] = send_buffers[p].size();
    }
    
    // Exchange counts with all processes
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Calculate displacements for sending and receiving
    std::vector<int> send_displs(num_procs, 0);
    std::vector<int> recv_displs(num_procs, 0);
    int total_send = 0, total_recv = 0;
    for (int p = 0; p < num_procs; p++) {
        send_displs[p] = total_send;
        recv_displs[p] = total_recv;
        total_send += send_counts[p];
        total_recv += recv_counts[p];
    }
    
    // Create buffer for sending data
    std::vector<BoundaryData> send_data(total_send);
    for (int p = 0; p < num_procs; p++) {
        if (send_counts[p] > 0) {
            std::copy(send_buffers[p].begin(), send_buffers[p].end(), send_data.begin() + send_displs[p]);
        }
    }
    
    // Create buffer for receiving data
    std::vector<BoundaryData> recv_data(total_recv);
    
    // Create MPI datatype for BoundaryData
    MPI_Datatype boundary_type;
    int blocklengths[3] = {1, 1, 1};
    MPI_Datatype types[3] = {MPI_INT, MPI_DOUBLE, MPI_INT};
    MPI_Aint offsets[3];
    BoundaryData temp;
    MPI_Aint base_address;
    MPI_Get_address(&temp, &base_address);
    MPI_Get_address(&temp.vertex, &offsets[0]);
    MPI_Get_address(&temp.distance, &offsets[1]);
    MPI_Get_address(&temp.parent, &offsets[2]);
    for (int i = 0; i < 3; i++) {
        offsets[i] = MPI_Aint_diff(offsets[i], base_address);
    }
    MPI_Type_create_struct(3, blocklengths, offsets, types, &boundary_type);
    MPI_Type_commit(&boundary_type);
    
    // Debug output ONLY from rank 0 and ONLY if there's something to send/receive
    /*
    if (rank == 0 && g_verbose_logging) {
        std::cout << "Rank " << rank << " send_counts: ";
        for (int p = 0; p < num_procs; p++) std::cout << send_counts[p] << " ";
        std::cout << "\nRank " << rank << " recv_counts: ";
        for (int p = 0; p < num_procs; p++) std::cout << recv_counts[p] << " ";
        std::cout << "\nTotal send: " << total_send << ", Total recv: " << total_recv << std::endl;
    }*/
    
    // Exchange data with all processes
    MPI_Alltoallv(
        send_data.data(), send_counts.data(), send_displs.data(), boundary_type,
        recv_data.data(), recv_counts.data(), recv_displs.data(), boundary_type,
        MPI_COMM_WORLD
    );
    
    // Process received data
    for (const auto& data : recv_data) {
        Vertex v = data.vertex;
        Weight dist = data.distance;
        Vertex par = data.parent;
        
        // Ensure v is within bounds
        if (v >= 0 && static_cast<size_t>(v) < sssp.distances.size()) {
            if (dist < sssp.distances[v]) {
                sssp.distances[v] = dist;
                sssp.parent[v] = par;
                // Mark neighbors as affected
                for (const auto& edge : graph.get_neighbors(v)) {
                    Vertex neighbor = edge.first;
                    if (graph.is_local_vertex(neighbor)) {
                        sssp.affected[neighbor] = true;
                    }
                }
            }
        }
    }
    
    MPI_Type_free(&boundary_type);
}

void broadcast_sssp_result(SSSPResult& sssp, int root_rank, MPI_Comm comm) {
    size_t n = sssp.distances.size();
    struct DistParentFlags {
        Weight dist;
        Vertex par;
        bool aff;
        bool aff_del;
    };
    std::vector<DistParentFlags> buffer(n);
    
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    if (rank == root_rank) {
        for (size_t i = 0; i < n; i++) {
            buffer[i].dist = sssp.distances[i];
            buffer[i].par = sssp.parent[i];
            buffer[i].aff = sssp.affected[i];
            buffer[i].aff_del = sssp.affected_del[i];
        }
    }
    
    MPI_Datatype dist_parent_flags_type;
    int blocklengths[4] = {1, 1, 1, 1};
    MPI_Datatype types[4] = {MPI_DOUBLE, MPI_INT, MPI_C_BOOL, MPI_C_BOOL};
    MPI_Aint offsets[4];
    DistParentFlags temp;
    MPI_Aint base_address;
    MPI_Get_address(&temp, &base_address);
    MPI_Get_address(&temp.dist, &offsets[0]);
    MPI_Get_address(&temp.par, &offsets[1]);
    MPI_Get_address(&temp.aff, &offsets[2]);
    MPI_Get_address(&temp.aff_del, &offsets[3]);
    for (int i = 0; i < 4; i++) {
        offsets[i] = MPI_Aint_diff(offsets[i], base_address);
    }
    MPI_Type_create_struct(4, blocklengths, offsets, types, &dist_parent_flags_type);
    MPI_Type_commit(&dist_parent_flags_type);
    
    MPI_Bcast(buffer.data(), n, dist_parent_flags_type, root_rank, comm);
    
    if (rank != root_rank) {
        for (size_t i = 0; i < n; i++) {
            sssp.distances[i] = buffer[i].dist;
            sssp.parent[i] = buffer[i].par;
            sssp.affected[i] = buffer[i].aff;
            sssp.affected_del[i] = buffer[i].aff_del;
        }
    }
    
    MPI_Type_free(&dist_parent_flags_type);
}

SSSPResult update_sssp_sequential(const Graph& graph, const SSSPResult& current_sssp, 
                                 const std::vector<EdgeChange>& changes, Vertex source) {
    SSSPResult updated_sssp = current_sssp;
    size_t n = graph.get_num_vertices();
    if (n > updated_sssp.distances.size()) {
        updated_sssp.distances.resize(n, INF);
        updated_sssp.parent.resize(n, -1);
        updated_sssp.affected.resize(n, false);
        updated_sssp.affected_del.resize(n, false);
    }
    
    std::fill(updated_sssp.affected.begin(), updated_sssp.affected.end(), false);
    std::fill(updated_sssp.affected_del.begin(), updated_sssp.affected_del.end(), false);
    
    for (const EdgeChange& change : changes) {
        Vertex u = change.source;
        Vertex v = change.target;
        Weight weight = change.weight;
        char op = change.operation;
        
        if (op == '+') {
            if (updated_sssp.distances[u] != INF && updated_sssp.distances[u] + weight < updated_sssp.distances[v]) {
                updated_sssp.distances[v] = updated_sssp.distances[u] + weight;
                updated_sssp.parent[v] = u;
                updated_sssp.affected[v] = true;
            } else if (updated_sssp.distances[v] != INF && updated_sssp.distances[v] + weight < updated_sssp.distances[u]) {
                updated_sssp.distances[u] = updated_sssp.distances[v] + weight;
                updated_sssp.parent[u] = v;
                updated_sssp.affected[u] = true;
            } else {
                if (updated_sssp.distances[u] != INF) updated_sssp.affected[u] = true;
                if (updated_sssp.distances[v] != INF) updated_sssp.affected[v] = true;
            }
        } else if (op == '-') {
            if (updated_sssp.parent[v] == u) {
                updated_sssp.distances[v] = INF;
                updated_sssp.parent[v] = -1;
                updated_sssp.affected_del[v] = true;
                updated_sssp.affected[v] = true;
            } else if (updated_sssp.parent[u] == v) {
                updated_sssp.distances[u] = INF;
                updated_sssp.parent[u] = -1;
                updated_sssp.affected_del[u] = true;
                updated_sssp.affected[u] = true;
            } else {
                updated_sssp.affected[u] = true;
                updated_sssp.affected[v] = true;
            }
        }
    }
    
    bool has_changes = true;
    while (has_changes) {
        has_changes = false;
        for (Vertex v = 0; v < static_cast<Vertex>(n); v++) {
            if (updated_sssp.affected[v]) {
                updated_sssp.affected[v] = false;
                for (const auto& edge : graph.get_neighbors(v)) {
                    Vertex neighbor = edge.first;
                    Weight weight = edge.second;
                    if (updated_sssp.distances[v] != INF && 
                        updated_sssp.distances[v] + weight < updated_sssp.distances[neighbor]) {
                        updated_sssp.distances[neighbor] = updated_sssp.distances[v] + weight;
                        updated_sssp.parent[neighbor] = v;
                        updated_sssp.affected[neighbor] = true;
                        has_changes = true;
                    }
                    if (updated_sssp.distances[neighbor] != INF && 
                        updated_sssp.distances[neighbor] + weight < updated_sssp.distances[v]) {
                        updated_sssp.distances[v] = updated_sssp.distances[neighbor] + weight;
                        updated_sssp.parent[v] = neighbor;
                        updated_sssp.affected[v] = true;
                        has_changes = true;
                    }
                }
            }
        }
    }
    
    return updated_sssp;
}

bool should_use_parallel(const Graph& graph, int num_changes, int num_procs) {
    int min_vertices_for_parallel = 1000;
    int min_changes_for_parallel = 50;
    return (num_procs > 1) && 
           (graph.get_num_vertices() > min_vertices_for_parallel || 
            num_changes > min_changes_for_parallel);
}
