#ifndef SSSP_H
#define SSSP_H

#include "graph.h"
#include <vector>
#include <queue>
#include <mpi.h>

struct SSSPResult {
    std::vector<Weight> distances;
    std::vector<Vertex> parent;
    std::vector<bool> affected;
    std::vector<bool> affected_del;
};

struct BoundaryData {
    Vertex vertex;
    Weight distance;
    Vertex parent;
};

// Initial SSSP calculation using Dijkstra's algorithm (parallel version)
SSSPResult calculate_initial_sssp_parallel(const Graph& graph, Vertex source, int rank, int num_procs);

// Update SSSP for a dynamic graph (parallel version)
SSSPResult update_sssp_parallel(const Graph& graph, const SSSPResult& current_sssp, 
                                const std::vector<EdgeChange>& changes, Vertex source, 
                                int rank, int num_procs, int num_threads);

// Process changed edges and identify affected vertices (for local subgraph)
void process_changed_edges_local(const Graph& graph, const SSSPResult& current_sssp, 
                                SSSPResult& updated_sssp, const std::vector<EdgeChange>& changes);

// Update affected vertices (for local subgraph)
void update_affected_vertices_local(const Graph& graph, SSSPResult& updated_sssp, int num_threads);

// Synchronize boundary vertices across processes (with reduced communication)
void synchronize_boundaries(const Graph& graph, SSSPResult& sssp, int rank, int num_procs);

// Sequential update for small graphs or when parallel is not beneficial
SSSPResult update_sssp_sequential(const Graph& graph, const SSSPResult& current_sssp, 
                                 const std::vector<EdgeChange>& changes, Vertex source);

// Broadcast SSSP results from root to all processes
void broadcast_sssp_result(SSSPResult& sssp, int root_rank, MPI_Comm comm);

// Heuristic to determine if parallel processing is beneficial
bool should_use_parallel(const Graph& graph, int num_changes, int num_procs);

// Process new edges directly to ensure shortest paths are correctly updated
void check_new_edges(const Graph& graph, SSSPResult& sssp, 
                    const std::vector<EdgeChange>& changes);

#endif // SSSP_H
