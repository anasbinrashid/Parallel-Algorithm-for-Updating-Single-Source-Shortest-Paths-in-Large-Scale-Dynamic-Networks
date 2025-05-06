#include "graph.h"
#include "sssp.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <string>
#include <fstream>
#include <mpi.h>
#include <ctime>
#include <cmath>
#include "metis_wrapper.h"

///bool g_verbose_logging = false;

struct TemporalChange {
    EdgeChange change;
    std::time_t timestamp;
    std::string description;
};

void print_sssp_result(const Graph& graph, const SSSPResult& sssp, Vertex source) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Fixed MPIiteral typo
    if (rank != 0) return; // Only rank 0 prints
    
    std::cout << "Source: " << source << std::endl;
    std::cout << "Vertex | Distance | Parent" << std::endl;
    std::cout << "------------------------" << std::endl;
    
    for (int v = 0; v < graph.get_num_vertices(); v++) {
        std::cout << std::setw(6) << v << " | ";
        if (sssp.distances[v] == INF) {
            std::cout << std::setw(8) << "INF" << " | ";
        } else {
            std::cout << std::setw(8) << std::fixed << std::setprecision(1) << sssp.distances[v] << " | ";
        }
        std::cout << (sssp.parent[v] == -1 ? "-" : std::to_string(sssp.parent[v])) << std::endl;
    }
}

bool compare_sssp_results(const SSSPResult& result1, const SSSPResult& result2) {
    if (result1.distances.size() != result2.distances.size()) {
        return false;
    }
    
    for (size_t i = 0; i < result1.distances.size(); i++) {
        if (result1.distances[i] == INF && result2.distances[i] == INF) {
            continue;
        }
        if (result1.distances[i] == INF || result2.distances[i] == INF) {
            return false;
        }
        if (std::fabs(result1.distances[i] - result2.distances[i]) > 1e-6) {
            std::cout << "Distance mismatch at vertex " << i << ": " 
                      << result1.distances[i] << " vs " << result2.distances[i] << std::endl;
            return false;
        }
        if (result1.parent[i] != result2.parent[i]) {
            std::cout << "Parent mismatch at vertex " << i << ": " 
                      << result1.parent[i] << " vs " << result2.parent[i] << std::endl;
            return false;
        }
    }
    return true;
}

int count_affected_vertices(const SSSPResult& before, const SSSPResult& after) {
    int count = 0;
    for (size_t i = 0; i < before.distances.size(); i++) {
        if (before.distances[i] != after.distances[i] || before.parent[i] != after.parent[i]) {
            count++;
        }
    }
    return count;
}

void output_temporal_dot_file(const Graph& graph, const SSSPResult& sssp, 
                              const std::vector<TemporalChange>& changes, 
                              int current_timestamp, 
                              const std::string& filename) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) return;
    
    std::ofstream dot_file(filename);
    dot_file << "digraph TemporalSSP {" << std::endl;
    dot_file << "  // Graph at timestamp: " << current_timestamp << std::endl;
    
    for (int v = 0; v < graph.get_num_vertices(); v++) {
        std::string distance_str = (sssp.distances[v] == INF) ? "INF" : 
                                   std::to_string(sssp.distances[v]);
        std::string user_name = "User " + std::to_string(v);
        if (v == 0) user_name = "Central Hub (User 0)";
        
        dot_file << "  " << v << " [label=\"" << user_name << "\\nDist: " << distance_str 
                 << "\", style=filled, fillcolor=" 
                 << (v == 0 ? "\"#a0ffa0\"" : (sssp.distances[v] == INF ? "\"#ffcccc\"" : "\"#e0e0ff\""))
                 << "];" << std::endl;
    }
    
    for (int v = 0; v < graph.get_num_vertices(); v++) {
        if (sssp.parent[v] != -1) {
            dot_file << "  " << sssp.parent[v] << " -> " << v 
                     << " [color=red, penwidth=2.0, label=\"" 
                     << graph.get_edge_weight(sssp.parent[v], v) << "\"];" << std::endl;
        }
    }
    
    for (int u = 0; u < graph.get_num_vertices(); u++) {
        for (const auto& edge : graph.get_neighbors(u)) {
            int v = edge.first;
            Weight w = edge.second;
            if (sssp.parent[v] != u && sssp.parent[u] != v) {
                dot_file << "  " << u << " -> " << v 
                         << " [label=\"" << w << "\"];" << std::endl;
            }
        }
    }
    
    dot_file << "  // Latest changes at this timestamp: " << std::endl;
    for (const auto& change : changes) {
        if (change.timestamp == current_timestamp) {
            dot_file << "  // " << change.description << std::endl;
        }
    }
    
    dot_file << "}" << std::endl;
    dot_file.close();
    
    std::cout << "DOT file created: " << filename << std::endl;
    std::cout << "You can visualize this file using Graphviz with the command:" << std::endl;
    std::cout << "  dot -Tpng " << filename << " -o " << filename.substr(0, filename.size() - 4) << ".png" << std::endl;
}

void run_performance_comparison(const Graph& graph, 
                               const SSSPResult& initial_sssp,
                               const std::vector<EdgeChange>& changes,
                               Vertex source,
                               int num_threads) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) return;
    
    std::cout << "\n===== PERFORMANCE COMPARISON =====" << std::endl;
    
    std::cout << "Running OpenMP only version with " << num_threads << " threads..." << std::endl;
    omp_set_num_threads(num_threads);
    
    auto start_time_omp = std::chrono::high_resolution_clock::now();
    SSSPResult updated_sssp_omp = update_sssp_parallel(
        graph, initial_sssp, changes, source, 0, 1, num_threads);
    auto end_time_omp = std::chrono::high_resolution_clock::now();
    auto duration_omp = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time_omp - start_time_omp).count();
    
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    std::cout << "Running OpenMP+MPI version with " << num_threads << " threads and " 
              << num_procs << " processes..." << std::endl;
    
    auto start_time_mpi = std::chrono::high_resolution_clock::now();
    SSSPResult updated_sssp_mpi = update_sssp_parallel(
        graph, initial_sssp, changes, source, rank, num_procs, num_threads);
    auto end_time_mpi = std::chrono::high_resolution_clock::now();
    auto duration_mpi = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time_mpi - start_time_mpi).count();
    
    std::cout << "\n===== PERFORMANCE RESULTS =====" << std::endl;
    std::cout << "OpenMP only time: " << duration_omp / 1000.0 << " ms" << std::endl;
    std::cout << "OpenMP+MPI time: " << duration_mpi / 1000.0 << " ms" << std::endl;
    
    if (duration_omp > 0 && duration_mpi > 0) {
        double speedup = static_cast<double>(duration_omp) / duration_mpi;
        std::cout << "Speedup with MPI: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }
    
    bool results_match = compare_sssp_results(updated_sssp_omp, updated_sssp_mpi);
    std::cout << "Results are " << (results_match ? "identical" : "different") << std::endl;
}

void test_sssp_dynamic_update(int num_threads) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        std::cout << "\n=================================================================================" << std::endl;
        std::cout << "           SOCIAL NETWORK TEMPORAL GRAPH DYNAMIC SSSP TEST                       " << std::endl;
        std::cout << "=================================================================================" << std::endl;
        std::cout << "Using " << num_threads << " OpenMP threads and " << size << " MPI processes" << std::endl;
    }
    
    omp_set_num_threads(num_threads);
    
    Graph social_network(12, false); // Undirected graph
    if (rank == 0) {
        std::cout << "\n===== INITIAL SOCIAL NETWORK =====" << std::endl;
        std::cout << "Creating a social network with 12 users and 19 initial connections." << std::endl;
        std::cout << "User 0 is the central hub (e.g., a popular influencer)" << std::endl;
        std::cout << "Edge weights represent connection strength (lower = stronger)" << std::endl;
    }
    
    social_network.add_edge(0, 1, 1.0);
    social_network.add_edge(0, 7, 1.5);
    social_network.add_edge(0, 11, 0.7);
    social_network.add_edge(1, 2, 1.2);
    social_network.add_edge(1, 9, 1.3);
    social_network.add_edge(2, 3, 0.9);
    social_network.add_edge(3, 4, 1.1);
    social_network.add_edge(3, 9, 1.4);
    social_network.add_edge(4, 5, 0.8);
    social_network.add_edge(4, 10, 1.0);
    social_network.add_edge(5, 6, 1.2);
    social_network.add_edge(5, 10, 1.5);
    social_network.add_edge(6, 7, 0.9);
    social_network.add_edge(6, 8, 1.0);
    social_network.add_edge(7, 8, 1.1);
    social_network.add_edge(8, 11, 0.8);
    social_network.add_edge(9, 10, 1.1);
    social_network.add_edge(9, 11, 1.3);
    social_network.add_edge(10, 8, 1.2);
    
    std::vector<int> partitions = partition_graph(social_network, size);
    social_network.distribute_graph(partitions, rank, size);
    
    Vertex source = 0;
    if (rank == 0) {
        std::cout << "\nComputing shortest paths from the central hub (User " << source << ")..." << std::endl;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    SSSPResult initial_sssp = calculate_initial_sssp_parallel(social_network, source, rank, size);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto init_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    
    if (rank == 0) {
        std::cout << "\n===== INITIAL SHORTEST PATHS =====" << std::endl;
        std::cout << "Calculation time: " << init_duration / 1000.0 << " ms" << std::endl;
        std::cout << "In social network terms, these paths represent the shortest/strongest" << std::endl;
        std::cout << "communication channels between the central hub and other users." << std::endl;
        print_sssp_result(social_network, initial_sssp, source);
    }
    
    std::vector<TemporalChange> temporal_changes;
    std::tm timepoint1 = {0, 15, 9, 2, 4, 125}; // 2025-05-02 09:15
    std::tm timepoint2 = {0, 30, 10, 2, 4, 125}; // 2025-05-02 10:30
    std::tm timepoint3 = {0, 45, 14, 2, 4, 125}; // 2025-05-02 14:45
    
    temporal_changes.push_back({
        EdgeChange(3, 4, 1.1, '-'),
        std::mktime(&timepoint1),
        "User 3 and User 4 disconnect (relationship ended at 9:15 AM)"
    });
    temporal_changes.push_back({
        EdgeChange(9, 11, 1.3, '-'),
        std::mktime(&timepoint1),
        "User 9 and User 11 disconnect (unfollowed at 9:15 AM)"
    });
    temporal_changes.push_back({
        EdgeChange(4, 8, 0.6, '+'),
        std::mktime(&timepoint2),
        "User 4 and User 8 form a strong new connection (new friendship at 10:30 AM)"
    });
    temporal_changes.push_back({
        EdgeChange(11, 3, 0.8, '+'),
        std::mktime(&timepoint2),
        "User 11 and User 3 form a strong new connection (collaboration started at 10:30 AM)"
    });
    temporal_changes.push_back({
        EdgeChange(5, 9, 1.7, '+'),
        std::mktime(&timepoint3),
        "User 5 and User 9 form a weak connection (casual introduction at 2:45 PM)"
    });
    
    std::vector<EdgeChange> changes;
    for (const auto& tc : temporal_changes) {
        changes.push_back(tc.change);
    }
    
    if (rank == 0) {
        std::cout << "\n===== TEMPORAL NETWORK CHANGES =====" << std::endl;
        for (const auto& tc : temporal_changes) {
            char time_buffer[80];
            std::strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&tc.timestamp));
            std::cout << "[" << time_buffer << "] " << tc.description << std::endl;
        }
    }
    
    if (rank == 0) {
        std::cout << "\n===== APPLYING CHANGES TO NETWORK =====" << std::endl;
    }
    for (const auto& change : changes) {
        if (rank == 0) {
            std::cout << "  " << change.operation << " Edge " 
                      << change.source << "->" << change.target 
                      << " (weight: " << change.weight << ")" << std::endl;
        }
        if (change.operation == '+') {
            social_network.add_edge(change.source, change.target, change.weight);
        } else if (change.operation == '-') {
            social_network.remove_edge(change.source, change.target);
        }
    }
    
    if (rank == 0) {
        std::cout << "\n===== UPDATING SHORTEST PATHS DYNAMICALLY =====" << std::endl;
        std::cout << "This simulates real-time path updates as network topology changes" << std::endl;
    }
    
    start_time = std::chrono::high_resolution_clock::now();
    SSSPResult updated_sssp = update_sssp_parallel(
        social_network, initial_sssp, changes, source, rank, size, num_threads);
    end_time = std::chrono::high_resolution_clock::now();
    auto update_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    
    if (rank == 0) {
        std::cout << "\n===== UPDATED SHORTEST PATHS =====" << std::endl;
        std::cout << "Update time: " << update_duration / 1000.0 << " ms" << std::endl;
        print_sssp_result(social_network, updated_sssp, source);
        
        int affected_count = count_affected_vertices(initial_sssp, updated_sssp);
        std::cout << "\n===== IMPACT ANALYSIS =====" << std::endl;
        std::cout << "Number of users affected by network changes: " << affected_count 
                  << " out of " << social_network.get_num_vertices() 
                  << " (" << (affected_count * 100.0 / social_network.get_num_vertices()) << "%)" << std::endl;
        std::cout << "Affected users: ";
        for (size_t i = 0; i < initial_sssp.distances.size(); i++) {
            if (initial_sssp.distances[i] != updated_sssp.distances[i] || 
                initial_sssp.parent[i] != updated_sssp.parent[i]) {
                std::cout << "User " << i << " ";
            }
        }
        std::cout << std::endl;
        
        std::cout << "\n===== VERIFICATION WITH RECOMPUTATION =====" << std::endl;
        std::cout << "Recomputing all shortest paths from scratch..." << std::endl;
        
        start_time = std::chrono::high_resolution_clock::now();
        SSSPResult recomputed_sssp = calculate_initial_sssp_parallel(social_network, source, 0, 1);
        end_time = std::chrono::high_resolution_clock::now();
        auto recompute_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        
        std::cout << "Recomputation time: " << recompute_duration / 1000.0 << " ms" << std::endl;
        
        bool is_correct = compare_sssp_results(updated_sssp, recomputed_sssp);
        std::cout << "\n===== VERIFICATION RESULTS =====" << std::endl;
        std::cout << (is_correct ? "✓ VERIFICATION SUCCESSFUL: Dynamic update matches recomputation!" : 
                                   "✗ VERIFICATION FAILED: Dynamic update does not match recomputation.") << std::endl;
        
        std::cout << "\n===== PERFORMANCE METRICS =====" << std::endl;
        std::cout << "Dynamic update time: " << update_duration / 1000.0 << " ms" << std::endl;
        std::cout << "Recomputation time: " << recompute_duration / 1000.0 << " ms" << std::endl;
        if (recompute_duration > 0) {
            double speedup = static_cast<double>(recompute_duration) / update_duration;
            std::cout << "Speedup from dynamic update: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        }
    }
    
    if (size > 1) {
        run_performance_comparison(social_network, initial_sssp, changes, source, num_threads);
    } else if (rank == 0) {
        std::cout << "\nNote: Running with a single MPI process. To see OpenMP+MPI comparison, run with multiple processes." << std::endl;
    }
    
    if (rank == 0) {
        std::cout << "\n===== GENERATING VISUALIZATION FILES =====" << std::endl;
        output_temporal_dot_file(social_network, initial_sssp, temporal_changes, 0, "social_network_initial.dot");
        output_temporal_dot_file(social_network, updated_sssp, temporal_changes, 
                                std::mktime(&timepoint3), "social_network_final.dot");
        
        std::cout << "\n=================================================================================" << std::endl;
        std::cout << "                      TEST COMPLETED SUCCESSFULLY                                " << std::endl;
        std::cout << "=================================================================================" << std::endl;
    }
}

int main_test(int argc, char* argv[]) {
    int num_threads = omp_get_max_threads();
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-threads" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        }
    }
    
    test_sssp_dynamic_update(num_threads);
    return 0;
}
