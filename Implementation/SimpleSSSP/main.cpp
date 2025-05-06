#include "graph.h"
#include "sssp.h"
#include "metis_wrapper.h"
#include <iostream>
#include <chrono>
#include <omp.h>
#include <mpi.h>
#include <string>
#include <fstream>
#include <iomanip>

void test_sssp_dynamic_update(int num_threads);

void run_sequential_sssp(const Graph& graph, Vertex source, const std::string& updates_file) {
    auto start_time = std::chrono::high_resolution_clock::now();
    int n = graph.get_num_vertices();
    std::vector<Weight> distances(n, INF);
    std::vector<Vertex> parent(n, -1);
    std::vector<bool> visited(n, false);
    
    distances[source] = 0;
    
    for (int i = 0; i < n; i++) {
        Vertex u = -1;
        Weight min_dist = INF;
        for (int v =0; v < n; v++) {
            if (!visited[v] && distances[v] < min_dist) {
                min_dist = distances[v];
                u = v;
            }
        }
        
        if (u == -1) break;
        visited[u] = true;
        
        for (const auto& edge : graph.get_neighbors(u)) {
            Vertex v = edge.first;
            Weight weight = edge.second;
            if (!visited[v] && distances[u] + weight < distances[v]) {
                distances[v] = distances[u] + weight;
                parent[v] = u;
            }
        }
    }
    
    auto seq_init_end_time = std::chrono::high_resolution_clock::now();
    auto seq_init_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        seq_init_end_time - start_time).count();
    
    Graph modified_graph = graph;
    modified_graph.apply_changes(updates_file);
    
    auto seq_update_start_time = std::chrono::high_resolution_clock::now();
    std::fill(distances.begin(), distances.end(), INF);
    std::fill(parent.begin(), parent.end(), -1);
    std::fill(visited.begin(), visited.end(), false);
    
    distances[source] = 0;
    
    for (int i = 0; i < n; i++) {
        Vertex u = -1;
        Weight min_dist = INF;
        for (int v = 0; v < n; v++) {
            if (!visited[v] && distances[v] < min_dist) {
                min_dist = distances[v];
                u = v;
            }
        }
        
        if (u == -1) break;
        visited[u] = true;
        
        for (const auto& edge : modified_graph.get_neighbors(u)) {
            Vertex v = edge.first;
            Weight weight = edge.second;
            if (!visited[v] && distances[u] + weight < distances[v]) {
                distances[v] = distances[u] + weight;
                parent[v] = u;
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto seq_update_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - seq_update_start_time).count();
    
    std::cout << "Sequential timings:" << std::endl;
    std::cout << "  Initial Dijkstra: " << seq_init_duration / 1000.0 << " ms" << std::endl;
    std::cout << "  Recompute: " << seq_update_duration / 1000.0 << " ms" << std::endl;
}

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " [-f <edges_file> -u <updates_file>] [--test [-threads <num_threads>]] [-p <num_parts>] [-s <source>] [-t <num_threads>] [-c <num_changes>] [-i <insertion_ratio>] [-a <async_level>]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -f  Edges file path" << std::endl;
    std::cerr << "  -u  Updates file path" << std::endl;
    std::cerr << "  -p  Number of partitions (default: number of MPI processes)" << std::endl;
    std::cerr << "  -s  Source vertex (default: 0)" << std::endl;
    std::cerr << "  -t  Number of OpenMP threads (default: system max)" << std::endl;
    std::cerr << "  -c  Number of changes to process (default: all)" << std::endl;
    std::cerr << "  -i  Insertion ratio (0.0-1.0, default: 0.5)" << std::endl;
    std::cerr << "  -a  Asynchrony level (default: 1)" << std::endl;
    std::cerr << "  --test  Run the social network test example" << std::endl;
    std::cerr << "  -threads  Number of threads to use in test mode (default: max available)" << std::endl;
}

int main(int argc, char* argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cerr << "Warning: MPI implementation does not support MPI_THREAD_MULTIPLE. Falling back to single-threaded MPI." << std::endl;
        }
    }
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    bool run_test = false;
    int test_threads = omp_get_max_threads();
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--test") {
            run_test = true;
        } else if (arg == "-threads" && i + 1 < argc && run_test) {
            test_threads = std::stoi(argv[++i]);
        }
    }
    
    if (run_test) {
        if (rank == 0) {
            std::cout << "Running social network temporal graph test with " 
                      << test_threads << " threads and " 
                      << size << " MPI processes..." << std::endl;
        }
        
        test_sssp_dynamic_update(test_threads);
        
        MPI_Finalize();
        return 0;
    }
    
    std::string edges_file;
    std::string updates_file;
    int num_parts = size;
    Vertex source = 0;
    int num_threads = omp_get_max_threads();
    int num_changes = -1;
    double insertion_ratio = 0.5;
    int async_level = 1;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-f" && i + 1 < argc) {
            edges_file = argv[++i];
        } else if (arg == "-u" && i + 1 < argc) {
            updates_file = argv[++i];
        } else if (arg == "-p" && i + 1 < argc) {
            num_parts = std::stoi(argv[++i]);
        } else if (arg == "-s" && i + 1 < argc) {
            source = std::stoi(argv[++i]);
        } else if (arg == "-t" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (arg == "-c" && i + 1 < argc) {
            num_changes = std::stoi(argv[++i]);
        } else if (arg == "-i" && i + 1 < argc) {
            insertion_ratio = std::stod(argv[++i]);
        } else if (arg == "-a" && i + 1 < argc) {
            async_level = std::stoi(argv[++i]);
        } else if (arg == "--test" || arg == "-threads") {
            if (arg == "-threads") i++;
        } else {
            if (rank == 0) {
                print_usage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (edges_file.empty() || updates_file.empty()) {
        if (rank == 0) {
            std::cerr << "Error: Edges file and updates file are required for regular mode." << std::endl;
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (num_parts > size) {
        if (rank == 0) {
            std::cout << "Warning: num_parts (" << num_parts << ") > num_processes (" 
                      << size << "). Setting num_parts = num_processes." << std::endl;
        }
        num_parts = size;
    }
    
    omp_set_num_threads(num_threads);
    
    Graph graph(0);
    std::vector<idx_t> partitions;
    
    if (rank == 0) {
        std::cout << "Reading .edges file: " << edges_file << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        graph = Graph::load_from_file(edges_file);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        int num_vertices = graph.get_num_vertices();
        int num_edges = graph.get_num_edges();
        std::cout << "Loaded graph with " << num_vertices << " vertices and " 
                  << num_edges << " edges in " << duration << " ms." << std::endl;
        
        std::cout << "Partitioning graph into " << num_parts << " parts using METIS..." << std::endl;
        partitions = partition_graph(graph, num_parts);
    }
    
    int num_vertices = 0;
    if (rank == 0) {
        num_vertices = graph.get_num_vertices();
    }
    MPI_Bcast(&num_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        graph = Graph(num_vertices);
    }
    
    if (rank == 0) {
        partitions.resize(num_vertices);
    } else {
        partitions.resize(num_vertices);
    }
    MPI_Bcast(partitions.data(), num_vertices, MPI_INT, 0, MPI_COMM_WORLD);
    
    std::vector<int> edge_data;
    if (rank == 0) {
        for (int v = 0; v < num_vertices; v++) {
            const auto& neighbors = graph.get_neighbors(v);
            edge_data.push_back(neighbors.size());
            for (const auto& edge : neighbors) {
                edge_data.push_back(edge.first);
                float weight_as_float = static_cast<float>(edge.second);
                edge_data.push_back(*reinterpret_cast<int*>(&weight_as_float));
            }
        }
    }
    
    int edge_data_size = 0;
    if (rank == 0) {
        edge_data_size = edge_data.size();
    }
    MPI_Bcast(&edge_data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        edge_data.resize(edge_data_size);
    }
    MPI_Bcast(edge_data.data(), edge_data_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        int idx = 0;
        for (int v = 0; v < num_vertices; v++) {
            int num_neighbors = edge_data[idx++];
            for (int i = 0; i < num_neighbors; i++) {
                int target = edge_data[idx++];
                float weight = *reinterpret_cast<float*>(&edge_data[idx++]);
                graph.add_edge(v, target, weight);
            }
        }
    }
    
    graph.distribute_graph(partitions, rank, size);
    
    if (rank == 0) {
        std::cout << "Calculating initial SSSP from source " << source << " using Dijkstra..." << std::endl;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    SSSPResult initial_sssp = calculate_initial_sssp_parallel(graph, source, rank, size);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto init_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    
    if (rank == 0) {
        std::cout << "Initial SSSP calculated in " << init_duration / 1000.0 << " ms." << std::endl;
        
        std::ifstream changes_file(updates_file);
        int total_changes = 0;
        std::string line;
        while (std::getline(changes_file, line)) {
            if (!line.empty() && line[0] != '#') {
                total_changes++;
            }
        }
        changes_file.close();
        
        if (num_changes == -1 || num_changes > total_changes) {
            num_changes = total_changes;
        }
        
        std::cout << "Loading " << num_changes << " out of " << total_changes << " edge changes from " << updates_file << "..." << std::endl;
    }
    
    std::vector<EdgeChange> changes = graph.broadcast_changes(updates_file, 0);
    
    if (num_changes > 0 && static_cast<size_t>(num_changes) < changes.size()) {
        changes.resize(num_changes);
    }
    
    for (const EdgeChange& change : changes) {
        if (change.operation == '+') {
            graph.add_edge(change.source, change.target, change.weight);
        } else if (change.operation == '-') {
            graph.remove_edge(change.source, change.target);
        }
    }
    
    if (rank == 0) {
        std::cout << "Running mode: OpenMP (" << num_threads << " threads) + MPI (" << size << " processes)" << std::endl;
        std::cout << "Updating SSSP..." << std::endl;
    }
    
    start_time = std::chrono::high_resolution_clock::now();
    SSSPResult updated_sssp = update_sssp_parallel(graph, initial_sssp, changes, source, rank, size, num_threads);
    end_time = std::chrono::high_resolution_clock::now();
    auto update_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    
    if (rank == 0) {
        run_sequential_sssp(graph, source, updates_file);
        std::cout << "--- Parallel Timings ---" << std::endl;
        std::cout << "Initial Dijkstra: " << init_duration / 1000.0 << " ms" << std::endl;
        std::cout << "Update (OpenMP " << num_threads << " threads + MPI " << size << " processes): " 
                  << update_duration / 1000.0 << " ms" << std::endl;
        std::cout << "Execution finished." << std::endl;
    }
    
    std::cout << "Process " << rank << " finished with exit code 0" << std::endl;
    
    MPI_Finalize();
    return 0;
}
