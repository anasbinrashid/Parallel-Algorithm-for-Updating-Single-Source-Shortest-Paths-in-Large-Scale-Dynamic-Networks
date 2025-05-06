/**
 * main_hybrid.cpp
 * Main program for Hybrid OpenMP+MPI+METIS SSSP update algorithm
 */

#include "types.h"
#include "graph.h"
#include "utils.h"
#include "sssp_hybrid.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <mpi.h>

// Helper functions for safe MPI operations
void broadcastChanges(std::vector<EdgeChange>& changes, int root);
void safeError(const char* msg) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cerr << "ERROR: " << msg << std::endl;
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
}

// Helper function to safely broadcast changes
void broadcastChanges(std::vector<EdgeChange>& changes, int root) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Broadcast the number of changes
    int num_changes = changes.size();
    MPI_Bcast(&num_changes, 1, MPI_INT, root, MPI_COMM_WORLD);
    
    // Resize vector on non-root processes
    if (rank != root) {
        changes.resize(num_changes);
    }
    
    // Pack the changes into separate arrays (to avoid MPI derived types)
    std::vector<int> sources(num_changes);
    std::vector<int> targets(num_changes);
    std::vector<double> weights(num_changes);
    std::vector<int> types(num_changes);
    
    if (rank == root) {
        for (int i = 0; i < num_changes; i++) {
            sources[i] = changes[i].source;
            targets[i] = changes[i].target;
            weights[i] = changes[i].weight;
            types[i] = changes[i].type == INSERTION ? 1 : 0;
        }
    }
    
    // Broadcast the arrays
    MPI_Bcast(sources.data(), num_changes, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(targets.data(), num_changes, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(weights.data(), num_changes, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Bcast(types.data(), num_changes, MPI_INT, root, MPI_COMM_WORLD);
    
    // Unpack on non-root processes
    if (rank != root) {
        for (int i = 0; i < num_changes; i++) {
            changes[i].source = sources[i];
            changes[i].target = targets[i];
            changes[i].weight = weights[i];
            changes[i].type = types[i] ? INSERTION : DELETION;
        }
    }
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    // Get rank and size
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Check command line arguments
    if (argc < 4) {
        if (rank == 0) {
            std::cout << "Usage: " << argv[0] << " <num_vertices> <num_changes> <num_threads> [source_vertex]" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    int num_vertices = std::atoi(argv[1]);
    int num_changes = std::atoi(argv[2]);
    int num_threads = std::atoi(argv[3]);
    VertexID source = (argc > 4) ? std::atoi(argv[4]) : 0;
    
    // Validate input
    if (num_vertices <= 0 || num_changes < 0 || num_threads <= 0 || source < 0 || source >= num_vertices) {
        if (rank == 0) {
            std::cerr << "Invalid input parameters" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        std::cout << "Running Hybrid OpenMP+MPI+METIS SSSP Update Algorithm:" << std::endl;
        std::cout << "  Vertices: " << num_vertices << std::endl;
        std::cout << "  Changes: " << num_changes << std::endl;
        std::cout << "  Threads per process: " << num_threads << std::endl;
        std::cout << "  MPI processes: " << size << std::endl;
        std::cout << "  Source: " << source << std::endl;
        std::cout << std::endl;
    }
    
    // Generate temporal graph with initial edges (roughly 3 edges per vertex)
    int num_initial_edges = num_vertices * 3;
    Graph graph = generateTemporalGraph(num_vertices, num_initial_edges, 0);
    
    // Broadcast graph to all processes
    // Note: In a real implementation, this would be handled more efficiently
    // by only sharing the necessary parts of the graph
    if (rank == 0) {
        // Save initial graph
        saveGraphToFile(graph, "hybrid_initial_graph.txt");
    }
    
    // Synchronize graph across processes
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Initialize SSSP with hybrid approach
    if (rank == 0) {
        std::cout << "Initializing SSSP tree..." << std::endl;
    }
    
    SSSPHybrid sssp(graph, source, num_threads);
    sssp.initialize();
    
    if (rank == 0) {
        // Generate visualization of initial SSSP tree
        generateSSSPVisualization(graph, sssp.getSSSPTree(), source, "hybrid_initial_sssp.dot");
        std::cout << "Initial SSSP tree saved to 'hybrid_initial_sssp.dot'" << std::endl;
    }
    
    // Generate changes
    std::vector<EdgeChange> all_changes;
    
    if (rank == 0) {
        std::cout << "Generating " << num_changes << " changes..." << std::endl;
    }
    
    // Generate changes in smaller batches to simulate temporal behavior
    int num_batches = 5;
    int changes_per_batch = num_changes / num_batches;
    
    if (rank == 0) {
        std::cout << "Processing changes in " << num_batches << " batches:" << std::endl;
    }
    
    for (int batch = 0; batch < num_batches; ++batch) {
        // Create a temporary graph for generating changes
        Graph temp_graph = graph;
        
        // Generate changes for this batch
        std::vector<EdgeChange> batch_changes;
        double insertion_ratio = 0.5;  // 50% insertions, 50% deletions
        
        // Different ratios for each batch to simulate temporal patterns
        switch (batch) {
            case 0: insertion_ratio = 0.8; break;  // Mostly insertions
            case 1: insertion_ratio = 0.6; break;
            case 2: insertion_ratio = 0.5; break;  // Equal insertions and deletions
            case 3: insertion_ratio = 0.4; break;
            case 4: insertion_ratio = 0.2; break;  // Mostly deletions
        }
        
        // Only master process generates changes
        if (rank == 0) {
            // Generate a new graph with the desired changes
            Graph next_graph = generateTemporalGraph(num_vertices, 0, changes_per_batch, insertion_ratio);
            
            // Extract changes by comparing the two graphs
            for (VertexID u = 0; u < num_vertices; ++u) {
                // Find deletions
                for (const auto& edge : graph.getNeighbors(u)) {
                    VertexID v = edge.first;
                    Weight w = edge.second;
                    
                    if (!next_graph.hasEdge(u, v)) {
                        batch_changes.push_back(EdgeChange(u, v, w, DELETION));
                    }
                }
                
                // Find insertions
                for (const auto& edge : next_graph.getNeighbors(u)) {
                    VertexID v = edge.first;
                    Weight w = edge.second;
                    
                    if (!graph.hasEdge(u, v)) {
                        batch_changes.push_back(EdgeChange(u, v, w, INSERTION));
                    }
                }
            }
            
            // Trim to exact batch size if necessary
            if (batch_changes.size() > changes_per_batch) {
                batch_changes.resize(changes_per_batch);
            }
            
            // Save changes to file
            std::string batch_filename = "hybrid_changes_batch_" + std::to_string(batch) + ".txt";
            saveChangesToFile(batch_changes, batch_filename);
            
            // Update the graph for the next batch
            graph = next_graph;
        }
        
        // Use our safer broadcast function instead of MPI_Type_create_struct
        broadcastChanges(batch_changes, 0);
        
        // Update graph on all processes
        graph.applyChanges(batch_changes);
        
        if (rank == 0) {
            std::cout << "Batch " << batch + 1 << ":" << std::endl;
            std::cout << "  Insertions: " << std::count_if(batch_changes.begin(), batch_changes.end(), 
                          [](const EdgeChange& e) { return e.type == INSERTION; }) << std::endl;
            std::cout << "  Deletions: " << std::count_if(batch_changes.begin(), batch_changes.end(), 
                          [](const EdgeChange& e) { return e.type == DELETION; }) << std::endl;
        }
        
        // Measure performance
        Timer timer;
        timer.start();
        Metrics metrics = sssp.update(batch_changes);
        timer.stop();
        
        // Add to all changes
        all_changes.insert(all_changes.end(), batch_changes.begin(), batch_changes.end());
        
        // Print performance metrics
        if (rank == 0) {
            std::cout << "  Update time: " << metrics.total_time << " seconds" << std::endl;
            std::cout << "    Step 1 (identifying affected subgraph): " << metrics.step1_time << " seconds" << std::endl;
            std::cout << "    Step 2 (updating affected subgraph): " << metrics.step2_time << " seconds" << std::endl;
            std::cout << "  Affected vertices: " << metrics.affected_vertices << std::endl;
            std::cout << std::endl;
            
            // Generate visualization for this batch
            std::string dot_filename = "hybrid_sssp_after_batch_" + std::to_string(batch) + ".dot";
            generateSSSPVisualization(graph, sssp.getSSSPTree(), source, dot_filename);
            
            // Verify correctness
            if (verifySSSP(graph, source, sssp.getSSSPTree())) {
                std::cout << "  SSSP verification: PASSED" << std::endl;
            } else {
                std::cout << "  SSSP verification: FAILED" << std::endl;
            }
            std::cout << std::endl;
        }
        
        // Synchronize all processes
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
        // Save final graph
        saveGraphToFile(graph, "hybrid_final_graph.txt");
        
        // Save all changes
        saveChangesToFile(all_changes, "hybrid_all_changes.txt");
        
        // Print final SSSP tree
        std::cout << "Final SSSP tree:" << std::endl;
        sssp.printTree();
        
        std::cout << "Final SSSP tree saved to 'hybrid_final_sssp.dot'" << std::endl;
        generateSSSPVisualization(graph, sssp.getSSSPTree(), source, "hybrid_final_sssp.dot");
    }
    
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}
