/**
 * main_temporal_hybrid.cpp
 * Main program for hybrid OpenMP+MPI+METIS temporal SSSP update algorithm
 */

#include "types.h"
#include "temporal_graph.h"
#include "utils_temporal.h"
#include "sssp_temporal_hybrid.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <mpi.h>
#include <random>    // For std::random_device, std::mt19937, std::uniform_real_distribution
#include <cmath>     // For std::round

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
        std::cout << "Running Hybrid Temporal SSSP Update Algorithm:" << std::endl;
        std::cout << "  Vertices: " << num_vertices << std::endl;
        std::cout << "  Changes: " << num_changes << std::endl;
        std::cout << "  Threads per process: " << num_threads << std::endl;
        std::cout << "  MPI processes: " << size << std::endl;
        std::cout << "  Source: " << source << std::endl;
        std::cout << std::endl;
    }
    
    // Generate initial graph (time 0)
    int num_initial_edges = num_vertices * 3;
    TemporalGraph graph(num_vertices);
    
    // Add initial edges at time 0 (only on master process)
    if (rank == 0) {
        // Create random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> vertex_dist(0, num_vertices - 1);
        std::uniform_real_distribution<> weight_dist(0.1, 10.0);
        
        // Set to keep track of added edges
        std::set<std::pair<VertexID, VertexID>> added_edges;
        
        // Add initial edges at time 0
        int edges_added = 0;
        while (edges_added < num_initial_edges) {
            VertexID u = vertex_dist(gen);
            VertexID v = vertex_dist(gen);
            
            // Avoid self-loops and duplicates
            if (u != v && !added_edges.count({u, v})) {
                Weight weight = std::round(weight_dist(gen) * 10) / 10;  // Round to 1 decimal place
                graph.addTimedEdge(u, v, weight, 0.0);
                added_edges.insert({u, v});
                edges_added++;
            }
        }
        
        // Save initial graph
        saveTemporalGraphToFile(graph, "hybrid_initial_temporal_graph.txt");
    }
    
    // Broadcast graph structure to all processes
    // This is a simplified version - in a real implementation, 
    // you would use a more efficient approach to distribute the graph
    
    // For now, we'll just have all processes use the same graph
    // Synchronize graph across processes
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Generate changes with timestamps (only on master process)
    std::vector<EdgeChange> all_changes;
    std::vector<Timestamp> timepoints;
    
    // Generate 5 timepoints (1.0, 2.0, 3.0, 4.0, 5.0)
    for (int i = 1; i <= 5; i++) {
        timepoints.push_back(static_cast<double>(i));
    }
    
    if (rank == 0) {
        // Generate changes for each timepoint
        int changes_per_timepoint = num_changes / timepoints.size();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> vertex_dist(0, num_vertices - 1);
        std::uniform_real_distribution<> weight_dist(0.1, 10.0);
        
        // Set to keep track of added edges
        std::set<std::pair<VertexID, VertexID>> added_edges;
        
        // Add all initial edges to the set
        for (VertexID v = 0; v < num_vertices; ++v) {
            for (const auto& edge : graph.getNeighbors(v)) {
                added_edges.insert({v, edge.first});
            }
        }
        
        for (size_t t = 0; t < timepoints.size(); t++) {
            Timestamp time = timepoints[t];
            
            // Different ratios for each time point to simulate temporal patterns
            double insertion_ratio;
            switch (t) {
                case 0: insertion_ratio = 0.8; break;  // Mostly insertions
                case 1: insertion_ratio = 0.6; break;
                case 2: insertion_ratio = 0.5; break;  // Equal insertions and deletions
                case 3: insertion_ratio = 0.4; break;
                case 4: insertion_ratio = 0.2; break;  // Mostly deletions
                default: insertion_ratio = 0.5;
            }
            
            std::cout << "Generating changes for time " << time << " (insertion ratio: " 
                      << insertion_ratio << ")" << std::endl;
            
            // Generate changes for this timepoint
            for (int i = 0; i < changes_per_timepoint; i++) {
                ChangeType type;
                
                // Determine change type based on insertion ratio
                if (std::uniform_real_distribution<>(0.0, 1.0)(gen) < insertion_ratio) {
                    type = INSERTION;
                    
                    // For insertion, find an edge that doesn't exist
                    VertexID u, v;
                    bool found = false;
                    
                    while (!found) {
                        u = vertex_dist(gen);
                        v = vertex_dist(gen);
                        
                        if (u != v && !added_edges.count({u, v})) {
                            Weight weight = std::round(weight_dist(gen) * 10) / 10;
                            EdgeChange change(u, v, weight, type, time);
                            all_changes.push_back(change);
                            added_edges.insert({u, v});
                            found = true;
                        }
                    }
                } else {
                    type = DELETION;
                    
                    // For deletion, find an existing edge
                    if (added_edges.empty()) {
                        // No edges to delete, switch to insertion
                        type = INSERTION;
                        VertexID u = vertex_dist(gen);
                        VertexID v = vertex_dist(gen);
                        
                        if (u != v && !added_edges.count({u, v})) {
                            Weight weight = std::round(weight_dist(gen) * 10) / 10;
                            EdgeChange change(u, v, weight, type, time);
                            all_changes.push_back(change);
                            added_edges.insert({u, v});
                        }
                    } else {
                        // Select a random existing edge
                        int edge_idx = std::uniform_int_distribution<>(0, added_edges.size() - 1)(gen);
                        auto it = added_edges.begin();
                        std::advance(it, edge_idx);
                        
                        VertexID u = it->first;
                        VertexID v = it->second;
                        
                        // Delete edge
                        EdgeChange change(u, v, 0.0, type, time);
                        all_changes.push_back(change);
                        added_edges.erase(it);
                    }
                }
            }
        }
        
        // Save changes
        saveEdgeChangesToFile(all_changes, "hybrid_all_temporal_changes.txt");
    }
    
    // Broadcast the number of changes
    int num_total_changes = all_changes.size();
    MPI_Bcast(&num_total_changes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Resize changes array on non-root processes
    if (rank != 0) {
        all_changes.resize(num_total_changes);
    }
    
    // Create MPI datatype for EdgeChange
    MPI_Datatype mpi_edge_change;
    int blocklengths[5] = {1, 1, 1, 1, 1};
    MPI_Aint displacements[5];
    MPI_Datatype types[5] = {MPI_INT, MPI_INT, MPI_DOUBLE, MPI_INT, MPI_DOUBLE};  // INT for enum
    
    // Get displacements
    EdgeChange temp(0, 0, 0.0, INSERTION, 0.0);
    MPI_Aint base_address;
    MPI_Get_address(&temp, &base_address);
    MPI_Get_address(&temp.source, &displacements[0]);
    MPI_Get_address(&temp.target, &displacements[1]);
    MPI_Get_address(&temp.weight, &displacements[2]);
    MPI_Get_address(&temp.type, &displacements[3]);
    MPI_Get_address(&temp.time, &displacements[4]);
    
    // Make relative to base_address
    for (int i = 0; i < 5; i++) {
        displacements[i] = MPI_Aint_diff(displacements[i], base_address);
    }
    
    // Create and commit MPI datatype
    MPI_Type_create_struct(5, blocklengths, displacements, types, &mpi_edge_change);
    MPI_Type_commit(&mpi_edge_change);
    
    // Broadcast changes to all processes
    MPI_Bcast(all_changes.data(), num_total_changes, mpi_edge_change, 0, MPI_COMM_WORLD);
    
    // Broadcast timepoints
    int num_timepoints = timepoints.size();
    MPI_Bcast(&num_timepoints, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        timepoints.resize(num_timepoints);
    }
    
    MPI_Bcast(timepoints.data(), num_timepoints, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Free MPI datatype
    MPI_Type_free(&mpi_edge_change);
    
    // Initialize SSSP with hybrid temporal approach
    if (rank == 0) {
        std::cout << "Initializing SSSP tree..." << std::endl;
    }
    
    SSSPTemporalHybrid sssp(graph, source, num_threads);
    sssp.initialize();
    
    if (rank == 0) {
        // Generate visualization of initial SSSP tree
        generateTemporalSSSPVisualization(graph, sssp.getSSSPTreeAtTime(0.0), 
                                         source, 0.0, "hybrid_temporal_initial_sssp.dot");
        std::cout << "Initial SSSP tree saved to 'hybrid_temporal_initial_sssp.dot'" << std::endl;
    }
    
    // Update SSSP tree for each timepoint
    if (rank == 0) {
        std::cout << "\nProcessing changes for each timepoint:" << std::endl;
    }
    
    for (Timestamp time : timepoints) {
        if (rank == 0) {
            std::cout << "\nTimepoint " << time << ":" << std::endl;
            
            // Count changes at this timepoint
            int insertions = 0, deletions = 0;
            for (const auto& change : all_changes) {
                if (change.time == time) {
                    if (change.type == INSERTION) insertions++;
                    else deletions++;
                }
            }
            
            std::cout << "  Insertions: " << insertions << std::endl;
            std::cout << "  Deletions: " << deletions << std::endl;
        }
        
        // Update graph to this timepoint
        graph.setCurrentTime(time);
        
        // Apply changes for this timepoint
        std::vector<EdgeChange> changes_at_time;
        for (const auto& change : all_changes) {
            if (change.time == time) {
                changes_at_time.push_back(change);
            }
        }
        
        // Update SSSP tree
        Timer timer;
        timer.start();
        Metrics metrics = sssp.updateAtTime(time);
        timer.stop();
        
        // Print performance metrics
        if (rank == 0) {
            std::cout << "  Update time: " << metrics.total_time << " seconds" << std::endl;
            std::cout << "    Step 1 (identifying affected subgraph): " << metrics.step1_time << " seconds" << std::endl;
            std::cout << "    Step 2 (updating affected subgraph): " << metrics.step2_time << " seconds" << std::endl;
            std::cout << "  Affected vertices: " << metrics.affected_vertices << std::endl;
            
            // Generate visualization for this timepoint
            std::string dot_filename = "hybrid_temporal_sssp_at_time_" + std::to_string(time) + ".dot";
            generateTemporalSSSPVisualization(graph, sssp.getSSSPTreeAtTime(time), 
                                             source, time, dot_filename);
            
            // Verify correctness (only on master process)
            if (verifyTemporalSSSP(graph, source, sssp.getSSSPTreeAtTime(time))) {
                std::cout << "  SSSP verification: PASSED" << std::endl;
            } else {
                std::cout << "  SSSP verification: FAILED" << std::endl;
            }
        }
        
        // Synchronize all processes
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // Print summary of all SSSP trees
    if (rank == 0) {
        std::cout << "\nSummary of SSSP trees at different timepoints:" << std::endl;
        std::cout << std::setw(10) << "Time" << std::setw(15) << "Reachable Nodes" 
                  << std::setw(15) << "Avg Distance" << std::setw(15) << "Max Distance" << std::endl;
        
        for (const auto& entry : sssp.getAllSSSPTrees()) {
            Timestamp time = entry.first;
            const auto& tree = entry.second;
            
            int reachable = 0;
            double sum_dist = 0.0;
            double max_dist = 0.0;
            
            for (const auto& node : tree) {
                if (node.distance != INF) {
                    reachable++;
                    sum_dist += node.distance;
                    max_dist = std::max(max_dist, node.distance);
                }
            }
            
            double avg_dist = (reachable > 0) ? sum_dist / reachable : 0.0;
            
            std::cout << std::setw(10) << time 
                      << std::setw(15) << reachable 
                      << std::setw(15) << std::fixed << std::setprecision(2) << avg_dist
                      << std::setw(15) << std::fixed << std::setprecision(2) << max_dist
                      << std::endl;
        }
        
        // Save final graph
        saveTemporalGraphToFile(graph, "hybrid_final_temporal_graph.txt");
        
        // Generate visualization of final SSSP tree
        generateTemporalSSSPVisualization(graph, sssp.getSSSPTreeAtTime(timepoints.back()), 
                                         source, timepoints.back(), "hybrid_temporal_final_sssp.dot");
        std::cout << "\nFinal SSSP tree saved to 'hybrid_temporal_final_sssp.dot'" << std::endl;
        
        // Generate temporal evolution animation script
        generateTemporalEvolutionAnimation(graph, sssp.getAllSSSPTrees(), 
                                         source, "generate_hybrid_temporal_animation.sh");
    }
    
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}
