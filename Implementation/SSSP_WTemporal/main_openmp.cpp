/**
 * main_openmp.cpp
 * Main program for OpenMP SSSP update algorithm
 */

#include "types.h"
#include "graph.h"
#include "utils.h"
#include "sssp_openmp.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>

int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <num_vertices> <num_changes> <num_threads> [source_vertex] [graph_file] [changes_file]" << std::endl;
        return 1;
    }
    
    int num_vertices = std::atoi(argv[1]);
    int num_changes = std::atoi(argv[2]);
    int num_threads = std::atoi(argv[3]);
    VertexID source = (argc > 4) ? std::atoi(argv[4]) : 0;
    std::string graph_file = (argc > 5) ? argv[5] : "example_graph.txt";
    std::string changes_file = (argc > 6) ? argv[6] : "example_changes.txt";
    
    // Validate input
    if (num_vertices <= 0 || num_changes < 0 || num_threads <= 0 || source < 0 || source >= num_vertices) {
        std::cerr << "Invalid input parameters" << std::endl;
        return 1;
    }
    
    std::cout << "Running OpenMP SSSP Update Algorithm:" << std::endl;
    std::cout << "  Vertices: " << num_vertices << std::endl;
    std::cout << "  Changes: " << num_changes << std::endl;
    std::cout << "  Threads: " << num_threads << std::endl;
    std::cout << "  Source: " << source << std::endl;
    std::cout << "  Graph file: " << graph_file << std::endl;
    std::cout << "  Changes file: " << changes_file << std::endl;
    std::cout << std::endl;
    
    // Initialize the original graph
    Graph graph(num_vertices);
    
    // Try to load graph from file
    bool graph_loaded = false;
    try {
        std::ifstream file(graph_file);
        if (file.is_open()) {
            graph = Graph(graph_file);
            std::cout << "Loaded graph from " << graph_file << std::endl;
            graph_loaded = true;
        }
    } catch (std::exception& e) {
        // Failed to load, will generate instead
        graph_loaded = false;
    }
    
    // Generate graph if not loaded
    if (!graph_loaded) {
        int num_initial_edges = num_vertices * 3;
        graph = generateTemporalGraph(num_vertices, num_initial_edges, 0);
        std::cout << "Generated random graph" << std::endl;
        
        // Save initial graph
        saveGraphToFile(graph, "initial_graph.txt");
    }
    
    // Initialize SSSP with OpenMP
    std::cout << "Initializing SSSP tree..." << std::endl;
    SSSPOpenMP sssp(graph, source, num_threads);
    sssp.initialize();
    
    // Generate visualization of initial SSSP tree
    generateSSSPVisualization(graph, sssp.getSSSPTree(), source, "initial_sssp.dot");
    std::cout << "Initial SSSP tree saved to 'initial_sssp.dot'" << std::endl;
    
    // Load or generate changes
    std::vector<EdgeChange> all_changes;
    bool changes_loaded = false;
    
    try {
        std::ifstream file(changes_file);
        if (file.is_open()) {
            all_changes = loadChangesFromFile(changes_file);
            std::cout << "Loaded " << all_changes.size() << " changes from " << changes_file << std::endl;
            changes_loaded = true;
        }
    } catch (std::exception& e) {
        changes_loaded = false;
    }
    
    // Generate changes if not loaded
    if (!changes_loaded) {
        std::cout << "Generating " << num_changes << " changes..." << std::endl;
        
        // Generate changes randomly
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        for (int i = 0; i < num_changes; ++i) {
            VertexID u = std::rand() % num_vertices;
            VertexID v = std::rand() % num_vertices;
            while (v == u) v = std::rand() % num_vertices; // Avoid self-loops
            
            Weight weight = (std::rand() % 100) / 10.0; // Random weight between 0 and 10
            
            ChangeType type;
            if (i < num_changes / 2) {
                type = INSERTION;
                if (graph.hasEdge(u, v)) continue; // Skip if edge already exists
            } else {
                type = DELETION;
                if (!graph.hasEdge(u, v)) continue; // Skip if edge doesn't exist
                weight = graph.getEdgeWeight(u, v); // Use existing weight
            }
            
            all_changes.push_back(EdgeChange(u, v, weight, type));
        }
        
        // Save generated changes
        saveChangesToFile(all_changes, "generated_changes.txt");
    }
    
    // Process changes in batches
    int num_batches = 5;
    int changes_per_batch = all_changes.size() / num_batches;
    if (changes_per_batch == 0) {
        changes_per_batch = 1;
        num_batches = all_changes.size();
    }
    
    std::cout << "Processing changes in " << num_batches << " batches:" << std::endl;
    
    for (int batch = 0; batch < num_batches; ++batch) {
        // Get changes for this batch
        int start_idx = batch * changes_per_batch;
        int end_idx = (batch == num_batches - 1) ? all_changes.size() : (batch + 1) * changes_per_batch;
        
        std::vector<EdgeChange> batch_changes(all_changes.begin() + start_idx, all_changes.begin() + end_idx);
        
        // Save changes to file
        std::string batch_filename = "changes_batch_" + std::to_string(batch) + ".txt";
        saveChangesToFile(batch_changes, batch_filename);
        
        // Count insertions and deletions
        int insertions = 0, deletions = 0;
        for (const auto& change : batch_changes) {
            if (change.type == INSERTION) insertions++;
            else deletions++;
        }
        
        // Update SSSP tree
        std::cout << "Batch " << batch + 1 << ":" << std::endl;
        std::cout << "  Insertions: " << insertions << std::endl;
        std::cout << "  Deletions: " << deletions << std::endl;
        
        // Measure performance and update the SSSP tree
        Timer timer;
        timer.start();
        
        // Create a copy of the graph for verification
        Graph graph_before_changes = graph;
        
        // Apply changes to the graph
        for (const auto& change : batch_changes) {
            if (change.type == INSERTION) {
                graph.addEdge(change.source, change.target, change.weight);
            } else {
                graph.removeEdge(change.source, change.target);
            }
        }
        
        // Update the SSSP tree
        Metrics metrics = sssp.update(batch_changes);
        timer.stop();
        
        // Print performance metrics
        std::cout << "  Update time: " << metrics.total_time << " seconds" << std::endl;
        std::cout << "    Step 1 (identifying affected subgraph): " << metrics.step1_time << " seconds" << std::endl;
        std::cout << "    Step 2 (updating affected subgraph): " << metrics.step2_time << " seconds" << std::endl;
        std::cout << "  Affected vertices: " << metrics.affected_vertices << std::endl;
        std::cout << std::endl;
        
        // Generate visualization for this batch
        std::string dot_filename = "sssp_after_batch_" + std::to_string(batch) + ".dot";
        generateSSSPVisualization(graph, sssp.getSSSPTree(), source, dot_filename);
        
        // Compute correct SSSP for verification 
        std::vector<SSSPNode> correct_sssp = graph.computeInitialSSSP(source);
        
        // Check if our result matches Dijkstra's algorithm
        bool verification_passed = true;
        for (VertexID v = 0; v < graph.getNumVertices(); ++v) {
            if (sssp.getSSSPTree()[v].distance != correct_sssp[v].distance) {
                verification_passed = false;
                break;
            }
        }
        
        // Print verification result
        if (verification_passed) {
            std::cout << "  SSSP verification: PASSED" << std::endl;
        } else {
            std::cout << "  SSSP verification: FAILED" << std::endl;
            std::cout << "  Debug info - vertex mismatches:" << std::endl;
            
            // Print mismatched vertices
            for (VertexID v = 0; v < graph.getNumVertices(); ++v) {
                if (sssp.getSSSPTree()[v].distance != correct_sssp[v].distance) {
                    std::cout << "    Vertex " << v << ": Got " << sssp.getSSSPTree()[v].distance
                              << ", expected " << correct_sssp[v].distance << std::endl;
                }
            }
            
            // Reset to correct SSSP tree
            sssp.resetToCorrectTree();
        }
        
        std::cout << std::endl;
    }
    
    // Save final graph
    saveGraphToFile(graph, "final_graph.txt");
    
    // Print final SSSP tree
    std::cout << "Final SSSP tree:" << std::endl;
    sssp.printTree();
    
    std::cout << "Final SSSP tree saved to 'final_sssp.dot'" << std::endl;
    generateSSSPVisualization(graph, sssp.getSSSPTree(), source, "final_sssp.dot");
    
    return 0;
}
