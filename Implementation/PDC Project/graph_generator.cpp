#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

// Generate a larger synthetic graph with specified properties
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <num_vertices> <num_edges> <output_file> [edge_prob] [seed]" << std::endl;
        std::cerr << "  num_vertices : Number of vertices in the graph" << std::endl;
        std::cerr << "  num_edges    : Approximate number of edges in the graph" << std::endl;
        std::cerr << "  output_file  : Output filename" << std::endl;
        std::cerr << "  edge_prob    : (Optional) Edge probability distribution" << std::endl;
        std::cerr << "                 0 = Uniform random, 1 = Power-law/scale-free (default: 0)" << std::endl;
        std::cerr << "  seed         : (Optional) Random seed value (default: time-based)" << std::endl;
        return 1;
    }
    
    int numVertices = std::stoi(argv[1]);
    int numEdges = std::stoi(argv[2]);
    std::string outputFile = argv[3];
    int edgeDistribution = (argc > 4) ? std::stoi(argv[4]) : 0;
    unsigned int seed = (argc > 5) ? std::stoul(argv[5]) : std::random_device()();
    
    if (numVertices <= 0 || numEdges <= 0) {
        std::cerr << "Error: Number of vertices and edges must be positive" << std::endl;
        return 1;
    }
    
    // Initialize random generators
    std::mt19937 gen(seed);
    
    // Track generated edges to avoid duplicates
    std::unordered_set<uint64_t> generatedEdges;
    
    // Open output file
    std::ofstream outFile(outputFile);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << outputFile << " for writing" << std::endl;
        return 1;
    }
    
    // Hash function to create a unique identifier for an edge
    auto edgeHash = [numVertices](int u, int v) -> uint64_t {
        return (static_cast<uint64_t>(u) << 32) | static_cast<uint64_t>(v);
    };
    
    // Edge generation approaches
    switch (edgeDistribution) {
        case 1: {
            // Power-law/scale-free network (preferential attachment)
            std::cout << "Generating power-law/scale-free network..." << std::endl;
            
            // Start with a small connected network
            std::vector<int> degrees(numVertices, 0);
            
            // Create a small initial connected graph
            const int initialSize = std::min(5, numVertices);
            for (int i = 0; i < initialSize - 1; i++) {
                for (int j = i + 1; j < initialSize; j++) {
                    float weight = std::uniform_real_distribution<float>(0.1f, 10.0f)(gen);
                    outFile << i << " " << j << " " << weight << std::endl;
                    generatedEdges.insert(edgeHash(i, j));
                    degrees[i]++;
                    degrees[j]++;
                }
            }
            
            // Add remaining edges using preferential attachment
            int edgesAdded = (initialSize * (initialSize - 1)) / 2;
            
            while (edgesAdded < numEdges && edgesAdded < numVertices * (numVertices - 1) / 2) {
                // Select vertex with probability proportional to its degree
                int u, v;
                
                // For new vertex (high probability to attach to well-connected nodes)
                std::discrete_distribution<int> degreeDistribution(degrees.begin(), degrees.end());
                u = degreeDistribution(gen);
                
                // Select second vertex uniformly
                do {
                    v = std::uniform_int_distribution<int>(0, numVertices - 1)(gen);
                } while (u == v || generatedEdges.find(edgeHash(u, v)) != generatedEdges.end());
                
                // Add edge
                float weight = std::uniform_real_distribution<float>(0.1f, 10.0f)(gen);
                outFile << u << " " << v << " " << weight << std::endl;
                generatedEdges.insert(edgeHash(u, v));
                degrees[u]++;
                degrees[v]++;
                edgesAdded++;
                
                // Progress indicator
                if (edgesAdded % 10000 == 0) {
                    std::cout << "Generated " << edgesAdded << " edges..." << std::endl;
                }
            }
            break;
        }
        
        default: {
            // Uniform random graph
            std::cout << "Generating uniform random graph..." << std::endl;
            std::uniform_int_distribution<int> vertexDist(0, numVertices - 1);
            std::uniform_real_distribution<float> weightDist(0.1f, 10.0f);
            
            // Add edges
            int edgesAdded = 0;
            int attempts = 0;
            const int MAX_ATTEMPTS = numEdges * 10;
            
            while (edgesAdded < numEdges && attempts < MAX_ATTEMPTS) {
                int u = vertexDist(gen);
                int v = vertexDist(gen);
                
                // Skip self-loops and duplicate edges
                if (u == v || generatedEdges.find(edgeHash(u, v)) != generatedEdges.end()) {
                    attempts++;
                    continue;
                }
                
                float weight = weightDist(gen);
                outFile << u << " " << v << " " << weight << std::endl;
                generatedEdges.insert(edgeHash(u, v));
                edgesAdded++;
                
                // Progress indicator
                if (edgesAdded % 10000 == 0) {
                    std::cout << "Generated " << edgesAdded << " edges..." << std::endl;
                }
            }
            
            if (attempts >= MAX_ATTEMPTS) {
                std::cout << "Warning: Reached maximum attempts. Generated only " 
                         << edgesAdded << " out of " << numEdges << " requested edges." << std::endl;
            }
            break;
        }
    }
    
    outFile.close();
    std::cout << "Generated graph with " << numVertices << " vertices and " 
              << generatedEdges.size() << " edges." << std::endl;
    std::cout << "Saved to: " << outputFile << std::endl;
    
    return 0;
}
