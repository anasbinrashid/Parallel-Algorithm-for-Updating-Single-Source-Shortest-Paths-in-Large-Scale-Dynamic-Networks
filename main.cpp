#include "graph.h"
#include "sequential_sssp.h"
#include "parallel_sssp.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <mpi.h>
#include <omp.h>

// Generate random edge changes
std::vector<EdgeChange> generateRandomChanges(const Graph& g, int numChanges, float insertRatio = 0.5f) {
    std::vector<EdgeChange> changes;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::vector<int> vertices = g.getAllVertices();
    if (vertices.empty()) {
        std::cerr << "Error: Graph has no vertices" << std::endl;
        return changes;
    }
    
    // Create a vector of existing edges for potential deletion
    std::vector<std::pair<int, int>> existingEdges;
    for (int u : vertices) {
        for (const auto& edge : g.getNeighbors(u)) {
            int v = g.reverseMapVertex(edge.first);
            existingEdges.push_back(std::make_pair(u, v));
        }
    }
    
    if (existingEdges.empty() && insertRatio < 1.0f) {
        std::cerr << "Warning: No existing edges for deletion, forcing insertRatio to 1.0" << std::endl;
        insertRatio = 1.0f;
    }
    
    // Prepare distributions
    std::uniform_int_distribution<> vertexDist(0, vertices.size() - 1);
    std::uniform_int_distribution<> edgeDist(0, existingEdges.size() - 1);
    std::uniform_real_distribution<> weightDist(0.1f, 10.0f);
    std::uniform_real_distribution<> typeDist(0.0f, 1.0f);
    
    // Generate changes
    changes.reserve(numChanges);
    for (int i = 0; i < numChanges; i++) {
        bool isInsert = (typeDist(gen) < insertRatio || existingEdges.empty());
        
        if (isInsert) {
            // Insert a new edge
            int u = vertices[vertexDist(gen)];
            int v = vertices[vertexDist(gen)];
            
            // Avoid self-loops and existing edges
            int attempts = 0;
            while ((u == v || g.hasEdge(u, v)) && attempts < 100) {
                v = vertices[vertexDist(gen)];
                attempts++;
            }
            
            // If we couldn't find a non-existing edge after many attempts, skip
            if (attempts >= 100) {
                continue;
            }
            
            float weight = weightDist(gen);
            changes.emplace_back(u, v, weight, true);
        } else {
            // Delete an existing edge
            auto& edge = existingEdges[edgeDist(gen)];
            changes.emplace_back(edge.first, edge.second, 0.0f, false);
        }
    }
    
    return changes;
}

// Generate a synthetic graph with specified vertices and edges
bool generateSyntheticGraph(const std::string& filename, int numVertices, int numEdges) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> vertexDist(0, numVertices - 1);
    std::uniform_real_distribution<> weightDist(0.1f, 10.0f);
    
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return false;
    }
    
    for (int i = 0; i < numEdges; i++) {
        int u = vertexDist(gen);
        int v = vertexDist(gen);
        
        // Avoid self-loops
        while (u == v) {
            v = vertexDist(gen);
        }
        
        float weight = weightDist(gen);
        outFile << u << " " << v << " " << weight << std::endl;
    }
    
    outFile.close();
    std::cout << "Generated graph with " << numVertices << " vertices and " << numEdges << " edges" << std::endl;
    return true;
}

void printUsage(char* programName) {
    std::cout << "Usage: " << programName << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -f <filename>      : Input graph file (edge list format)" << std::endl;
    std::cout << "  -g <v> <e>         : Generate synthetic graph with <v> vertices and <e> edges" << std::endl;
    std::cout << "  -s <vertex>        : Source vertex (default: 0)" << std::endl;
    std::cout << "  -c <count>         : Number of edge changes (default: 1000)" << std::endl;
    std::cout << "  -t <threads>       : Number of OpenMP threads per process (default: 2)" << std::endl;
    std::cout << "  -i <ratio>         : Ratio of insertions to deletions (0-1, default: 0.7)" << std::endl;
    std::cout << "  -a <level>         : Asynchrony level (default: 2)" << std::endl;
    std::cout << "  -m <iterations>    : Maximum iterations (default: 200)" << std::endl;
    std::cout << "  -o <filename>      : Output file for performance results" << std::endl;
    std::cout << "  -v                 : Verbose output" << std::endl;
    std::cout << "  -h                 : Show this help message" << std::endl;
}

int main(int argc, char *argv[]) {
    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    if (provided < MPI_THREAD_MULTIPLE) {
        std::cerr << "Warning: MPI implementation does not fully support MPI_THREAD_MULTIPLE" << std::endl;
    }
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Parse command line arguments
    std::string filename = "bio-CE-HT.edges";  // Default input file
    std::string outputFile = "";               // Default: no output file
    int sourceVertex = 0;                      // Default source vertex
    int numChanges = 1000;                     // Default number of changes
    int numThreads = 2;                        // Default number of threads
    float insertRatio = 0.7f;                  // Default insertion ratio (70% insertions)
    int asyncLevel = 2;                        // Default asynchrony level
    int maxIterations = 200;                   // Default maximum iterations
    bool verbose = false;                      // Default: no verbose output
    bool generateGraph = false;                // Default: don't generate graph
    int genVertices = 0, genEdges = 0;         // Parameters for graph generation
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-f" && i + 1 < argc) {
            filename = argv[++i];
        } else if (arg == "-g" && i + 2 < argc) {
            generateGraph = true;
            genVertices = std::stoi(argv[++i]);
            genEdges = std::stoi(argv[++i]);
        } else if (arg == "-s" && i + 1 < argc) {
            sourceVertex = std::stoi(argv[++i]);
        } else if (arg == "-c" && i + 1 < argc) {
            numChanges = std::stoi(argv[++i]);
        } else if (arg == "-t" && i + 1 < argc) {
            numThreads = std::stoi(argv[++i]);
        } else if (arg == "-i" && i + 1 < argc) {
            insertRatio = std::stof(argv[++i]);
        } else if (arg == "-a" && i + 1 < argc) {
            asyncLevel = std::stoi(argv[++i]);
        } else if (arg == "-m" && i + 1 < argc) {
            maxIterations = std::stoi(argv[++i]);
        } else if (arg == "-o" && i + 1 < argc) {
            outputFile = argv[++i];
        } else if (arg == "-v") {
            verbose = true;
        } else if (arg == "-h") {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::cerr << "Unknown option: " << arg << std::endl;
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    // Validate parameters
    if (numThreads <= 0) numThreads = 1;
    if (insertRatio < 0.0f) insertRatio = 0.0f;
    if (insertRatio > 1.0f) insertRatio = 1.0f;
    if (asyncLevel < 0) asyncLevel = 0;
    if (maxIterations <= 0) maxIterations = 100;
    
    // Generate synthetic graph if requested
    if (generateGraph && rank == 0) {
        filename = "synthetic_" + std::to_string(genVertices) + "v_" + std::to_string(genEdges) + "e.edges";
        if (!generateSyntheticGraph(filename, genVertices, genEdges)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
    
    // Broadcast the filename to all processes if it was changed by rank 0
    int filenameLength = filename.length();
    MPI_Bcast(&filenameLength, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    char* filenameBuffer = new char[filenameLength + 1];
    if (rank == 0) {
        strcpy(filenameBuffer, filename.c_str());
    }
    MPI_Bcast(filenameBuffer, filenameLength + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        filename = std::string(filenameBuffer);
    }
    delete[] filenameBuffer;
    
    // Load graph on all ranks (in a real implementation, each process would load only its portion)
    Graph g;
    if (rank == 0) {
        std::cout << "Loading graph from " << filename << "..." << std::endl;
    }
    
    // Try to load graph with multiple attempts and fallback to sequential loading if needed
    bool graphLoaded = false;
    int loadAttempts = 0;
    const int MAX_LOAD_ATTEMPTS = 3;
    
    while (!graphLoaded && loadAttempts < MAX_LOAD_ATTEMPTS) {
        graphLoaded = g.loadFromEdgeFile(filename);
        if (!graphLoaded) {
            if (rank == 0) {
                std::cerr << "Error loading graph from " << filename << " (attempt " 
                         << loadAttempts + 1 << "/" << MAX_LOAD_ATTEMPTS << ")" << std::endl;
            }
            loadAttempts++;
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    
    if (!graphLoaded) {
        if (rank == 0) {
            std::cerr << "Failed to load graph after multiple attempts. Aborting." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        std::cout << "Graph loaded: " << g.getNumVertices() << " vertices, " 
                  << g.getNumEdges() << " edges" << std::endl;
    }
    
    // Generate random edge changes on rank 0 and broadcast
    std::vector<EdgeChange> changes;
    if (rank == 0) {
        std::cout << "Generating " << numChanges << " random edge changes..." << std::endl;
        changes = generateRandomChanges(g, numChanges, insertRatio);
        std::cout << "Changes generated: " << changes.size() << " changes ("
                  << (insertRatio * 100) << "% insertions)" << std::endl;
    }
    
    // Broadcast number of changes
    int changesSize = (rank == 0) ? changes.size() : 0;
    MPI_Bcast(&changesSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        changes.resize(changesSize);
    }
    
    // Broadcast changes to all processes
    for (int i = 0; i < changesSize; i++) {
        int source = (rank == 0) ? changes[i].source : 0;
        int target = (rank == 0) ? changes[i].target : 0;
        float weight = (rank == 0) ? changes[i].weight : 0.0f;
        int isInsert = (rank == 0) ? (changes[i].isInsert ? 1 : 0) : 0;
        
        MPI_Bcast(&source, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&target, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&weight, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&isInsert, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank != 0) {
            changes[i] = EdgeChange(source, target, weight, isInsert == 1);
        }
    }
    
    // Benchmark sequential implementation (only on rank 0)
    double sequentialTime = 0.0;
    if (rank == 0) {
        std::cout << "\nRunning sequential implementation..." << std::endl;
        
        SSSPTree sequentialTree(g.getNumVertices(), g.mapVertex(sourceVertex));
        sequentialTree.initialize(g, sourceVertex);
        
        auto start = std::chrono::high_resolution_clock::now();
        SequentialSSSP::updateSSSP(g, sequentialTree, changes);
        auto end = std::chrono::high_resolution_clock::now();
        
        sequentialTime = std::chrono::duration<double>(end - start).count();
        std::cout << "Sequential time: " << std::fixed << std::setprecision(6) << sequentialTime << " seconds" << std::endl;
    }
    
    // Warm-up run for the parallel implementation (to initialize memory, etc.)
    {
        ParallelSSSP warmupSSSP(rank, size);
        warmupSSSP.initialize(g, sourceVertex, numThreads, asyncLevel);
        
        // Small subset of changes for warmup
        std::vector<EdgeChange> warmupChanges;
        if (!changes.empty()) {
            warmupChanges.push_back(changes[0]);
        }
        
        warmupSSSP.updateSSSP(warmupChanges);
        MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes finish warmup
    }
    
    // Benchmark parallel implementation
    MPI_Barrier(MPI_COMM_WORLD);
    auto parallelStart = std::chrono::high_resolution_clock::now();
    
    ParallelSSSP parallelSSSP(rank, size);
    parallelSSSP.initialize(g, sourceVertex, numThreads, asyncLevel);
    parallelSSSP.setMaxIterations(maxIterations);
    parallelSSSP.updateSSSP(changes);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto parallelEnd = std::chrono::high_resolution_clock::now();
    double parallelTime = std::chrono::duration<double>(parallelEnd - parallelStart).count();
    
    if (rank == 0) {
        std::cout << "\nParallel implementation (MPI + OpenMP):" << std::endl;
        std::cout << "Parallel time: " << std::fixed << std::setprecision(6) << parallelTime << " seconds" << std::endl;
        
        if (sequentialTime > 0) {
            double speedup = sequentialTime / parallelTime;
            std::cout << "Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
            std::cout << "Efficiency: " << std::fixed << std::setprecision(2) 
                      << (speedup / (size * numThreads)) * 100 << "%" << std::endl;
        }
        
        // Gather results and verify correctness
        SSSPTree sequentialTree(g.getNumVertices(), g.mapVertex(sourceVertex));
        sequentialTree.initialize(g, sourceVertex);
        SequentialSSSP::updateSSSP(g, sequentialTree, changes);
        
        SSSPTree parallelTree(g.getNumVertices(), g.mapVertex(sourceVertex));
        parallelSSSP.gatherResults(g, parallelTree);
        
        // Verify correctness
        bool correct = true;
        int mismatchCount = 0;
        int infinityMismatch = 0;
        int totalVerified = 0;
        
        for (int v = 0; v < g.getNumVertices(); v++) {
            totalVerified++;
            
            // Check if distances match (within a small epsilon for floating point)
            if (std::abs(sequentialTree.distance[v] - parallelTree.distance[v]) > 1e-3) {
                // Allow both to be infinity
                if (std::isinf(sequentialTree.distance[v]) && std::isinf(parallelTree.distance[v])) {
                    continue;
                }
                
                // Track type of mismatch
                if (std::isinf(sequentialTree.distance[v]) || std::isinf(parallelTree.distance[v])) {
                    infinityMismatch++;
                }
                
                correct = false;
                mismatchCount++;
                
                if (mismatchCount < 10 && verbose) { // Limit the number of mismatches to display
                    int originalV = g.reverseMapVertex(v);
                    std::cout << "Mismatch at vertex " << originalV << ": "
                              << "Sequential = " << sequentialTree.distance[v] << ", "
                              << "Parallel = " << parallelTree.distance[v] << std::endl;
                }
            }
        }
        
        if (correct) {
            std::cout << "\nVerification: Results are correct! (" << totalVerified << " vertices verified)" << std::endl;
        } else {
            std::cout << "\nVerification: Results have " << mismatchCount << " mismatches out of " 
                      << totalVerified << " vertices (" << infinityMismatch << " infinity mismatches)" << std::endl;
            
            // Calculate error percentage
            double errorPercentage = (mismatchCount / static_cast<double>(totalVerified)) * 100.0;
            std::cout << "Error percentage: " << std::fixed << std::setprecision(2) 
                      << errorPercentage << "%" << std::endl;
        }
        
        // Print performance summary
        std::cout << "\nPerformance Summary:" << std::endl;
        std::cout << "=====================" << std::endl;
        std::cout << "Graph: " << filename << std::endl;
        std::cout << "Vertices: " << g.getNumVertices() << ", Edges: " << g.getNumEdges() << std::endl;
        std::cout << "Source vertex: " << sourceVertex << std::endl;
        std::cout << "Edge changes: " << changes.size() << " (" << (insertRatio * 100) << "% insertions)" << std::endl;
        std::cout << "MPI processes: " << size << std::endl;
        std::cout << "OpenMP threads per process: " << numThreads << std::endl;
        std::cout << "Asynchrony level: " << asyncLevel << std::endl;
        std::cout << "Sequential time: " << std::fixed << std::setprecision(6) << sequentialTime << " seconds" << std::endl;
        std::cout << "Parallel time: " << std::fixed << std::setprecision(6) << parallelTime << " seconds" << std::endl;
        
        if (sequentialTime > 0) {
            double speedup = sequentialTime / parallelTime;
            std::cout << "Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
            std::cout << "Efficiency: " << std::fixed << std::setprecision(2) 
                      << (speedup / (size * numThreads)) * 100 << "%" << std::endl;
        }
        
        // Write results to output file if specified
        if (!outputFile.empty()) {
            std::ofstream out(outputFile, std::ios::app);
            if (out.is_open()) {
                out << filename << ","
                    << g.getNumVertices() << ","
                    << g.getNumEdges() << ","
                    << changes.size() << ","
                    << insertRatio << ","
                    << size << ","
                    << numThreads << ","
                    << asyncLevel << ","
                    << sequentialTime << ","
                    << parallelTime << ","
                    << (sequentialTime > 0 ? sequentialTime / parallelTime : 0) << ","
                    << (sequentialTime > 0 ? (sequentialTime / parallelTime) / (size * numThreads) * 100 : 0) << ","
                    << mismatchCount << ","
                    << (mismatchCount / static_cast<double>(totalVerified)) * 100.0 << std::endl;
                out.close();
                std::cout << "Results saved to " << outputFile << std::endl;
            } else {
                std::cerr << "Error: Could not open output file " << outputFile << std::endl;
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
