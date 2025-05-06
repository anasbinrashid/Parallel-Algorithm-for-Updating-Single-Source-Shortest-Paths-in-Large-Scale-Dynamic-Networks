#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <limits>
#include <iostream>
#include <mpi.h>

// Declare global verbose logging flag
extern bool g_verbose_logging;

typedef int Vertex;
typedef double Weight;
const Weight INF = std::numeric_limits<Weight>::infinity();

struct Edge {
    Vertex source;
    Vertex target;
    Weight weight;
    
    Edge(Vertex s, Vertex t, Weight w) : source(s), target(t), weight(w) {}
    
    bool operator==(const Edge& other) const {
        return source == other.source && target == other.target;
    }
};

// Struct for communicating edge changes between processes
struct EdgeChange {
    Vertex source;
    Vertex target;
    Weight weight;
    char operation; // '+' for addition, '-' for removal
    
    EdgeChange() : source(0), target(0), weight(0), operation('+') {}
    EdgeChange(Vertex s, Vertex t, Weight w, char op) 
        : source(s), target(t), weight(w), operation(op) {}
};

// Create MPI datatype for EdgeChange
MPI_Datatype create_edge_change_type();

class Graph {
private:
    int num_vertices;
    std::vector<std::vector<std::pair<Vertex, Weight>>> adj_list;
    bool is_directed;
    
    // For distributed processing
    std::vector<int> partition;          // Partition assignment for each vertex
    std::vector<Vertex> local_vertices;  // Vertices owned by this process
    std::vector<Vertex> ghost_vertices;  // Boundary vertices from other partitions
    std::set<std::pair<Vertex, Vertex>> boundary_edges;  // Edges crossing partition boundaries

public:
    Graph(int n, bool directed = false);
    
    // Load graph from edge file
    static Graph load_from_file(const std::string& filename);
    
    // Add edge to the graph
    void add_edge(Vertex source, Vertex target, Weight weight);
    
    // Remove edge from the graph
    void remove_edge(Vertex source, Vertex target);
    
    // Get neighbors of a vertex
    const std::vector<std::pair<Vertex, Weight>>& get_neighbors(Vertex v) const;
    
    // Get number of vertices
    int get_num_vertices() const;
    
    // Get number of edges
    int get_num_edges() const;
    
    // Get edge weight
    Weight get_edge_weight(Vertex source, Vertex target) const;
    
    // Check if edge exists
    bool has_edge(Vertex source, Vertex target) const;
    
    // Apply batch changes to graph
    void apply_changes(const std::string& changes_file);
    
    // Distribute graph using METIS partitioning
    void distribute_graph(const std::vector<int>& parts, int rank, int num_procs);
    
    // Get local vertices
    const std::vector<Vertex>& get_local_vertices() const;
    
    // Get ghost vertices
    const std::vector<Vertex>& get_ghost_vertices() const;
    
    // Get boundary edges
    const std::set<std::pair<Vertex, Vertex>>& get_boundary_edges() const;
    
    // Check if vertex is local to this process
    bool is_local_vertex(Vertex v) const;
    
    // Get partition ID of a vertex
    int get_partition(Vertex v) const;
    
    // Broadcast edge changes to all processes
    std::vector<EdgeChange> broadcast_changes(const std::string& changes_file, int root_rank);
};

#endif // GRAPH_H
