#include "graph.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_set>
#include <map>

// Define global verbose logging flag
bool g_verbose_logging = true; // Changed to true to enable debug logging

// Debug macro
#define DEBUG(rank, msg) if (g_verbose_logging) { std::cout << "[DEBUG] Rank " << rank << ": " << msg << std::endl; }

MPI_Datatype create_edge_change_type() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //DEBUG(rank, "Creating EdgeChange MPI datatype");
    
    MPI_Datatype edge_change_type;
    int blocklengths[4] = {1, 1, 1, 1};
    MPI_Datatype types[4] = {MPI_INT, MPI_INT, MPI_DOUBLE, MPI_CHAR};
    MPI_Aint offsets[4];
    
    EdgeChange temp;
    
    MPI_Aint base_address;
    MPI_Get_address(&temp, &base_address);
    MPI_Get_address(&temp.source, &offsets[0]);
    MPI_Get_address(&temp.target, &offsets[1]);
    MPI_Get_address(&temp.weight, &offsets[2]);
    MPI_Get_address(&temp.operation, &offsets[3]);
    
    for (int i = 0; i < 4; i++) {
        offsets[i] = MPI_Aint_diff(offsets[i], base_address);
        //DEBUG(rank, "Offset[" << i << "] = " << offsets[i]);
    }
    
    MPI_Type_create_struct(4, blocklengths, offsets, types, &edge_change_type);
    MPI_Type_commit(&edge_change_type);
    
    //DEBUG(rank, "EdgeChange MPI datatype created successfully");
    return edge_change_type;
}

Graph::Graph(int n, bool directed) 
    : num_vertices(n), adj_list(n), is_directed(directed), partition(n, -1) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    DEBUG(rank, "Creating graph with " << n << " vertices, directed=" << directed);
}

Graph Graph::load_from_file(const std::string& filename) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    DEBUG(rank, "Loading graph from file: " << filename);
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        break;
    }
    
    int max_vertex = -1;
    std::ifstream vertex_scan(filename);
    int src, dst;
    double weight;
    
    while (vertex_scan >> src >> dst >> weight) {
        max_vertex = std::max(max_vertex, std::max(src, dst));
    }
    vertex_scan.close();
    
    //DEBUG(rank, "Max vertex ID found: " << max_vertex);
    Graph graph(max_vertex + 1);
    
    file.clear();
    file.seekg(0);
    int edge_count = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        
        std::istringstream iss(line);
        if (!(iss >> src >> dst >> weight)) {
            continue;
        }
        
        graph.add_edge(src, dst, weight);
        edge_count++;
    }
    
    file.close();
    DEBUG(rank, "Loaded graph with " << graph.get_num_vertices() << " vertices and " << edge_count << " edges");
    return graph;
}

void Graph::add_edge(Vertex source, Vertex target, Weight weight) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (source >= num_vertices || target >= num_vertices) {
        int new_size = std::max(source, target) + 1;
        //DEBUG(rank, "Resizing graph from " << num_vertices << " to " << new_size << " vertices");
        adj_list.resize(new_size);
        partition.resize(new_size, -1);
        num_vertices = new_size;
    }
    
    adj_list[source].push_back(std::make_pair(target, weight));
    if (!is_directed) {
        adj_list[target].push_back(std::make_pair(source, weight));
    }
    
    //DEBUG(rank, "Added edge " << source << "->" << target << " (weight=" << weight << ")");
}

void Graph::remove_edge(Vertex source, Vertex target) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    //DEBUG(rank, "Removing edge " << source << "->" << target);
    
    if (source < num_vertices) {
        auto& neighbors = adj_list[source];
        int before_size = neighbors.size();
        neighbors.erase(
            std::remove_if(
                neighbors.begin(), 
                neighbors.end(),
                [target](const std::pair<Vertex, Weight>& edge) { 
                    return edge.first == target; 
                }
            ),
            neighbors.end()
        );
        int after_size = neighbors.size();
        
        if (before_size == after_size) {
            //DEBUG(rank, "Warning: Edge " << source << "->" << target << " not found for removal");
        }
    } else {
        //DEBUG(rank, "Warning: Source vertex " << source << " out of range for edge removal");
    }
    
    if (!is_directed && target < num_vertices) {
        auto& neighbors = adj_list[target];
        int before_size = neighbors.size();
        neighbors.erase(
            std::remove_if(
                neighbors.begin(), 
                neighbors.end(),
                [source](const std::pair<Vertex, Weight>& edge) { 
                    return edge.first == source; 
                }
            ),
            neighbors.end()
        );
        int after_size = neighbors.size();
        
        if (before_size == after_size) {
            //DEBUG(rank, "Warning: Edge " << target << "->" << source << " not found for removal");
        }
    } else if (!is_directed) {
        //DEBUG(rank, "Warning: Target vertex " << target << " out of range for edge removal");
    }
    
    boundary_edges.erase(std::make_pair(source, target));
    if (!is_directed) {
        boundary_edges.erase(std::make_pair(target, source));
    }
}

const std::vector<std::pair<Vertex, Weight>>& Graph::get_neighbors(Vertex v) const {
    if (v >= num_vertices) {
        static std::vector<std::pair<Vertex, Weight>> empty;
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        //DEBUG(rank, "Warning: get_neighbors called with out of range vertex " << v);
        return empty;
    }
    return adj_list[v];
}

int Graph::get_num_vertices() const {
    return num_vertices;
}

int Graph::get_num_edges() const {
    int count = 0;
    for (const auto& neighbors : adj_list) {
        count += neighbors.size();
    }
    return is_directed ? count : count / 2;
}

Weight Graph::get_edge_weight(Vertex source, Vertex target) const {
    if (source >= num_vertices) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        //DEBUG(rank, "Warning: get_edge_weight called with out of range source " << source);
        return INF;
    }
    
    for (const auto& edge : adj_list[source]) {
        if (edge.first == target) {
            return edge.second;
        }
    }
    
    return INF;
}

bool Graph::has_edge(Vertex source, Vertex target) const {
    if (source >= num_vertices) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        //DEBUG(rank, "Warning: has_edge called with out of range source " << source);
        return false;
    }
    
    for (const auto& edge : adj_list[source]) {
        if (edge.first == target) {
            return true;
        }
    }
    
    return false;
}

void Graph::apply_changes(const std::string& changes_file) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //DEBUG(rank, "Applying changes from file: " << changes_file);
    
    std::ifstream file(changes_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open changes file: " << changes_file << std::endl;
        exit(1);
    }
    
    int src, dst;
    double weight;
    char op;
    int change_count = 0;
    
    while (file >> src >> dst >> weight >> op) {
        if (op == '+') {
            add_edge(src, dst, weight);
        } else if (op == '-') {
            remove_edge(src, dst);
        }
        change_count++;
    }
    
    file.close();
    //DEBUG(rank, "Applied " << change_count << " changes from file");
}

void Graph::distribute_graph(const std::vector<int>& parts, int rank, int num_procs) {
    //DEBUG(rank, "Distributing graph to " << num_procs << " processes");
    
    if (static_cast<size_t>(parts.size()) < static_cast<size_t>(num_vertices)) {
        std::cerr << "Error: partition array size doesn't match number of vertices" << std::endl;
        exit(1);
    }
    
    partition = parts;
    local_vertices.clear();
    ghost_vertices.clear();
    boundary_edges.clear();
    
    for (Vertex v = 0; v < num_vertices; v++) {
        if (partition[v] == rank) {
            local_vertices.push_back(v);
        }
    }
    
    //DEBUG(rank, "Local vertex count: " << local_vertices.size());
    
    std::unordered_set<Vertex> ghost_set;
    for (Vertex v : local_vertices) {
        for (const auto& edge : adj_list[v]) {
            Vertex neighbor = edge.first;
            if (partition[neighbor] != rank && partition[neighbor] >= 0 && partition[neighbor] < num_procs) {
                ghost_set.insert(neighbor);
                boundary_edges.insert(std::make_pair(v, neighbor));
                if (!is_directed) {
                    boundary_edges.insert(std::make_pair(neighbor, v));
                }
            }
        }
    }
    
    ghost_vertices.assign(ghost_set.begin(), ghost_set.end());
    
    //DEBUG(rank, "Local vertices: " << local_vertices.size() 
          //<< ", Ghost vertices: " << ghost_vertices.size() 
          //<< ", Boundary edges: " << boundary_edges.size());
    
    // Print the first few local and ghost vertices for debugging
    if (g_verbose_logging) {
        std::stringstream local_ss, ghost_ss;
        local_ss << "Sample local vertices: ";
        for (size_t i = 0; i < std::min(local_vertices.size(), (size_t)5); i++) {
            local_ss << local_vertices[i] << " ";
        }
        if (local_vertices.size() > 5) local_ss << "...";
        
        ghost_ss << "Sample ghost vertices: ";
        for (size_t i = 0; i < std::min(ghost_vertices.size(), (size_t)5); i++) {
            ghost_ss << ghost_vertices[i] << " ";
        }
        if (ghost_vertices.size() > 5) ghost_ss << "...";
        
        //DEBUG(rank, local_ss.str());
        //DEBUG(rank, ghost_ss.str());
    }
}

const std::vector<Vertex>& Graph::get_local_vertices() const {
    return local_vertices;
}

const std::vector<Vertex>& Graph::get_ghost_vertices() const {
    return ghost_vertices;
}

const std::set<std::pair<Vertex, Vertex>>& Graph::get_boundary_edges() const {
    return boundary_edges;
}

bool Graph::is_local_vertex(Vertex v) const {
    if (v >= num_vertices) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        //DEBUG(rank, "Warning: is_local_vertex called with out of range vertex " << v);
        return false;
    }
    return std::find(local_vertices.begin(), local_vertices.end(), v) != local_vertices.end();
}

int Graph::get_partition(Vertex v) const {
    if (v < 0 || static_cast<size_t>(v) >= partition.size()) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        //DEBUG(rank, "Warning: get_partition called with out of range vertex " << v);
        return -1;
    }
    return partition[v];
}

std::vector<EdgeChange> Graph::broadcast_changes(const std::string& changes_file, int root_rank) {
    std::vector<EdgeChange> changes;
    int num_changes = 0;
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //DEBUG(rank, "Broadcasting changes from " << changes_file << ", root=" << root_rank);
    
    MPI_Datatype edge_change_type = create_edge_change_type();
    
    if (rank == root_rank) {
        std::ifstream file(changes_file);
        if (!file.is_open()) {
            std::cerr << "Failed to open changes file: " << changes_file << std::endl;
            exit(1);
        }
        
        int src, dst;
        double weight;
        char op;
        
        while (file >> src >> dst >> weight >> op) {
            changes.push_back(EdgeChange(src, dst, weight, op));
        }
        
        file.close();
        num_changes = changes.size();
        //DEBUG(root_rank, "Read " << num_changes << " changes from file");
    }
    
    // Broadcast the number of changes
    //DEBUG(rank, "Broadcasting number of changes: " << num_changes);
    MPI_Bcast(&num_changes, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
    //DEBUG(rank, "After broadcast, num_changes = " << num_changes);
    
    if (rank != root_rank) {
        //DEBUG(rank, "Resizing changes array to " << num_changes);
        changes.resize(num_changes);
    }
    
    // Broadcast the changes
    //DEBUG(rank, "Broadcasting changes array");
    MPI_Bcast(changes.data(), num_changes, edge_change_type, root_rank, MPI_COMM_WORLD);
    //DEBUG(rank, "Changes broadcast complete");
    
    MPI_Type_free(&edge_change_type);
    
    // Print a sample of changes for debugging
    if (g_verbose_logging) {
        std::stringstream ss;
        ss << "Sample changes: ";
        for (size_t i = 0; i < std::min(changes.size(), (size_t)3); i++) {
            ss << changes[i].operation << "(" << changes[i].source << "," 
               << changes[i].target << "," << changes[i].weight << ") ";
        }
        if (changes.size() > 3) ss << "...";
        //DEBUG(rank, ss.str());
    }
    
    return changes;
}
