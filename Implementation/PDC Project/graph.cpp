#include "graph.h"

// Graph class implementation
Graph::Graph() : numVertices(0) {}

Graph::Graph(const Graph& other) : 
    numVertices(other.numVertices),
    adjacencyList(other.adjacencyList),
    vertexMap(other.vertexMap),
    reverseMap(other.reverseMap) {}

bool Graph::loadFromEdgeFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }
    
    int u, v;
    float weight;
    int vertexCounter = 0;
    
    // Clear existing data
    adjacencyList.clear();
    vertexMap.clear();
    reverseMap.clear();
    
    // Read edges from file
    while (file >> u >> v >> weight) {
        // Map vertices to internal indices
        if (vertexMap.find(u) == vertexMap.end()) {
            vertexMap[u] = vertexCounter;
            reverseMap[vertexCounter] = u;
            vertexCounter++;
        }
        
        if (vertexMap.find(v) == vertexMap.end()) {
            vertexMap[v] = vertexCounter;
            reverseMap[vertexCounter] = v;
            vertexCounter++;
        }
        
        // Ensure adjacency list has enough space
        if (adjacencyList.size() <= static_cast<size_t>(vertexCounter)) {
            adjacencyList.resize(vertexCounter);
        }
        
        // Add edge
        int mappedU = vertexMap[u];
        int mappedV = vertexMap[v];
        
        // Add directed edge (u,v)
        adjacencyList[mappedU].push_back(std::make_pair(mappedV, weight));
    }
    
    numVertices = vertexCounter;
    adjacencyList.resize(numVertices);
    
    file.close();
    return true;
}

void Graph::addEdge(int u, int v, float weight) {
    // Map vertices if they don't exist
    if (vertexMap.find(u) == vertexMap.end()) {
        vertexMap[u] = numVertices;
        reverseMap[numVertices] = u;
        adjacencyList.resize(numVertices + 1);
        numVertices++;
    }
    
    if (vertexMap.find(v) == vertexMap.end()) {
        vertexMap[v] = numVertices;
        reverseMap[numVertices] = v;
        adjacencyList.resize(numVertices + 1);
        numVertices++;
    }
    
    int mappedU = vertexMap[u];
    int mappedV = vertexMap[v];
    
    // Check if edge already exists and update weight
    for (auto& edge : adjacencyList[mappedU]) {
        if (edge.first == mappedV) {
            edge.second = weight;
            return;
        }
    }
    
    // Add new edge
    adjacencyList[mappedU].push_back(std::make_pair(mappedV, weight));
}

void Graph::removeEdge(int u, int v) {
    if (vertexMap.find(u) == vertexMap.end() || vertexMap.find(v) == vertexMap.end()) {
        return; // Vertices don't exist
    }
    
    int mappedU = vertexMap[u];
    int mappedV = vertexMap[v];
    
    // Find and remove the edge
    auto& neighbors = adjacencyList[mappedU];
    neighbors.erase(
        std::remove_if(neighbors.begin(), neighbors.end(),
            [mappedV](const std::pair<int, float>& edge) {
                return edge.first == mappedV;
            }),
        neighbors.end()
    );
}

bool Graph::hasEdge(int u, int v) const {
    if (vertexMap.find(u) == vertexMap.end() || vertexMap.find(v) == vertexMap.end()) {
        return false; // Vertices don't exist
    }
    
    int mappedU = vertexMap.at(u);
    int mappedV = vertexMap.at(v);
    
    for (const auto& edge : adjacencyList[mappedU]) {
        if (edge.first == mappedV) {
            return true;
        }
    }
    
    return false;
}

float Graph::getEdgeWeight(int u, int v) const {
    if (!hasEdge(u, v)) {
        return std::numeric_limits<float>::infinity();
    }
    
    int mappedU = vertexMap.at(u);
    int mappedV = vertexMap.at(v);
    
    for (const auto& edge : adjacencyList[mappedU]) {
        if (edge.first == mappedV) {
            return edge.second;
        }
    }
    
    return std::numeric_limits<float>::infinity();
}

int Graph::getNumVertices() const {
    return numVertices;
}

int Graph::getNumEdges() const {
    int count = 0;
    for (const auto& neighbors : adjacencyList) {
        count += neighbors.size();
    }
    return count;
}

const std::vector<std::pair<int, float>>& Graph::getNeighbors(int v) const {
    static std::vector<std::pair<int, float>> empty;
    
    if (vertexMap.find(v) == vertexMap.end()) {
        return empty;
    }
    return adjacencyList[vertexMap.at(v)];
}

std::vector<int> Graph::getAllVertices() const {
    std::vector<int> vertices;
    for (const auto& pair : vertexMap) {
        vertices.push_back(pair.first);
    }
    return vertices;
}

int Graph::mapVertex(int v) const {
    auto it = vertexMap.find(v);
    if (it == vertexMap.end()) {
        return -1; // Vertex not found
    }
    return it->second;
}

int Graph::reverseMapVertex(int v) const {
    auto it = reverseMap.find(v);
    if (it == reverseMap.end()) {
        return -1; // Internal vertex not found
    }
    return it->second;
}

const std::unordered_map<int, int>& Graph::getVertexMap() const {
    return vertexMap;
}

const std::unordered_map<int, int>& Graph::getReverseMap() const {
    return reverseMap;
}

Graph Graph::createSubgraph(const std::vector<int>& vertices) {
    Graph subgraph;
    
    // Add vertices and edges to subgraph
    for (int v : vertices) {
        if (vertexMap.find(v) == vertexMap.end()) {
            continue; // Skip if vertex doesn't exist
        }
        
        int mappedV = vertexMap[v];
        
        for (const auto& edge : adjacencyList[mappedV]) {
            int target = reverseMap[edge.first];
            
            // Only add edge if target is also in the subgraph
            if (std::find(vertices.begin(), vertices.end(), target) != vertices.end()) {
                subgraph.addEdge(v, target, edge.second);
            }
        }
    }
    
    return subgraph;
}

void Graph::printGraph() const {
    std::cout << "Graph with " << numVertices << " vertices and " << getNumEdges() << " edges:" << std::endl;
    
    for (int i = 0; i < numVertices; ++i) {
        int originalVertex = reverseMap.at(i);
        std::cout << "Vertex " << originalVertex << ": ";
        for (const auto& edge : adjacencyList[i]) {
            std::cout << "(" << reverseMap.at(edge.first) << ", " << edge.second << ") ";
        }
        std::cout << std::endl;
    }
}

// SSSPTree implementation
SSSPTree::SSSPTree(int numVerts, int source) : 
    sourceVertex(source),
    numVertices(numVerts),
    parent(numVerts, -1),
    distance(numVerts, std::numeric_limits<float>::infinity()),
    affectedDel(numVerts, false),
    affected(numVerts, false) {
    
    // Set source distance to 0
    if (source >= 0 && source < numVerts) {
        distance[source] = 0.0f;
    }
}

void SSSPTree::initialize(const Graph& g, int source) {
    // Map the source vertex to internal index
    int mappedSource = g.mapVertex(source);
    if (mappedSource == -1) {
        std::cerr << "Error: Source vertex not found in graph" << std::endl;
        return;
    }
    
    sourceVertex = mappedSource;
    numVertices = g.getNumVertices();
    
    // Reset vectors
    parent.assign(numVertices, -1);
    distance.assign(numVertices, std::numeric_limits<float>::infinity());
    affectedDel.assign(numVertices, false);
    affected.assign(numVertices, false);
    
    // Set source distance to 0
    distance[mappedSource] = 0.0f;
    
    // Priority queue for Dijkstra's algorithm
    std::priority_queue<
        std::pair<float, int>,
        std::vector<std::pair<float, int>>,
        std::greater<std::pair<float, int>>
    > pq;
    
    pq.push(std::make_pair(0.0f, mappedSource));
    
    // Dijkstra's algorithm
    while (!pq.empty()) {
        float dist = pq.top().first;
        int u = pq.top().second;
        pq.pop();
        
        // Skip if we already found a better path
        if (dist > distance[u]) {
            continue;
        }
        
        // Process neighbors
        int originalU = g.reverseMapVertex(u);
        for (const auto& edge : g.getNeighbors(originalU)) {
            int v = edge.first;
            float weight = edge.second;
            
            // Relaxation step
            if (distance[u] + weight < distance[v]) {
                distance[v] = distance[u] + weight;
                parent[v] = u;
                pq.push(std::make_pair(distance[v], v));
            }
        }
    }
}

int SSSPTree::getSource() const {
    return sourceVertex;
}

bool SSSPTree::isValid() const {
    // Check if source distance is 0
    if (sourceVertex >= 0 && sourceVertex < numVertices && 
        distance[sourceVertex] != 0.0f) {
        return false;
    }
    
    // Check if there are negative cycles (parent cycle detection)
    for (int i = 0; i < numVertices; ++i) {
        if (i == sourceVertex) continue;
        
        // Skip disconnected vertices
        if (distance[i] == std::numeric_limits<float>::infinity()) {
            continue;
        }
        
        // Check for cycles in parent relationship
        std::vector<bool> visited(numVertices, false);
        int current = i;
        
        while (current != -1 && !visited[current]) {
            visited[current] = true;
            current = parent[current];
        }
        
        if (current != -1 && current != sourceVertex) {
            return false; // Cycle detected
        }
    }
    
    return true;
}

void SSSPTree::printTree(const Graph& g) const {
    std::cout << "SSSP Tree from source " << g.reverseMapVertex(sourceVertex) << ":" << std::endl;
    
    for (int i = 0; i < numVertices; ++i) {
        int originalVertex = g.reverseMapVertex(i);
        std::cout << "Vertex " << originalVertex << ": ";
        
        if (distance[i] == std::numeric_limits<float>::infinity()) {
            std::cout << "Unreachable" << std::endl;
        } else {
            std::cout << "Distance = " << distance[i] << ", Parent = ";
            if (parent[i] == -1) {
                std::cout << "None (Source)" << std::endl;
            } else {
                std::cout << g.reverseMapVertex(parent[i]) << std::endl;
            }
        }
    }
}
