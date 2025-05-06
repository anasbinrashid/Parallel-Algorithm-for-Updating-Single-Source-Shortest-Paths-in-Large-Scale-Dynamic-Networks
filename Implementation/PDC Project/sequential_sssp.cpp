#include "sequential_sssp.h"
#include <iostream>
#include <queue>
#include <limits>

void SequentialSSSP::updateSSSP(const Graph& g, SSSPTree& tree, const std::vector<EdgeChange>& changes) {
    // Step 1: Identify vertices affected by changes
    identifyAffectedVertices(g, tree, changes);
    
    // Step 2: Update affected subgraphs
    updateAffectedSubgraphs(g, tree);
}

void SequentialSSSP::identifyAffectedVertices(const Graph& g, SSSPTree& tree, const std::vector<EdgeChange>& changes) {
    // Process each changed edge
    for (const EdgeChange& edge : changes) {
        if (!edge.isInsert) {
            // Process edge deletion
            processEdgeDeletion(g, tree, edge);
        } else {
            // Process edge insertion
            processEdgeInsertion(g, tree, edge);
        }
    }
}

void SequentialSSSP::processEdgeDeletion(const Graph& g, SSSPTree& tree, const EdgeChange& edge) {
    int u = g.mapVertex(edge.source);
    int v = g.mapVertex(edge.target);
    
    // Skip if vertices don't exist
    if (u == -1 || v == -1) {
        return;
    }
    
    // Check if the edge is part of the SSSP tree
    if (tree.parent[v] == u || tree.parent[u] == v) {
        // Determine which vertex has greater distance (child in SSSP tree)
        int child = (tree.distance[u] > tree.distance[v]) ? u : v;
        
        // Mark the child as affected by deletion
        tree.distance[child] = std::numeric_limits<float>::infinity();
        tree.parent[child] = -1;
        tree.affectedDel[child] = true;
        tree.affected[child] = true;
    }
}

void SequentialSSSP::processEdgeInsertion(const Graph& g, SSSPTree& tree, const EdgeChange& edge) {
    int u = g.mapVertex(edge.source);
    int v = g.mapVertex(edge.target);
    
    // Skip if vertices don't exist
    if (u == -1 || v == -1) {
        return;
    }
    
    // Determine which vertex has the smaller distance from source (closer to root)
    int parent, child;
    if (tree.distance[u] < tree.distance[v]) {
        parent = u;
        child = v;
    } else {
        parent = v;
        child = u;
    }
    
    // Check if the new edge provides a shorter path
    float newDistance = tree.distance[parent] + edge.weight;
    if (newDistance < tree.distance[child]) {
        tree.distance[child] = newDistance;
        tree.parent[child] = parent;
        tree.affected[child] = true;
    }
}

void SequentialSSSP::updateAffectedSubgraphs(const Graph& g, SSSPTree& tree) {
    // First, update vertices affected by deletion (mark all descendants as affected)
    bool hasDeleteAffected = true;
    while (hasDeleteAffected) {
        hasDeleteAffected = false;
        
        for (int v = 0; v < tree.numVertices; v++) {
            if (tree.affectedDel[v]) {
                // Reset this flag
                tree.affectedDel[v] = false;
                hasDeleteAffected = true;
                
                // Update the disconnected subtree
                updateDisconnectedSubtree(g, tree, v);
            }
        }
    }
    
    // Then update all affected vertices
    bool hasAffected = true;
    while (hasAffected) {
        hasAffected = false;
        
        for (int v = 0; v < tree.numVertices; v++) {
            if (tree.affected[v]) {
                // Reset this flag
                tree.affected[v] = false;
                
                int originalV = g.reverseMapVertex(v);
                
                // Process neighbors of this affected vertex
                for (const auto& edge : g.getNeighbors(originalV)) {
                    int n = edge.first;
                    float weight = edge.second;
                    
                    // Check if neighbor distance can be improved through v
                    if (tree.distance[n] > tree.distance[v] + weight) {
                        tree.distance[n] = tree.distance[v] + weight;
                        tree.parent[n] = v;
                        tree.affected[n] = true;
                        hasAffected = true;
                    }
                    
                    // Check if v's distance can be improved through neighbor
                    if (tree.distance[v] > tree.distance[n] + weight) {
                        tree.distance[v] = tree.distance[n] + weight;
                        tree.parent[v] = n;
                        tree.affected[v] = true;
                        hasAffected = true;
                    }
                }
            }
        }
    }
}

void SequentialSSSP::updateDisconnectedSubtree(const Graph& g, SSSPTree& tree, int vertex) {
    // Find all vertices in the subtree rooted at 'vertex'
    std::queue<int> q;
    q.push(vertex);
    
    while (!q.empty()) {
        int curr = q.front();
        q.pop();
        
        // Find all children of current vertex in the SSSP tree
        for (int child = 0; child < tree.numVertices; child++) {
            if (tree.parent[child] == curr) {
                // This is a child in the SSSP tree
                tree.distance[child] = std::numeric_limits<float>::infinity();
                tree.parent[child] = -1;
                tree.affectedDel[child] = true;
                tree.affected[child] = true;
                
                // Process this child's subtree
                q.push(child);
            }
        }
    }
}
