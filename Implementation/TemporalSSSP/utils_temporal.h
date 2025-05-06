/**
 * utils_temporal.h
 * Additional utility functions for temporal graphs
 */

#ifndef UTILS_TEMPORAL_H
#define UTILS_TEMPORAL_H

#include "types.h"
#include "temporal_graph.h"
#include <string>
#include <vector>

// Save temporal graph to file
void saveTemporalGraphToFile(const TemporalGraph& graph, const std::string& filename);

// Save edge changes to file
void saveEdgeChangesToFile(const std::vector<EdgeChange>& changes, 
                         const std::string& filename);

// Load edge changes from file
std::vector<EdgeChange> loadEdgeChangesFromFile(const std::string& filename);

// Verify SSSP solution against Dijkstra's algorithm for temporal graph
bool verifyTemporalSSSP(const TemporalGraph& graph, VertexID source, 
                      const std::vector<SSSPNode>& sssp);

// Generate visualization of SSSP tree for temporal graph (DOT format)
void generateTemporalSSSPVisualization(const TemporalGraph& graph, 
                                     const std::vector<SSSPNode>& sssp, 
                                     VertexID source, Timestamp time,
                                     const std::string& filename);

// Generate temporal evolution animation script (for GraphViz)
void generateTemporalEvolutionAnimation(const TemporalGraph& graph,
                                      const std::map<Timestamp, std::vector<SSSPNode>>& sssp_trees,
                                      VertexID source,
                                      const std::string& filename);

// Generate a random temporal graph for testing
TemporalGraph generateRandomTemporalGraph(int num_vertices, int num_initial_edges, 
                                        int num_timepoints, int changes_per_timepoint);

#endif // UTILS_TEMPORAL_H
