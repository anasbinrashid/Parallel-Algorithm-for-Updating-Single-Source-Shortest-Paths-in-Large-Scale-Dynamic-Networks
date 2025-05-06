#ifndef METIS_WRAPPER_H
#define METIS_WRAPPER_H

#include "graph.h"
#include <metis.h>
#include <vector>

// Partition graph using METIS
std::vector<idx_t> partition_graph(const Graph& graph, int num_parts);

#endif // METIS_WRAPPER_H
