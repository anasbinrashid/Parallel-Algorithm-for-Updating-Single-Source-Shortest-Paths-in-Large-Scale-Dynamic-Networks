# Parallel Algorithm for Updating Single-Source Shortest Paths in Dynamic Networks

## Project Overview

This project implements and analyzes a parallel algorithm for updating Single-Source Shortest Paths (SSSP) in large-scale dynamic networks based on the paper "A Parallel Algorithm Template for Updating Single-Source Shortest Paths in Large-Scale Dynamic Networks" by Khanda et al. Our implementation focuses on providing efficient solutions for handling network changes without requiring complete recomputation of shortest paths.

### Problem Statement

In real-world applications, networks are rarely static - they continuously evolve with nodes and edges being added or removed. Traditional SSSP algorithms operate on static graphs, requiring complete recomputation when the network structure changes. This approach becomes prohibitively expensive for large-scale networks. Our project addresses this gap by implementing an efficient parallel algorithm for incrementally updating SSSP trees when network changes occur.

## Key Features

- **Parallel Implementation**: Utilizing both MPI (for distributed memory) and OpenMP (for shared memory) parallelism
- **Dynamic Network Support**: Efficiently handling both edge insertions and deletions
- **Rooted Tree Data Structure**: Maintaining and updating a specialized data structure for SSSP information
- **Graph Partitioning**: Using METIS for optimal graph partitioning to enhance parallel processing
- **Scalability**: Designed to handle large-scale networks with millions of vertices and edges

## Implementation Approach

Our implementation follows a two-step algorithm:

1. **Identify Affected Subgraphs**: Identify portions of the network affected by changes (edge additions/deletions)
2. **Update SSSP Information**: Update the distances and parent relationships only for affected vertices

The algorithm maintains the SSSP tree as a rooted structure and efficiently traverses and updates this structure as the network evolves, avoiding the overhead of full recomputation.

## Tools and Technologies

- **MPI (Message Passing Interface)**: For distributed memory parallelism across multiple nodes
- **OpenMP**: For shared memory parallelism within each compute node
- **METIS**: For optimal graph partitioning to distribute workload efficiently
- **C/C++**: Core implementation language
- **Python**: For results analysis and visualization
- **Public Graph Datasets**: For testing and performance evaluation

## Performance Benefits

Our implementation offers several advantages over traditional approaches:

- **Reduced Computation Time**: By updating only affected portions of the graph instead of recomputing from scratch
- **Scalability**: Efficiently scales with increasing graph sizes and processor counts
- **Memory Efficiency**: Requires less memory than full recomputation approaches
- **Adaptability**: Performs well across various network structures and change patterns
- **Practical Applicability**: Suitable for real-world applications with dynamic network structures

Based on the original paper's findings, we expect our implementation to demonstrate:
- Up to 5.6x speedup compared to state-of-the-art GPU-based recomputation approaches
- Up to 5x speedup compared to shared-memory recomputation algorithms

## Datasets

We test our implementation on several large-scale networks, including:
- Social networks (Orkut, LiveJournal)
- Synthetic networks (RMAT, Graph500)
- Domain-specific networks

## How to Run

### Prerequisites
- MPI implementation (e.g., MPICH, OpenMPI)
- OpenMP-compatible compiler (e.g., GCC 4.9+)
- METIS library installed
- CMake 3.10+

### Building the Project
```bash
mkdir build && cd build
cmake ..
make
```

### Running Tests
```bash
mpirun -np <NUM_PROCESSES> ./sssp_update <INPUT_GRAPH> <CHANGES_FILE> <OUTPUT_FILE>
```

## Project Timeline

- **Phase 1 (April 21)**: Research paper presentation and implementation strategy
- **Phase 2 (May 5)**: Implementation demonstration and performance analysis



## Team Members

- Amna Tahir 
- Hasnain Akhtar
- Anas Bin Rashid

## Acknowledgements

We acknowledge the authors of the original paper:
Arindam Khanda, Sriram Srinivasan, Sanjukta Bhowmick, Boyana Norris, and Sajal K. Das
