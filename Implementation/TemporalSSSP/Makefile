CC = g++
MPICC = mpic++
CFLAGS = -std=c++11 -Wall -O3
OPENMP_FLAGS = -fopenmp
METIS_FLAGS = -DUSE_METIS -I/usr/include/metis -lmetis
LDFLAGS = -L/path/to/metis/lib -lmetis

# Temporal Hybrid version
temporal_hybrid: main_temporal_hybrid.o temporal_graph.o utils_temporal.o sssp_temporal_hybrid.o
	$(MPICC) $(CFLAGS) $(OPENMP_FLAGS) $(METIS_FLAGS) -o sssp_temporal_hybrid main_temporal_hybrid.o temporal_graph.o utils_temporal.o sssp_temporal_hybrid.o $(LDFLAGS)

# Temporal version
temporal: main_temporal.o temporal_graph.o utils_temporal.o sssp_temporal.o
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) -o sssp_temporal main_temporal.o temporal_graph.o utils_temporal.o sssp_temporal.o

# OpenMP version
openmp: main_openmp.o graph.o utils.o sssp_openmp.o
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) -o sssp_openmp main_openmp.o graph.o utils.o sssp_openmp.o

# Hybrid version
hybrid: main_hybrid.o graph.o utils.o sssp_hybrid.o
	$(MPICC) $(CFLAGS) $(OPENMP_FLAGS) $(METIS_FLAGS) -o sssp_hybrid main_hybrid.o graph.o utils.o sssp_hybrid.o $(LDFLAGS)

# Object files - Temporal Hybrid
main_temporal_hybrid.o: main_temporal_hybrid.cpp
	$(MPICC) $(CFLAGS) $(OPENMP_FLAGS) $(METIS_FLAGS) -c main_temporal_hybrid.cpp

sssp_temporal_hybrid.o: sssp_temporal_hybrid.cpp sssp_temporal_hybrid.h types.h temporal_graph.h
	$(MPICC) $(CFLAGS) $(OPENMP_FLAGS) $(METIS_FLAGS) -c sssp_temporal_hybrid.cpp

# Object files - Temporal
main_temporal.o: main_temporal.cpp
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) -c main_temporal.cpp

temporal_graph.o: temporal_graph.cpp temporal_graph.h types.h
	$(CC) $(CFLAGS) -c temporal_graph.cpp

utils_temporal.o: utils_temporal.cpp utils_temporal.h types.h temporal_graph.h
	$(CC) $(CFLAGS) -c utils_temporal.cpp

sssp_temporal.o: sssp_temporal.cpp sssp_temporal.h types.h temporal_graph.h
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) -c sssp_temporal.cpp

# Object files - Original
main_openmp.o: main_openmp.cpp
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) -c main_openmp.cpp

main_hybrid.o: main_hybrid.cpp
	$(MPICC) $(CFLAGS) $(OPENMP_FLAGS) $(METIS_FLAGS) -c main_hybrid.cpp

graph.o: graph.cpp graph.h types.h
	$(CC) $(CFLAGS) -c graph.cpp

utils.o: utils.cpp utils.h types.h graph.h
	$(CC) $(CFLAGS) -c utils.cpp

sssp_openmp.o: sssp_openmp.cpp sssp_openmp.h types.h graph.h
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) -c sssp_openmp.cpp

sssp_hybrid.o: sssp_hybrid.cpp sssp_hybrid.h types.h graph.h
	$(MPICC) $(CFLAGS) $(OPENMP_FLAGS) $(METIS_FLAGS) -c sssp_hybrid.cpp

# Utility targets
clean:
	rm -f *.o sssp_openmp sssp_hybrid sssp_temporal sssp_temporal_hybrid

# Generate test graphs
generate_test_graph:
	./sssp_temporal 15 30 4 0

# Generate visualizations from DOT files
visualize:
	for dot_file in *.dot; do \
		dot -Tpng $$dot_file -o $${dot_file%.dot}.png; \
	done

# Generate temporal animation
animate:
	./generate_animation.sh

# Run all versions with the same input
run_all: openmp hybrid temporal temporal_hybrid
	@echo "Running OpenMP version..."
	./sssp_openmp 15 25 4 0
	@echo "\nRunning Hybrid version..."
	mpirun -np 4 ./sssp_hybrid 15 25 1 0
	@echo "\nRunning Temporal version..."
	./sssp_temporal 15 25 4 0
	@echo "\nRunning Temporal Hybrid version..."
	mpirun -np 4 ./sssp_temporal_hybrid 15 25 1 0

# Run a performance comparison
benchmark: openmp hybrid temporal temporal_hybrid
	@echo "Benchmarking all implementations..."
	./run_benchmarks.sh

.PHONY: clean visualize generate_test_graph animate run_all benchmark
