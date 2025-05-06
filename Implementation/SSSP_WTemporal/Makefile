CC = g++
MPICC = mpic++
CFLAGS = -std=c++11 -Wall -O3
OPENMP_FLAGS = -fopenmp
METIS_INCLUDE = -DUSE_METIS -I/usr/include/metis
METIS_LIB = -lmetis
LDFLAGS = 

# OpenMP version
openmp: main_openmp.o graph.o utils.o sssp_openmp.o
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) -o sssp_openmp main_openmp.o graph.o utils.o sssp_openmp.o

# Hybrid version
hybrid: main_hybrid.o graph.o utils.o sssp_hybrid.o
	$(MPICC) $(CFLAGS) $(OPENMP_FLAGS) -o sssp_hybrid main_hybrid.o graph.o utils.o sssp_hybrid.o $(METIS_LIB) $(LDFLAGS)

# Object files
main_openmp.o: main_openmp.cpp
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) -c main_openmp.cpp

main_hybrid.o: main_hybrid.cpp
	$(MPICC) $(CFLAGS) $(OPENMP_FLAGS) $(METIS_INCLUDE) -c main_hybrid.cpp

graph.o: graph.cpp graph.h types.h
	$(CC) $(CFLAGS) -c graph.cpp

utils.o: utils.cpp utils.h types.h graph.h
	$(CC) $(CFLAGS) -c utils.cpp

sssp_openmp.o: sssp_openmp.cpp sssp_openmp.h types.h graph.h
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) -c sssp_openmp.cpp

sssp_hybrid.o: sssp_hybrid.cpp sssp_hybrid.h types.h graph.h
	$(MPICC) $(CFLAGS) $(OPENMP_FLAGS) $(METIS_INCLUDE) -c sssp_hybrid.cpp

# Utility targets
clean:
	rm -f *.o sssp_openmp sssp_hybrid

# Generate test graphs
generate_test_graph:
	./sssp_openmp 15 30 4 0

# Generate visualizations from DOT files
visualize:
	for dot_file in *.dot; do \
		dot -Tpng $$dot_file -o $${dot_file%.dot}.png; \
	done

.PHONY: clean visualize generate_test_graph
