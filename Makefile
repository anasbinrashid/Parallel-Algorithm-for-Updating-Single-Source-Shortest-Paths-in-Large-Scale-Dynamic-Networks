CXX = mpic++
CXXFLAGS = -std=c++14 -O3 -fopenmp -Wall
LDFLAGS = -lmetis

SRCS = graph.cpp sequential_sssp.cpp metis_wrapper.cpp parallel_sssp.cpp main.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = sssp_dynamic

GEN_SRC = graph_generator.cpp
GEN_OBJ = $(GEN_SRC:.cpp=.o)
GEN_TARGET = graph_generator

all: $(TARGET) $(GEN_TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(GEN_TARGET): $(GEN_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(GEN_OBJ) $(TARGET) $(GEN_TARGET)

test: $(TARGET)
	mpirun -np 2 ./$(TARGET) -f bio-CE-HT.edges -t 2 -c 1000 -i 0.7 -a 2

generate: $(GEN_TARGET)
	./$(GEN_TARGET) 10000 50000 synthetic_graph.edges

big_test: $(TARGET) generate
	mpirun -np 2 ./$(TARGET) -f synthetic_graph.edges -t 2 -c 1000 -i 0.7 -a 2

.PHONY: all clean test generate big_test
