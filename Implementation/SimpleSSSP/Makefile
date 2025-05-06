CXX = mpic++
CXXFLAGS = -std=c++11 -O3 -fopenmp -Wall
LDFLAGS = -lmetis

SRCS = graph.cpp metis_wrapper.cpp sssp.cpp main.cpp test_sssp.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = sssp_dynamic

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) *.dot

.PHONY: all clean
