CXX:=g++
CXXFLAGS:= -std=c++20


all: CXXFLAGS+= -O1 -Wall -fopenmp
all: test speed_test

single: CXXFLAGS+= -Ofast -march=native
single: speed_test

release: CXXFLAGS+= -Ofast -fopenmp -march=native
release: test speed_test

profile: CXXFLAGS+= -O1 -pg
profile: speed_test

debug: CXXFLAGS+= -Og -g
debug: test speed_test


test: tests/test.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

speed_test:  tests/speed_test.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	-rm *test
	clear

	
