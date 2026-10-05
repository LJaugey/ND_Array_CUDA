CXX:=nvcc
CXXFLAGS:= --std=c++20


all: CXXFLAGS+= -O1 -Xcompiler -fopenmp
all: test speed_test

release: CXXFLAGS+= -O3 --use_fast_math -extra-device-vectorization -Xcompiler -fopenmp -Xcompiler -march=native
release: test speed_test

profile: CXXFLAGS+= -O1 -pg
profile: speed_test

debug: CXXFLAGS+= -O0 -g
debug: test speed_test


speed_test:  tests/speed_test.cu
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	-rm *test
	clear

	
