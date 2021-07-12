all: query_binary_index

OBJ = SIMD-BP/bitpacking.o SIMD-BP/simdbitpacking.o

query_binary_index: query_binary_index.o BinaryIndex.o PostingList.o $(OBJ)
	g++ -o $@ $^

%.o : %.cc
	g++ -std=c++11 -O3 -c $<

clean:
	rm query_binary_index *.o
