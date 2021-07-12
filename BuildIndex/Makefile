APP1 = buildFromIndri
SRC1 = $(APP1).cc
OBJ = SIMD-BP/bitpacking.o SIMD-BP/simdbitpacking.o

## change this according to your installation of indri
prefix = /home/shao158/Documents/indri

exec_prefix = ${prefix}
libdir = ${exec_prefix}/lib
includedir = ${prefix}/include
INCPATH = -I$(includedir)
LIBPATH = -L$(libdir)
CXXFLAGS = -std=c++11 -DPACKAGE_NAME=\"Indri\" -DPACKAGE_TARNAME=\"indri\" -DPACKAGE_VERSION=\"5.11\" -DPACKAGE_STRING=\"Indri\ 5.11\" -DPACKAGE_BUGREPORT=\"project@lemurproject.org\" -DINDRI_STANDALONE=1 -DHAVE_LIBM=1 -DHAVE_LIBPTHREAD=1 -DHAVE_LIBZ=1 -DHAVE_NAMESPACES=/\*\*/ -DISNAN_IN_NAMESPACE_STD=/\*\*/ -DSTDC_HEADERS=1 -DHAVE_SYS_TYPES_H=1 -DHAVE_SYS_STAT_H=1 -DHAVE_STDLIB_H=1 -DHAVE_STRING_H=1 -DHAVE_MEMORY_H=1 -DHAVE_STRINGS_H=1 -DHAVE_INTTYPES_H=1 -DHAVE_STDINT_H=1 -DHAVE_UNISTD_H=1 -DHAVE_FSEEKO=1 -DHAVE_EXT_ATOMICITY_H=1 -DP_NEEDS_GNU_CXX_NAMESPACE=1 -DHAVE_MKSTEMP=1 -DHAVE_MKSTEMPS=1 -DNDEBUG=1 -g -O3 $(INCPATH)
CPPLDFLAGS = -lindri -lz -lpthread -lm 

all: buildFromIndri

buildFromIndri: buildFromIndri.o $(OBJ) 
	$(CXX) -o $@ $< $(OBJ) $(LIBPATH) $(CPPLDFLAGS)

buildFromIndri.o:
	$(CXX) $(CXXFLAGS) -c $(SRC1)

clean:
	rm -f $(APP1) *.o
