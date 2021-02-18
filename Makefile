
SIMPLEDATA := arch/sse/static_data.cc
USE_LIKWID := false
CXXFLAGS_SVE_O1 := -O1
CXXFLAGS_SVE_O3 := -O3
CXXFLAGS_SVE_NOSCHED_GCC := -O3 -fno-schedule-insns -fno-schedule-insns2

#OMP:=-std=c++11 -DSVM
ifeq (${USE_LIKWID},true)
	OMP:=-std=c++11 -DSVM -DOMP -fopenmp  -I${LIKWID_INCDIR} -DLIKWID_PERFMON
else
	OMP:=-std=c++11 -DSVM -DOMP -fopenmp
endif

#OMP:=-std=c++11  -O3 -fno-schedule-insns -fno-schedule-insns2 -fno-sched-interblock -DSVM
#OMP:=-std=c++11 -O3

#CXX       := mpicxx-openmpi-devel-clang40
#CXX       := mpiicpc
#CXX       := g++-7
#CXX        := dpcpp
CXX       := g++
#CXX       := mpicxx
#CXX       := clang++-mp-6.0
#CXXCL     := clang++-mp-7.0
#CXXCL     := ${HOME}/QCD/build/bin/clang++
CXXCL     := dpcpp
CXXFLAGSCL:=
LDFLAGSCL:=
CXXFLAGS  := $(OMP)

AVX512_DATA   := arch/avx512/static_data.cc
AVX2_DATA     := arch/avx/static_data_gauge.cc arch/avx/static_data_fermion.cc
AVX_DATA      := arch/avx/static_data_gauge.cc arch/avx/static_data_fermion.cc
SSE_DATA      := arch/sse/static_data.cc
RRII_DATA     := arch/gen64/static_data.cc
RIRI_DATA     := arch/gen64/static_data.cc
RRII_DATA_SVE := arch/sve/rrii/static_data.cc
RIRI_DATA_SVE := arch/sve/riri/static_data.cc


#############################################
# Intel
#############################################
#AVX512_CXXFLAGS  := -DAVX512 -xcore-avx512 $(OMP)
#AVX2_CXXFLAGS    := -DAVX2  -march=core-avx2 -xcore-avx2 $(OMP)
#AVX_CXXFLAGS     := -DAVX1  -mavx -xavx $(OMP)
#SSE_CXXFLAGS     := -DSSE4  -msse4.2 -xsse4.2  $(OMP)
#RRII_CXXFLAGS     := -DRRII  -mavx2 -mfma  $(OMP)

#############################################
# CLANG
#############################################
AVX512_CXXFLAGS  := -DAVX512 -march=skylake-avx512 -O3 $(OMP)
#AVX512_CXXFLAGS  := -DAVX512 -mavx512f -mavx512pf -mavx512er -mavx512cd -O3 $(OMP)
AVX2_CXXFLAGS    := -DAVX2  -mavx2 -mfma $(OMP)
AVX_CXXFLAGS     := -DAVX1  -mavx $(OMP)
SSE_CXXFLAGS     := -DSSE4  -msse4.2  $(OMP)
RRII_CXXFLAGS     := -DRRII  -mavx2 -mfma  $(OMP) -DGEN_SIMD_WIDTH=64
RIRI_CXXFLAGS     := -DRIRI  -mavx2 -mfma  $(OMP) -DGEN_SIMD_WIDTH=64 -g
RRII_CXXFLAGSXX   := -DRRII  -march=knl  $(OMP) -DGEN_SIMD_WIDTH=128


RRII_CXXFLAGS_SVE_GCC               := -DRRII  -march=armv8-a+sve -msve-vector-bits=512  $(OMP) -DGEN_SIMD_WIDTH=64 -DSVE
RRII_CXXFLAGS_SVE_INTRIN_GCC        := -DRRII  -march=armv8-a+sve -msve-vector-bits=512  $(OMP) -DGEN_SIMD_WIDTH=64 -DINTRIN -DSVE -g
RIRI_CXXFLAGS_SVE_INTRIN_GCC        := -DRIRI  -march=armv8-a+sve -msve-vector-bits=512  $(OMP) -DGEN_SIMD_WIDTH=64 -DINTRIN -DSVE
#RIRI_CXXFLAGS_SVE_INTRIN_GCC        := -march=armv8-a+sve -msve-vector-bits=512  $(OMP) -DGEN_SIMD_WIDTH=64 -DINTRIN -DSVE
#RRII_CXXFLAGS_SVE_INTRIN_ARMCLANG   := -DRRII  -march=armv8-a+sve $(OMP) -DGEN_SIMD_WIDTH=64 -DINTRIN -DSVE
RRII_CXXFLAGS_SVE_INTRIN_ARMCLANG   := -DRRII  -mcpu=a64fx $(OMP) -DGEN_SIMD_WIDTH=64 -DINTRIN -DSVE
RRII_CXXFLAGS_SVE_INTRIN_FCC        := -DRRII  -Nclang -Ofast $(OMP) -DGEN_SIMD_WIDTH=64 -DINTRIN -DSVE
RIRI_CXXFLAGS_SVE_INTRIN_FCC        := -DRIRI  -Nclang -Ofast $(OMP) -DGEN_SIMD_WIDTH=64 -DINTRIN -DSVE
#RRII_CXXFLAGS_SVE_INTRIN_FCC        := -DRRII  -Nclang -O1 $(OMP) -DGEN_SIMD_WIDTH=64 -DINTRIN -DSVE

#############################################
# G++
#############################################
#AVX512_CXXFLAGS  := -DAVX512 -mavx512f -mavx512pf -mavx512er -mavx512cd -O3 $(OMP)
#AVX2_CXXFLAGS    := -DAVX2  -mavx2 -mfma $(OMP)
#AVX_CXXFLAGS     := -DAVX1  -mavx $(OMP)
#SSE_CXXFLAGS     := -DSSE4  -msse4.2  $(OMP)

#Generic options
GENERIC_CXXFLAGS  := -DGEN -O3 -DGEN_SIMD_WIDTH=16 $(OMP)
GENERIC_DATA      := arch/sse/static_data.cc

################################################################################
# NVCC and gpu ; 512 bit vector coalescing
################################################################################

# VOLTA
GPUARCH    := --relocatable-device-code=true -gencode arch=compute_70,code=sm_70

# PASCAL
#GPUARCH    := --relocatable-device-code=true -gencode arch=compute_60,code=sm_60

GPUCC      := nvcc
GPULINK    := nvcc $(GPUARCH)
GPU_CXXFLAGS  := -x cu -DVGPU -DGEN_SIMD_WIDTH=64 -I. -O3 -ccbin g++ -std=c++11 --expt-relaxed-constexpr --expt-extended-lambda $(GPUARCH)  -Xcompiler -fno-strict-aliasing
GPU_LDFLAGS  := -link -ccbin g++
GPU_DATA      := arch/avx512/static_data.cc
################################################################################
LDLIBS    := -lm
ifeq (${USE_LIKWID},true)
	LDFLAGS   := -llikwid -L${LIKWID_LIBDIR}
else
	LDFLAGS   :=
endif

all: bench.avx512 bench.avx2 bench.avx bench.sse bench.gen bench.simple bench.sycl \
	bench.rrii.omp.cpu bench.rrii.sycl.cpu bench.rrii.sycl.cpu.simt  bench.rrii.sycl.gpu bench.rrii.sycl.gpu.simt bench.riri.sycl.gpu.simt

#bench.avx512: bench_static.cc $(AVX512_DATA)  WilsonKernelsHand.h Makefile
#	$(CXX) $(AVX512_CXXFLAGS) bench_static.cc $(AVX512_DATA) $(LDLIBS) $(LDFLAGS) -o bench.avx512

bench.avx2: bench_static.cc $(AVX2_DATA)  WilsonKernelsHand.h Makefile
	$(CXX) $(AVX2_CXXFLAGS) bench_static.cc $(AVX2_DATA) $(LDLIBS) $(LDFLAGS) -o bench.avx2

bench.rrii.omp.cpu: bench_static.cc $(RRII_DATA)  WilsonKernelsHand.h Makefile
	$(CXX) $(RRII_CXXFLAGS) bench_static.cc $(RRII_DATA) $(LDLIBS) $(LDFLAGS) -o bench.rrii.omp.cpu


# AVX512 RRII

bench.rrii.avx512.gccvectors.clang: bench_file.cc  WilsonKernelsHand.h Makefile arch/avx512/rrii/AVX512GCCVectors.h
	clang++ -DRRII $(AVX512_CXXFLAGS) bench_file.cc $(LDLIBS) $(LDFLAGS) -o bench.rrii.avx512.gccvectors.clang

bench.rrii.avx512.intrinsics.clang: bench_file.cc  WilsonKernelsHand.h Makefile arch/avx512/rrii/AVX512_rrii.h
	clang++ -DRRII -DINTRIN $(AVX512_CXXFLAGS) bench_file.cc $(LDLIBS) $(LDFLAGS) -o bench.rrii.avx512.intrinsics.clang

bench.rrii.avx512.gccvectors.gcc: bench_file.cc  WilsonKernelsHand.h Makefile arch/avx512/rrii/AVX512GCCVectors.h
	g++ -DRRII $(AVX512_CXXFLAGS) bench_file.cc $(LDLIBS) $(LDFLAGS) -o bench.rrii.avx512.gccvectors.gcc

# AVX512 RIRI

bench.riri.avx512.intrinsics.clang: bench_file.cc  WilsonKernelsHand.h Makefile arch/avx512/Simd_avx512.h
	clang++ -DRIRI -DINTRIN $(AVX512_CXXFLAGS) bench_file.cc $(LDLIBS) $(LDFLAGS) -o bench.riri.avx512.intrinsics.clang


# SVE RRII
bench.rrii.sve.gccvectors.gcc: bench_file.cc  WilsonKernelsHand.h Makefile arch/sve/rrii/SVE_rrii.h
	$(CXX) $(RRII_CXXFLAGS_SVE_GCC) $(CXXFLAGS_SVE_NOSCHED_GCC) bench_file.cc $(LDLIBS) $(LDFLAGS) -o bench.rrii.sve.gccvectors.gcc

bench.rrii.sve.intrinsics.gcc: bench_file.cc   WilsonKernelsHand.h Makefile arch/sve/rrii/SVE_rrii.h
	#$(CXX) $(RRII_CXXFLAGS_SVE_INTRIN_GCC) $(CXXFLAGS_SVE_O1) bench_file.cc $(LDLIBS) $(LDFLAGS) -o bench.rrii.sve.intrinsics.gcc
	$(CXX) $(RRII_CXXFLAGS_SVE_INTRIN_GCC) $(CXXFLAGS_SVE_O1) -c arch/sve/rrii/rrii_kernel.cpp -o kernel.rrii.sve.intrinsics.gcc.o

bench.rrii.sve.intrinsics.armclang: bench_file.cc  WilsonKernelsHand.h Makefile arch/sve/rrii/SVE_rrii.h
	$(CXX) $(RRII_CXXFLAGS_SVE_INTRIN_ARMCLANG) $(CXXFLAGS_SVE_O3) bench_file.cc $(LDLIBS) $(LDFLAGS) -o bench.rrii.sve.intrinsics.armclang

bench.rrii.sve.intrinsics.fcc: bench_file.cc  WilsonKernelsHand.h Makefile arch/sve/rrii/SVE_rrii.h
	$(CXX) $(RRII_CXXFLAGS_SVE_INTRIN_FCC) bench_file.cc $(LDLIBS) $(LDFLAGS) -o bench.rrii.sve.intrinsics.fcc

#bench.rrii.sve.gccvectors.gcc: bench_static.cc $(RRII_DATA_SVE)  WilsonKernelsHand.h Makefile arch/sve/rrii/SVE_rrii.h
#	$(CXX) $(RRII_CXXFLAGS_SVE_GCC) $(CXXFLAGS_SVE_NOSCHED_GCC) bench_static.cc $(RRII_DATA_SVE) $(LDLIBS) $(LDFLAGS) -o bench.rrii.sve.gccvectors.gcc

#bench.rrii.sve.intrinsics.gcc: bench_static.cc $(RRII_DATA_SVE)  WilsonKernelsHand.h Makefile arch/sve/rrii/SVE_rrii.h
#	$(CXX) $(RRII_CXXFLAGS_SVE_INTRIN_GCC) $(CXXFLAGS_SVE_O1) bench_static.cc $(RRII_DATA_SVE) $(LDLIBS) $(LDFLAGS) -o bench.rrii.sve.intrinsics.gcc

#bench.rrii.sve.intrinsics.armclang: bench_static.cc $(RRII_DATA_SVE)  WilsonKernelsHand.h Makefile arch/sve/rrii/SVE_rrii.h
#	$(CXX) $(RRII_CXXFLAGS_SVE_INTRIN_ARMCLANG) $(CXXFLAGS_SVE_O3) bench_static.cc $(RRII_DATA_SVE) $(LDLIBS) $(LDFLAGS) -o bench.rrii.sve.intrinsics.armclang

#bench.rrii.sve.intrinsics.fcc: bench_static.cc $(RRII_DATA_SVE)  WilsonKernelsHand.h Makefile arch/sve/rrii/SVE_rrii.h
#	$(CXX) $(RRII_CXXFLAGS_SVE_INTRIN_FCC) bench_static.cc $(RRII_DATA_SVE) $(LDLIBS) $(LDFLAGS) -o bench.rrii.sve.intrinsics.fcc


# SVE RIRI
#bench.riri.sve.intrinsics.gcc: bench_static.cc $(RIRI_DATA_SVE)  WilsonKernelsHand.h Makefile arch/sve/riri/wi.h arch/sve/riri/SVE_riri.h
#	$(CXX) $(RIRI_CXXFLAGS_SVE_INTRIN_GCC) $(CXXFLAGS_SVE_O1) bench_static.cc $(RIRI_DATA_SVE) $(LDLIBS) $(LDFLAGS) -o bench.riri.sve.intrinsics.gcc

bench.riri.sve.intrinsics.gcc: bench_file.cc WilsonKernelsHand.h Makefile arch/sve/riri/wi.h arch/sve/riri/SVE_riri.h
	#$(CXX) $(RIRI_CXXFLAGS_SVE_INTRIN_GCC) $(CXXFLAGS_SVE_O1) -fopenmp -c bench_file.cc -o bench_file.o
	#$(CXX) bench_file.o kernel.riri.sve.intrinsics.gcc.o -fopenmp $(LDLIBS) $(LDFLAGS) -o bench.riri.sve.intrinsics.gcc
	#$(CXX) $(RIRI_CXXFLAGS_SVE_INTRIN_GCC) $(CXXFLAGS_SVE_O1) bench_file.cc kernel.riri.sve.intrinsics.o $(LDLIBS) $(LDFLAGS) -o bench.riri.sve.intrinsics.gcc
	$(CXX) $(RIRI_CXXFLAGS_SVE_INTRIN_GCC) $(CXXFLAGS_SVE_O1) -c arch/sve/riri/riri_kernel.cpp -o kernel.riri.sve.intrinsics.gcc.o


	#$(CXX) $(RIRI_CXXFLAGS_SVE_INTRIN_GCC) $(CXXFLAGS_SVE_O1) -nostdinc -isystem /vol0001/apps/oss/spack-v0.15.4/opt/spack/linux-rhel8-a64fx/gcc-8.3.1/gcc-10.1.0-2wrm3no2w6uzicvmvrozdz4xhu6nsj7t/include/c++/10.1.0/tr1 -isystem /vol0001/apps/oss/spack-v0.15.4/opt/spack/linux-rhel8-a64fx/gcc-8.3.1/gcc-10.1.0-2wrm3no2w6uzicvmvrozdz4xhu6nsj7t/include/c++/10.1.0 -c arch/sve/riri/riri_kernel.c -o kernel.riri.sve.intrinsics.gcc.o

bench.riri.sve.intrinsics.fcc: bench_file.cc  WilsonKernelsHand.h Makefile arch/sve/riri/SVE_riri.h
	$(CXX) $(RIRI_CXXFLAGS_SVE_INTRIN_FCC) bench_file.cc $(LDLIBS) $(LDFLAGS) -o bench.riri.sve.intrinsics.fcc



bench.avx: bench_static.cc $(AVX_DATA)  WilsonKernelsHand.h Makefile
	$(CXX) $(AVX_CXXFLAGS) bench_static.cc $(AVX_DATA) $(LDLIBS) $(LDFLAGS) -o bench.avx

bench.sse: bench_static.cc $(SSE_DATA)  WilsonKernelsHand.h Makefile
	$(CXX) $(SSE_CXXFLAGS) bench_static.cc $(SSE_DATA) $(LDLIBS) $(LDFLAGS) -o bench.sse

bench.gen: bench_static.cc $(GENERIC_DATA)  WilsonKernelsHand.h Makefile
	$(CXX) $(GENERIC_CXXFLAGS) bench_static.cc $(GENERIC_DATA) $(LDLIBS) $(LDFLAGS) -o bench.gen

#	nvcc -x cu -DVGPU -DGEN_SIMD_WIDTH=64 -I. -O3 -ccbin g++ -std=c++11 -Xcompiler -Wno-deprecated-gpu-targets --expt-relaxed-constexpr --expt-extended-lambda --relocatable-device-code=true -gencode arch=compute_60,code=sm_60  -Xcompiler -fno-strict-aliasing -c bench_static.cc -o bench.gpu.o
bench.gpu: bench_static.cc $(GPU_DATA)  WilsonKernelsHand.h Makefile
	$(GPUCC) $(GPU_CXXFLAGS) -c bench_static.cc -o bench.gpu.o
	$(GPUCC) $(GPU_CXXFLAGS) -c $(GPU_DATA) -o data.gpu.o
	$(GPULINK) $(GPU_LDFLAGS) bench.gpu.o data.gpu.o -o bench.gpu $(LDLIBS) $(LDFLAGS)

bench.simple: bench_simple.cc $(SIMPLEDATA) dslash_simple.h Makefile
	$(CXX) $(CXXFLAGS) bench_simple.cc $(SIMPLEDATA) -I/usr/local/Cellar/boost/1.68.0_1/include -o bench.simple

bench.sycl: bench_sycl.cc $(SIMPLEDATA) dslash_simple.h Makefile
	$(CXXCL) -O3 -std=c++17 $(CXXFLAGSCL) bench_sycl.cc $(SIMPLEDATA) $(LDLIBS) $(LDFLAGS) $(LDFLAGSCL) -o bench.sycl

######################
# Build a test from triSYCL distro to check compiler working
######################
#
#parallel_vector_add:
#	clang++-mp-7.0  -std=c++17 -I/Users/ayamaguc/Grid/triSYCL-master/include parallel_vector_add.cpp  -I/usr/local/Cellar/boost/1.68.0_1/include
#

clean:
	rm -f  bench.avx512 bench.avx2 bench.avx bench.sse bench.gen  bench.simple TableGenerate bench.gpu bench.sycl bench.rrii* bench.riri*
	rm -rf  *.dSYM*
	rm -f  *~
