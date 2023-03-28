#install root
# CYTNX_ROOT:=/home/petjelinux/Cytnx_lib
# ITENSOR_ROOT:=/home/petjelinux/itensor
# BENCHMARK_ROOT:=/home/petjelinux/benchmark
CYTNX_ROOT:=/home/j9263178/cytnx_bk_new
ITENSOR_ROOT:=/home/j9263178/itensor
BENCHMARK_ROOT:=/home/j9263178/benchmark

CC:=g++

#cytnx flags
# CYTNX_INC := $(shell python -c "exec(\"import sys\nsys.path.append(\'$(CYTNX_ROOT)\')\nimport cytnx\nprint(cytnx.__cpp_include__)\")")
# CYTNX_LDFLAGS := $(shell python -c "exec(\"import sys\nsys.path.append(\'$(CYTNX_ROOT)\')\nimport cytnx\nprint(cytnx.__cpp_linkflags__)\")")
# CYTNX_LIB := $(shell python -c "exec(\"import sys\nsys.path.append(\'$(CYTNX_ROOT)\')\nimport cytnx\nprint(cytnx.__cpp_lib__)\")")/libcytnx.a
# CYTNX_CXXFLAGS := $(shell python -c "exec(\"import sys\nsys.path.append(\'$(CYTNX_ROOT)\')\nimport cytnx\nprint(cytnx.__cpp_flags__)\")")

#cytnx flags
# CYTNX_INC:=/home/petjelinux/Cytnx_lib/include
# CYTNX_LIB:=/home/petjelinux/Cytnx_lib/lib64/libcytnx.a
# CYTNX_LINK:=/home/petjelinux/anaconda3/envs/cytnx/lib/libmkl_intel_ilp64.so /home/petjelinux/anaconda3/envs/cytnx/lib/libmkl_intel_thread.so /home/petjelinux/anaconda3/envs/cytnx/lib/libmkl_core.so /home/petjelinux/anaconda3/envs/cytnx/lib/libiomp5.so -lpthread -lm -ldl -lpthread -lm -ldl -Wl,-rpath,/home/petjelinux/anaconda3/envs/cytnx/lib /home/petjelinux/Cytnx_lib/hptt/lib/libhptt.a
# CYTNX_CXXFLAGS:=-I/home/petjelinux/anaconda3/envs/cytnx/include -fvisibility-inlines-hidden -std=c++17 -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem /home/petjelinux/anaconda3/envs/cytnx/include -Wformat=0 -Wno-c++11-narrowing -w -fopenmp -DUNI_HPTT
CYTNX_INC:=/home/j9263178/cytnx_bk_new/include
CYTNX_LIB:=/home/j9263178/cytnx_bk_new/lib64/libcytnx.a
CYTNX_LINK:=/home/j9263178/anaconda3/envs/cytnx4/lib/libmkl_intel_lp64.so /home/j9263178/anaconda3/envs/cytnx4/lib/libmkl_intel_thread.so /home/j9263178/anaconda3/envs/cytnx4/lib/libmkl_core.so /home/j9263178/anaconda3/envs/cytnx4/lib/libiomp5.so -lpthread -lm -ldl -lpthread -lm -ldl -Wl,-rpath,/home/j9263178/anaconda3/envs/cytnx4/lib /home/j9263178/cytnx_bk_new/hptt/lib/libhptt.a
CYTNX_CXXFLAGS:=-I/home/j9263178/anaconda3/envs/cytnx4/include -fvisibility-inlines-hidden -std=c++17 -fmessage-length=0 -fpermissive -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem /home/j9263178/anaconda3/envs/cytnx4/include -Wformat=0 -Wno-c++11-narrowing -w -fopenmp -DUNI_HPTT

#itensor flags
ITENSOR_CXXFLAGS:=-std=c++17
ITENSOR_HEADER:=$(ITENSOR_ROOT)
ITENSOR_LIBDIR:=$(ITENSOR_ROOT)/lib

#benchmark flags
BENCHMARK_HEADER:=$(BENCHMARK_ROOT)/include
BENCHMARK_LIB:=$(BENCHMARK_ROOT)/build/lib
BENCHMARK_SRC:=$(BENCHMARK_ROOT)/build/src

HDF5_PREFIX=/home/j9263178/anaconda3/envs/cytnx4

# TARGETS:=Hpsi_test.e
# TARGETS:=useitensor.e
# TARGETS:=dmrg_U1_test.e
TARGETS:=dmrg_dense_test.e
$(TARGETS): %.e:%.cpp
#	$(CC) $(CYTNX_CXXFLAGS) -I$(BENCHMARK_HEADER) -I${CYTNX_INC}  $< -L$(BENCHMARK_SRC) ${CYTNX_LIB} ${CYTNX_LINK} -I$(ITENSOR_ROOT) -L$(HDF5_PREFIX)/lib -lhdf5 -lhdf5_hl -g -DNDEBUG -Wall -Wno-unknown-pragmas -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_rt -lmkl_core -liomp5 -L$(ITENSOR_LIBDIR) -litensor -lrt -lbenchmark -o $@
#	$(CC) $(CYTNX_CXXFLAGS) -I$(BENCHMARK_HEADER) $< -L$(BENCHMARK_SRC) ${CYTNX_LINK} -I$(ITENSOR_ROOT) -L$(HDF5_PREFIX)/lib -lhdf5 -lhdf5_hl -g -DNDEBUG -Wall -Wno-unknown-pragmas -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_rt -lmkl_core -liomp5 -L$(ITENSOR_LIBDIR) -litensor -lrt -lbenchmark -o $@
	$(CC) $(CYTNX_CXXFLAGS) -I$(BENCHMARK_HEADER) -I${CYTNX_INC}  $< -L$(BENCHMARK_SRC) ${CYTNX_LIB} ${CYTNX_LINK} -I$(ITENSOR_ROOT) -lgcov -coverage -lrt -O2 -DNDEBUG -Wall -L$(ITENSOR_LIBDIR) -litensor -lrt -lbenchmark -o $@
#clean
.phony: clean

clean:
	rm -f *.o *.e *.gcno *.gcda



