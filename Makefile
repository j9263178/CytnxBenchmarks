#install root
CYTNX_ROOT:=/home/petjelinux/Cytnx_lib
ITENSOR_ROOT:=/home/petjelinux/itensor
BENCHMARK_ROOT:=/home/petjelinux/benchmark

CC:=g++

#cytnx flags
# CYTNX_INC := $(shell python -c "exec(\"import sys\nsys.path.append(\'$(CYTNX_ROOT)\')\nimport cytnx\nprint(cytnx.__cpp_include__)\")")
# CYTNX_LDFLAGS := $(shell python -c "exec(\"import sys\nsys.path.append(\'$(CYTNX_ROOT)\')\nimport cytnx\nprint(cytnx.__cpp_linkflags__)\")")
# CYTNX_LIB := $(shell python -c "exec(\"import sys\nsys.path.append(\'$(CYTNX_ROOT)\')\nimport cytnx\nprint(cytnx.__cpp_lib__)\")")/libcytnx.a
# CYTNX_CXXFLAGS := $(shell python -c "exec(\"import sys\nsys.path.append(\'$(CYTNX_ROOT)\')\nimport cytnx\nprint(cytnx.__cpp_flags__)\")")

#cytnx flags
CYTNX_INC:=/home/petjelinux/Cytnx_lib/include
CYTNX_LIB:=/home/petjelinux/Cytnx_lib/lib64/libcytnx.a
CYTNX_LINK:=/home/petjelinux/anaconda3/envs/cytnx/lib/libmkl_intel_ilp64.so /home/petjelinux/anaconda3/envs/cytnx/lib/libmkl_intel_thread.so /home/petjelinux/anaconda3/envs/cytnx/lib/libmkl_core.so /home/petjelinux/anaconda3/envs/cytnx/lib/libiomp5.so -lpthread -lm -ldl -lpthread -lm -ldl -Wl,-rpath,/home/petjelinux/anaconda3/envs/cytnx/lib /home/petjelinux/Cytnx_lib/hptt/lib/libhptt.a
CYTNX_CXXFLAGS:=-I/home/petjelinux/anaconda3/envs/cytnx/include -fvisibility-inlines-hidden -std=c++17 -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe -isystem /home/petjelinux/anaconda3/envs/cytnx/include -Wformat=0 -Wno-c++11-narrowing -w -fopenmp -DUNI_HPTT

#itensor flags
ITENSOR_CXXFLAGS:=-std=c++17
ITENSOR_HEADER:=$(ITENSOR_ROOT)
ITENSOR_LIBDIR:=$(ITENSOR_ROOT)/lib

#benchmark flags
BENCHMARK_HEADER:=$(BENCHMARK_ROOT)/include
BENCHMARK_LIB:=$(BENCHMARK_ROOT)/build/lib
BENCHMARK_SRC:=$(BENCHMARK_ROOT)/build/src


TARGETS:=Hpsi_test.e
$(TARGETS): %.e:%.cpp
	$(CC) $(CYTNX_CXXFLAGS) -I$(BENCHMARK_HEADER) -I${CYTNX_INC}  $< -L$(BENCHMARK_SRC) ${CYTNX_LIB} ${CYTNX_LINK} -I$(ITENSOR_ROOT) -O2 -DNDEBUG -Wall -Wno-unknown-pragmas -L$(ITENSOR_LIBDIR) -litensor -lrt -lbenchmark -o $@

#clean
.phony: clean

clean:
	rm -f *.o *.e *.gcno *.gcda


