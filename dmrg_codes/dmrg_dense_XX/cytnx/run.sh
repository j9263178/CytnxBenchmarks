export CYTNX_INC=$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_include__)\")")
export CYTNX_LIB=$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_lib__)\")")/libcytnx.a
export CYTNX_LINK="$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_linkflags__)\")")"
export CYTNX_CXXFLAGS="$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_flags__)\")")"

# g++ -I${CYTNX_INC} ${CYTNX_CXXFLAGS} dmrg_dense_benchmark.cpp ${CYTNX_LIB} ${CYTNX_LINK} -g -o dmrg_dense_benchmark -lrt
g++ -I${CYTNX_INC} ${CYTNX_CXXFLAGS} dmrg_dense_benchmark.cpp ${CYTNX_LIB} ${CYTNX_LINK} -g -o dmrg_dense_benchmark -lrt
./dmrg_dense_benchmark>>dense_hptt_out.txt
# gdb ./dmrg_dense_benchmark
# ./dmrg_dense_benchmark