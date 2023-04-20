# make -B
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
# ./Hpsi_test.e --benchmark_repetitions=10 --benchmark_min_warmup_time=1 --benchmark_format=csv
# ./Hpsi_test.e --benchmark_repetitions=10 --benchmark_min_warmup_time=1
# ./Hpsi_test.e --benchmark_repetitions=10
# export LD_PRELOAD=/home/petjelinux/anaconda3/envs/cytnx/lib/libtbbmalloc_proxy.so.2
./Hpsi_test.e
# ./dmrg_dense_test.e --benchmark_repetitions=1 --benchmark_time_unit=s
# ./dmrg_dense_test.e
# export LD_PRELOAD=/home/petjelinux/anaconda3/envs/cytnx/lib/libtbbmalloc_proxy.so.2
# ./dmrg_U1_test.e --benchmark_repetitions=1 --benchmark_time_unit=s
# ./dmrg_U1_test.e