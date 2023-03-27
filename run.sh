export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
# ./Hpsi_test.e --benchmark_repetitions=10 --benchmark_min_warmup_time=1 --benchmark_format=csv
./dmrg_U1_test.e --benchmark_repetitions=1
