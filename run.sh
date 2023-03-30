# export MKL_NUM_THREADS=1
# export OMP_NUM_THREADS=1
# # ./Hpsi_test.e --benchmark_repetitions=10 --benchmark_min_warmup_time=1 --benchmark_format=csv
# ./dmrg_dense_test.e --benchmark_repetitions=1 --benchmark_time_unit=s --benchmark_format=csv
# # ./dmrg_U1_test.e --benchmark_repetitions=1 --benchmark_time_unit=s

# export MKL_NUM_THREADS=8
# export OMP_NUM_THREADS=1
# ./dmrg_dense_test.e --benchmark_repetitions=1 --benchmark_time_unit=s --benchmark_format=csv

# export MKL_NUM_THREADS=1
# export OMP_NUM_THREADS=1
# ./dmrg_U1_test.e --benchmark_repetitions=1 --benchmark_time_unit=s --benchmark_format=csv

# export MKL_NUM_THREADS=8
# export OMP_NUM_THREADS=1
# ./dmrg_U1_test.e --benchmark_repetitions=6 --benchmark_time_unit=s --benchmark_format=csv


export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
./Hpsi_test.e --benchmark_repetitions=1 --benchmark_format=csv --benchmark_time_unit=s

# export MKL_NUM_THREADS=8
# export OMP_NUM_THREADS=1
# ./Hpsi_test.e --benchmark_repetitions=3 --benchmark_format=csv --benchmark_time_unit=s