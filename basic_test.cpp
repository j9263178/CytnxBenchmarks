#include <benchmark/benchmark.h>
#include <array>
#include <cytnx.hpp>
#include <itensor/all.h>
 
// Cytnx test
static void BM_Cytnx_declare(benchmark::State& state)
{
    for (auto _: state) {
	cytnx::Tensor A;
    }
}
BENCHMARK(BM_Cytnx_declare);

static void BM_Cytnx_reshape(benchmark::State& state)
{
    for (auto _: state) {
	    auto A = cytnx::arange(24);
	    auto B = A.reshape(2, 3, 4);
    }
}
BENCHMARK(BM_Cytnx_reshape);

static void BM_Cytnx_contract(benchmark::State& state)
{
	for (auto _: state) {
		auto A = cytnx::UniTensor(cytnx::ones({3, 3, 3}));
		A.set_labels(std::vector<long int>{1l, 2l, 3l});
		auto B = cytnx::UniTensor(cytnx::ones({3, 3, 3, 3}));
		B.set_labels(std::vector<long int>{2l, 3l, 4l, 5l});
		auto C = cytnx::Contract(A, B);
    }
}
BENCHMARK(BM_Cytnx_contract);

// itensor test
static void BM_itensor_declare(benchmark::State& state)
{
    for (auto _: state) {
	    auto i = itensor::Index(4, "index i");
	    auto j = itensor::Index(6, "index j");
	    auto T = itensor::ITensor(i,j);
	    T.set(i=3, j = 2, 3.14159);
    }
}
BENCHMARK(BM_itensor_declare);
 
BENCHMARK_MAIN();
