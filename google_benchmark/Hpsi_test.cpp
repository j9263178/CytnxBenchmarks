#include <benchmark/benchmark.h>
#include <array>
#include <cytnx.hpp>
#include <itensor/all.h>

using namespace cytnx;

// Cytnx dense test
// static void Cytnx_declare_dense(benchmark::State& state)
// {
//     for (auto _: state) {
// 	cytnx::UniTensor psi = cytnx::UniTensor(cytnx::zeros({100,2,2,100}));
//     }
// }
// BENCHMARK(Cytnx_declare_dense);


static void Cytnx_Hpsi_dense_D64(benchmark::State& state)
{	
	cytnx_int64 D = 64;
	cytnx::UniTensor L = cytnx::UniTensor(cytnx::zeros({4,D,D}));
	cytnx::UniTensor R = cytnx::UniTensor(cytnx::zeros({4,D,D}));
	cytnx::UniTensor M1 = cytnx::UniTensor(cytnx::zeros({4,4,2,2}));
	cytnx::UniTensor M2 = cytnx::UniTensor(cytnx::zeros({4,4,2,2}));
	cytnx::UniTensor psi = cytnx::UniTensor(cytnx::zeros({D,2,2,D}));

	//// relabel have bugs!
	// auto L_ = L.relabels({-5,-1,0});
	// auto R_ = R.relabels({-7,-4,3});
	// auto M1_ = M1.relabels({-5,-6,-2,1});
	// auto M2_ = M2.relabels({-6,-7,-3,2});
	// auto psi_ = psi.relabels({-1,-2,-3,-4});

	auto L_ = L.set_labels({-5,-1,0});
	auto R_ = R.set_labels({-7,-4,3});
	auto M1_ = M1.set_labels({-5,-6,-2,1});
	auto M2_ = M2.set_labels({-6,-7,-3,2});
	auto psi_ = psi.set_labels({-1,-2,-3,-4});

	for (auto _: state) {
		auto out = L_.contract(M1_.contract(M2_.contract(psi_.contract(R_, false, false), false, false), false, false), false, false);
    }
}
BENCHMARK(Cytnx_Hpsi_dense_D64);


static void Cytnx_Hpsi_dense_D100(benchmark::State& state)
{
	cytnx_int64 D = 100;
	cytnx::UniTensor L = cytnx::UniTensor(cytnx::zeros({4,D,D}));
	cytnx::UniTensor R = cytnx::UniTensor(cytnx::zeros({4,D,D}));
	cytnx::UniTensor M1 = cytnx::UniTensor(cytnx::zeros({4,4,2,2}));
	cytnx::UniTensor M2 = cytnx::UniTensor(cytnx::zeros({4,4,2,2}));
	cytnx::UniTensor psi = cytnx::UniTensor(cytnx::zeros({D,2,2,D}));

	auto L_ = L.set_labels({-5,-1,0});
	auto R_ = R.set_labels({-7,-4,3});
	auto M1_ = M1.set_labels({-5,-6,-2,1});
	auto M2_ = M2.set_labels({-6,-7,-3,2});
	auto psi_ = psi.set_labels({-1,-2,-3,-4});

	for (auto _: state) {
		auto out = L_.contract(M1_.contract(M2_.contract(psi_.contract(R_, false, false), false, false), false, false), false, false);
    }
}
BENCHMARK(Cytnx_Hpsi_dense_D100);


static void Cytnx_Hpsi_dense_D200(benchmark::State& state)
{
	cytnx_int64 D = 200;
	cytnx::UniTensor L = cytnx::UniTensor(cytnx::zeros({4,D,D}));
	cytnx::UniTensor R = cytnx::UniTensor(cytnx::zeros({4,D,D}));
	cytnx::UniTensor M1 = cytnx::UniTensor(cytnx::zeros({4,4,2,2}));
	cytnx::UniTensor M2 = cytnx::UniTensor(cytnx::zeros({4,4,2,2}));
	cytnx::UniTensor psi = cytnx::UniTensor(cytnx::zeros({D,2,2,D}));

	auto L_ = L.set_labels({-5,-1,0});
	auto R_ = R.set_labels({-7,-4,3});
	auto M1_ = M1.set_labels({-5,-6,-2,1});
	auto M2_ = M2.set_labels({-6,-7,-3,2});
	auto psi_ = psi.set_labels({-1,-2,-3,-4});

	for (auto _: state) {
		auto out = L_.contract(M1_.contract(M2_.contract(psi_.contract(R_, false, false), false, false), false, false), false, false);
    }
}
BENCHMARK(Cytnx_Hpsi_dense_D200);



static void Cytnx_Hpsi_dense_D300(benchmark::State& state)
{
	cytnx_int64 D = 300;
	cytnx::UniTensor L = cytnx::UniTensor(cytnx::zeros({4,D,D}));
	cytnx::UniTensor R = cytnx::UniTensor(cytnx::zeros({4,D,D}));
	cytnx::UniTensor M1 = cytnx::UniTensor(cytnx::zeros({4,4,2,2}));
	cytnx::UniTensor M2 = cytnx::UniTensor(cytnx::zeros({4,4,2,2}));
	cytnx::UniTensor psi = cytnx::UniTensor(cytnx::zeros({D,2,2,D}));

	auto L_ = L.set_labels({-5,-1,0});
	auto R_ = R.set_labels({-7,-4,3});
	auto M1_ = M1.set_labels({-5,-6,-2,1});
	auto M2_ = M2.set_labels({-6,-7,-3,2});
	auto psi_ = psi.set_labels({-1,-2,-3,-4});

	for (auto _: state) {
		auto out = L_.contract(M1_.contract(M2_.contract(psi_.contract(R_, false, false), false, false), false, false), false, false);
    }
}
BENCHMARK(Cytnx_Hpsi_dense_D300);


static void Cytnx_Hpsi_dense_D400(benchmark::State& state)
{
	cytnx_int64 D = 400;
	cytnx::UniTensor L = cytnx::UniTensor(cytnx::zeros({4,D,D}));
	cytnx::UniTensor R = cytnx::UniTensor(cytnx::zeros({4,D,D}));
	cytnx::UniTensor M1 = cytnx::UniTensor(cytnx::zeros({4,4,2,2}));
	cytnx::UniTensor M2 = cytnx::UniTensor(cytnx::zeros({4,4,2,2}));
	cytnx::UniTensor psi = cytnx::UniTensor(cytnx::zeros({D,2,2,D}));

	auto L_ = L.set_labels({-5,-1,0});
	auto R_ = R.set_labels({-7,-4,3});
	auto M1_ = M1.set_labels({-5,-6,-2,1});
	auto M2_ = M2.set_labels({-6,-7,-3,2});
	auto psi_ = psi.set_labels({-1,-2,-3,-4});

	for (auto _: state) {
		auto out = L_.contract(M1_.contract(M2_.contract(psi_.contract(R_, false, false), false, false), false, false), false, false);
    }
}
BENCHMARK(Cytnx_Hpsi_dense_D400);


// Cytnx U1 test
// static void Cytnx_declare_U1(benchmark::State& state)
// {
// 	//auto envB1 = Bond(4,BD_KET, {{0}, {-2}, {2}, {0}});
// 	auto envB1 = Bond(BD_IN, {Qs(-2),Qs(0),Qs(2)}, {1, 2, 1});
// 	//auto envLB2 = Bond(100,BD_KET, {{-6}, {-6}, {-4}, {-4}, {-4}, {-4}, {-4}, {-4}, {-4}, {-4}, {-4}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {4}, {4}, {4}, {4}, {4}, {4}, {4}, {4}, {4}, {4}, {6}, {6}});
// 	auto envLB2 = Bond(BD_IN, {Qs(-6),Qs(-4),Qs(-2),Qs(0),Qs(2),Qs(4),Qs(6)}, {2,9,22,31,24,10,2}); //D=100
// 	//auto envRB2 = Bond(100,BD_KET, {{-6}, {-4}, {-4}, {-4}, {-4}, {-4}, {-4}, {-4}, {-4}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {4}, {4}, {4}, {4}, {4}, {4}, {4}, {4}, {4}, {4}, {4}, {6}, {6}});
// 	auto envRB2 = Bond(BD_IN, {Qs(-6),Qs(-4),Qs(-2),Qs(0),Qs(2),Qs(4),Qs(6)}, {1,8,23,31,24,11,2}); //D=100
// 	// auto phyB = Bond(2,BD_KET,{{1}, {-1}});
// 	auto phyB = Bond(BD_IN, {Qs(-1),Qs(1)}, {1,1}); 
// 	// auto MB = Bond(4,BD_KET, {{0}, {-2}, {2}, {0}});
// 	auto MB = Bond(BD_IN, {Qs(-2),Qs(0),Qs(2)}, {1, 2, 1});
//     for (auto _: state) {
// 		auto psi =  UniTensor({envLB2, phyB.redirect(), phyB.redirect(), envRB2.redirect()});
//     }
// }
// BENCHMARK(Cytnx_declare_U1);

static void Cytnx_Hpsi_U1_D100(benchmark::State& state)
{
	//auto envB1 = Bond(4,BD_KET, {{0}, {-2}, {2}, {0}});
	auto envB1 = Bond(BD_IN, {Qs(-2),Qs(0),Qs(2)}, {1, 2, 1});
	//auto envLB2 = Bond(100,BD_KET, {{-6}, {-6}, {-4}, {-4}, {-4}, {-4}, {-4}, {-4}, {-4}, {-4}, {-4}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {4}, {4}, {4}, {4}, {4}, {4}, {4}, {4}, {4}, {4}, {6}, {6}});
	auto envLB2 = Bond(BD_IN, {Qs(-6),Qs(-4),Qs(-2),Qs(0),Qs(2),Qs(4),Qs(6)}, {2,9,22,31,24,10,2}); //D=100
	//auto envRB2 = Bond(100,BD_KET, {{-6}, {-4}, {-4}, {-4}, {-4}, {-4}, {-4}, {-4}, {-4}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {-2}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {2}, {4}, {4}, {4}, {4}, {4}, {4}, {4}, {4}, {4}, {4}, {4}, {6}, {6}});
	auto envRB2 = Bond(BD_IN, {Qs(-6),Qs(-4),Qs(-2),Qs(0),Qs(2),Qs(4),Qs(6)}, {1,8,23,31,24,11,2}); //D=100
	// auto phyB = Bond(2,BD_KET,{{1}, {-1}});
	auto phyB = Bond(BD_IN, {Qs(-1),Qs(1)}, {1,1}); 
	// auto MB = Bond(4,BD_KET, {{0}, {-2}, {2}, {0}});
	auto MB = Bond(BD_IN, {Qs(-2),Qs(0),Qs(2)}, {1, 2, 1});

	auto M1 = UniTensor({MB, MB.redirect(), phyB, phyB.redirect()});
	auto M2 =  UniTensor({MB, MB.redirect(), phyB, phyB.redirect()});
	auto psi =  UniTensor({envLB2, phyB.redirect(), phyB.redirect(), envRB2.redirect()});
	auto L = UniTensor({envB1.redirect(), envLB2.redirect(), envLB2});
	auto R = UniTensor({envB1, envRB2, envRB2.redirect()});

	auto L_ = L.relabels({-5,-1,0});
	auto R_ = R.relabels({-7,-4,3});
	auto M1_ = M1.relabels({-5,-6,-2,1});
	auto M2_ = M2.relabels({-6,-7,-3,2});
	auto psi_ = psi.relabels({-1,-2,-3,-4});

	for (auto _: state) {
		auto out = L_.contract(M1_.contract(M2_.contract(psi_.contract(R_, false, false), false, false), false, false), false, false);
    }
}
BENCHMARK(Cytnx_Hpsi_U1_D100);

// // itensor dense test
// static void BM_itensor_declare(benchmark::State& state)
// {
//     for (auto _: state) {
// 	    auto i = itensor::Index(4, "index i");
// 	    auto j = itensor::Index(6, "index j");
// 	    auto T = itensor::ITensor(i,j);
// 	    T.set(i=3, j = 2, 3.14159);
//     }
// }
// BENCHMARK(BM_itensor_declare);
 
BENCHMARK_MAIN();

