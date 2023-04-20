#include <benchmark/benchmark.h>
#include <array>
#include <itensor/all.h>
#include <cytnx.hpp>
#include <malloc.h>

using namespace cytnx;
using namespace itensor;

cytnx_int64 D = 64; 
cytnx::UniTensor L = cytnx::UniTensor(zeros({4,D,D}));

static void STANDALONECytnx_Hpsi_dense()
{
	malloc_trim(0);
	cytnx_int64 D = 64;
	cytnx::UniTensor L = cytnx::UniTensor(zeros({4,D,D}));
	cytnx::UniTensor R = cytnx::UniTensor(zeros({4,D,D}));
	cytnx::UniTensor M1 = cytnx::UniTensor(zeros({4,4,2,2}));
	cytnx::UniTensor M2 = cytnx::UniTensor(zeros({4,4,2,2}));
	cytnx::UniTensor psi = cytnx::UniTensor(zeros({D,2,2,D}));

	//// relabel have bugs!
	// auto L_ = L.relabels({-5,-1,0});
	// auto R_ = R.relabels({-7,-4,3});
	// auto M1_ = M1.relabels({-5,-6,-2,1});
	// auto M2_ = M2.relabels({-6,-7,-3,2});
	// auto psi_ = psi.relabels({-1,-2,-3,-4});

	// auto L_ = L.set_labels({-5,-1,0});
	// auto R_ = R.set_labels({-7,-4,3});
	// auto M1_ = M1.set_labels({-5,-6,-2,1});
	// auto M2_ = M2.set_labels({-6,-7,-3,2});
	// auto psi_ = psi.set_labels({-1,-2,-3,-4});

	L.set_labels({-5,-1,0});
	R.set_labels({-7,-4,3});
	M1.set_labels({-5,-6,-2,1});
	M2.set_labels({-6,-7,-3,2});
	psi.set_labels({-1,-2,-3,-4});

	// for (auto _: state) {
		// auto out = L_.contract(M1_.contract(M2_.contract(psi_.contract(R_, true, true), true, true), true, true), true, true);
		auto out = L.contract(M1.contract(M2.contract(psi.contract(R, true, true), true, true), true, true), true, true);
    // }
}

static void STANDALONEitensor_Hpsi_dense() 
{
	malloc_trim(0);
	int D = 64;
	auto Mb1 = Index(4, "Mb1");
	auto Mb2 = Index(4, "Mb2");
	auto phy1 = Index(2,"phy1");
	auto phy2 = Index(2,"phy2");
	auto Lb = Index(D, "Lb");
	auto Rb = Index(D, "Rb");

	auto M1 = randomITensor(Mb1, prime(Mb1), phy1, prime(phy1));
	auto M2 = randomITensor(prime(Mb1), Mb2, phy2, prime(phy2));
	auto L = randomITensor(Mb1, Lb, prime(Lb));
	auto R = randomITensor(Mb2, Rb, prime(Rb));
	auto psi = randomITensor(Lb, phy1, phy2, Rb);


	// for (auto _: state) { 
		auto out = L*(M1*(M2*(psi*R)));
	// }
}


static void Cytnx_Hpsi_dense_(benchmark::State& state)
{	
	malloc_trim(0);
	cytnx_int64 D = state.range(0);
	cytnx::UniTensor L = cytnx::UniTensor(zeros({4,D,D}));
	cytnx::UniTensor R = cytnx::UniTensor(zeros({4,D,D}));
	cytnx::UniTensor M1 = cytnx::UniTensor(zeros({4,4,2,2}));
	cytnx::UniTensor M2 = cytnx::UniTensor(zeros({4,4,2,2}));
	cytnx::UniTensor psi = cytnx::UniTensor(zeros({D,2,2,D}));

	//// relabel have bugs!
	// auto L_ = L.relabels({-5,-1,0});
	// auto R_ = R.relabels({-7,-4,3});
	// auto M1_ = M1.relabels({-5,-6,-2,1});
	// auto M2_ = M2.relabels({-6,-7,-3,2});
	// auto psi_ = psi.relabels({-1,-2,-3,-4});

	// auto L_ = L.set_labels({-5,-1,0});
	// auto R_ = R.set_labels({-7,-4,3});
	// auto M1_ = M1.set_labels({-5,-6,-2,1});
	// auto M2_ = M2.set_labels({-6,-7,-3,2});
	// auto psi_ = psi.set_labels({-1,-2,-3,-4});

	L.set_labels({-5,-1,0});
	R.set_labels({-7,-4,3});
	M1.set_labels({-5,-6,-2,1});
	M2.set_labels({-6,-7,-3,2});
	psi.set_labels({-1,-2,-3,-4});

	for (auto _: state) {
		// auto out = L.contract(M1.contract(M2.contract(psi.contract(R, true, true), true, true), true, true), true, true);
		benchmark::DoNotOptimize(L.contract(M1.contract(M2.contract(psi.contract(R, true, true), true, true), true, true), true, true));
    }
}

static void Cytnx_Hpsi_U1_D64(benchmark::State& state){
	// ( 0 , 2) ( 2 , 1) ( -2 , 1) L:  None
	// ( 0 , 20) ( 2 , 15) ( 4 , 6) ( 6 , 1) ( -6 , 1) ( -4 , 6) ( -2 , 15) L:  None
	// ( 0 , 20) ( 2 , 15) ( 4 , 6) ( 6 , 1) ( -6 , 1) ( -4 , 6) ( -2 , 15) L:  None
	// ( 0 , 2) ( 2 , 1) ( -2 , 1) R:  None
	// ( 0 , 20) ( 2 , 15) ( 4 , 6) ( 6 , 1) ( -6 , 1) ( -4 , 6) ( -2 , 15) R:  None
	// ( 0 , 20) ( 2 , 15) ( 4 , 6) ( 6 , 1) ( -6 , 1) ( -4 , 6) ( -2 , 15) R:  None

	malloc_trim(0);
	auto envB1 = Bond(BD_IN, {Qs(-2),Qs(0),Qs(2)}, {1, 2, 1});
	auto envLB2 = Bond(BD_IN, {Qs(-6),Qs(-4),Qs(-2),Qs(0),Qs(2),Qs(4),Qs(6)}, {1,6,15,20,15,6,1}); //D=64
	auto envRB2 = Bond(BD_IN, {Qs(-6),Qs(-4),Qs(-2),Qs(0),Qs(2),Qs(4),Qs(6)}, {1,6,15,20,15,6,1}); //D=64
	auto phyB = Bond(BD_IN, {Qs(-1),Qs(1)}, {1,1}); 
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
		auto out = L_.contract(M1_.contract(M2_.contract(psi_.contract(R_, true, true), true, true), true, true), true, true);
    }
}

static void Cytnx_Hpsi_U1_D100(benchmark::State& state)
{
	malloc_trim(0);
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
		auto out = L_.contract(M1_.contract(M2_.contract(psi_.contract(R_, true, true), true, true), true, true), true, true);
    }
}

static void Cytnx_Hpsi_U1_D200(benchmark::State& state){
	// ( 0 , 61) ( 2 , 50) ( 4 , 23) ( 6 , 5) ( -6 , 3) ( -4 , 15) ( -2 , 43) L:  None
	// ( 0 , 61) ( 2 , 50) ( 4 , 23) ( 6 , 5) ( -6 , 3) ( -4 , 15) ( -2 , 43) L:  None
	// ( 0 , 62) ( 2 , 48) ( 4 , 21) ( 6 , 4) ( 8 , 1) ( -6 , 3) ( -4 , 17) ( -2 , 44) R:  None
	// ( 0 , 62) ( 2 , 48) ( 4 , 21) ( 6 , 4) ( 8 , 1) ( -6 , 3) ( -4 , 17) ( -2 , 44) R:  None

	malloc_trim(0);
	auto envB1 = Bond(BD_IN, {Qs(-2),Qs(0),Qs(2)}, {1, 2, 1});
	auto envLB2 = Bond(BD_IN, {Qs(-6),Qs(-4),Qs(-2),Qs(0),Qs(2),Qs(4),Qs(6)}, {3,15,43,61,50,23,5}); //D=200
	auto envRB2 = Bond(BD_IN, {Qs(-6),Qs(-4),Qs(-2),Qs(0),Qs(2),Qs(4),Qs(6)}, {3,17,44,62,48,21,4}); //D=200
	auto phyB = Bond(BD_IN, {Qs(-1),Qs(1)}, {1,1}); 
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

	// L.set_labels({-5,-1,0});
	// R.set_labels({-7,-4,3});
	// M1.set_labels({-5,-6,-2,1});
	// M2.set_labels({-6,-7,-3,2});
	// psi.set_labels({-1,-2,-3,-4});

	for (auto _: state) {
		auto out = L_.contract(M1_.contract(M2_.contract(psi_.contract(R_, true, true), true, true), true, true), true, true);
		// auto out = L.contract(M1.contract(M2.contract(psi.contract(R, true, true), true, true), true, true), true, true);
    }
}

static void Cytnx_Hpsi_U1_D300(benchmark::State& state){
	// ( 1 , 86) ( 3 , 52) ( 5 , 16) ( 7 , 3) ( -7 , 2) ( -5 , 14) ( -3 , 45) ( -1 , 82) L:  None
	// ( 1 , 86) ( 3 , 52) ( 5 , 16) ( 7 , 3) ( -7 , 2) ( -5 , 14) ( -3 , 45) ( -1 , 82) L:  None
	// ( 1 , 87) ( 3 , 54) ( 5 , 18) ( 7 , 3) ( -7 , 2) ( -5 , 13) ( -3 , 43) ( -1 , 80) R:  None
	// ( 1 , 87) ( 3 , 54) ( 5 , 18) ( 7 , 3) ( -7 , 2) ( -5 , 13) ( -3 , 43) ( -1 , 80) R:  None

	malloc_trim(0);
	auto envB1 = Bond(BD_IN, {Qs(-2),Qs(0),Qs(2)}, {1, 2, 1});
	auto envLB2 = Bond(BD_IN, {Qs(1),Qs(3),Qs(5),Qs(7),Qs(-7),Qs(-5),Qs(-3),Qs(-1)}, {86,52,16,3,2,14,45,82}); //D=300
	auto envRB2 = Bond(BD_IN, {Qs(1),Qs(3),Qs(5),Qs(7),Qs(-7),Qs(-5),Qs(-3),Qs(-1)}, {87,54,18,3,2,13,43,80}); //D=300
	auto phyB = Bond(BD_IN, {Qs(-1),Qs(1)}, {1,1}); 
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
		auto out = L_.contract(M1_.contract(M2_.contract(psi_.contract(R_, true, true), true, true), true, true), true, true);
    }
}

static void Cytnx_Hpsi_U1_D400(benchmark::State& state){
	// ( 1 , 113) ( 3 , 69) ( 5 , 23) ( 7 , 4) ( -7 , 3) ( -5 , 19) ( -3 , 61) ( -1 , 108) L:  None
	// ( 1 , 112) ( 3 , 72) ( 5 , 26) ( 7 , 4) ( -7 , 3) ( -5 , 19) ( -3 , 59) ( -1 , 105) R:  None
	malloc_trim(0);
	auto envB1 = Bond(BD_IN, {Qs(-2),Qs(0),Qs(2)}, {1, 2, 1});
	auto envLB2 = Bond(BD_IN, {Qs(1),Qs(3),Qs(5),Qs(7),Qs(-7),Qs(-5),Qs(-3),Qs(-1)}, {113,69,23,4,3,19,61,108}); //D=400
	auto envRB2 = Bond(BD_IN, {Qs(1),Qs(3),Qs(5),Qs(7),Qs(-7),Qs(-5),Qs(-3),Qs(-1)}, {112,72,26,4,3,19,59,105}); //D=400
	auto phyB = Bond(BD_IN, {Qs(-1),Qs(1)}, {1,1}); 
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
		auto out = L_.contract(M1_.contract(M2_.contract(psi_.contract(R_, true, true), true, true), true, true), true, true);
    }
}

static void Cytnx_Hpsi_U1_D1000(benchmark::State& state){
	// ( 1 , 113) ( 3 , 69) ( 5 , 23) ( 7 , 4) ( -7 , 3) ( -5 , 19) ( -3 , 61) ( -1 , 108) L:  None
	// ( 1 , 112) ( 3 , 72) ( 5 , 26) ( 7 , 4) ( -7 , 3) ( -5 , 19) ( -3 , 59) ( -1 , 105) R:  None
	malloc_trim(0);
	auto envB1 = Bond(BD_IN, {Qs(-2),Qs(0),Qs(2)}, {1, 2, 1});
	auto envLB2 = Bond(BD_IN, {Qs(1),Qs(3),Qs(5),Qs(7),Qs(-7),Qs(-5),Qs(-3),Qs(-1)}, {350,100,30,20,20,30,100,350}); //D=400
	auto envRB2 = Bond(BD_IN, {Qs(1),Qs(3),Qs(5),Qs(7),Qs(-7),Qs(-5),Qs(-3),Qs(-1)}, {350,100,30,20,20,30,100,350}); //D=400
	auto phyB = Bond(BD_IN, {Qs(-1),Qs(1)}, {1,1}); 
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
		auto out = L_.contract(M1_.contract(M2_.contract(psi_.contract(R_, true, true), true, true), true, true), true, true);
    }
}

static void Cytnx_Hpsi_U1_D2000(benchmark::State& state){
	// ( 1 , 113) ( 3 , 69) ( 5 , 23) ( 7 , 4) ( -7 , 3) ( -5 , 19) ( -3 , 61) ( -1 , 108) L:  None
	// ( 1 , 112) ( 3 , 72) ( 5 , 26) ( 7 , 4) ( -7 , 3) ( -5 , 19) ( -3 , 59) ( -1 , 105) R:  None
	malloc_trim(0);
	auto envB1 = Bond(BD_IN, {Qs(-2),Qs(0),Qs(2)}, {1, 2, 1});
	auto envLB2 = Bond(BD_IN, {Qs(1),Qs(3),Qs(5),Qs(7),Qs(-7),Qs(-5),Qs(-3),Qs(-1)}, {600,300,50,50,50,50,300,600}); //D=400
	auto envRB2 = Bond(BD_IN, {Qs(1),Qs(3),Qs(5),Qs(7),Qs(-7),Qs(-5),Qs(-3),Qs(-1)}, {600,300,50,50,50,50,300,600}); //D=400
	auto phyB = Bond(BD_IN, {Qs(-1),Qs(1)}, {1,1}); 
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
		auto out = L_.contract(M1_.contract(M2_.contract(psi_.contract(R_, true, true), true, true), true, true), true, true);
    }
}

static void Cytnx_Hpsi_U1_D3000(benchmark::State& state){
	// ( 1 , 113) ( 3 , 69) ( 5 , 23) ( 7 , 4) ( -7 , 3) ( -5 , 19) ( -3 , 61) ( -1 , 108) L:  None
	// ( 1 , 112) ( 3 , 72) ( 5 , 26) ( 7 , 4) ( -7 , 3) ( -5 , 19) ( -3 , 59) ( -1 , 105) R:  None
	malloc_trim(0);
	auto envB1 = Bond(BD_IN, {Qs(-2),Qs(0),Qs(2)}, {1, 2, 1});
	auto envLB2 = Bond(BD_IN, {Qs(1),Qs(3),Qs(5),Qs(7),Qs(-7),Qs(-5),Qs(-3),Qs(-1)}, {1000,250,150,100,100,150,250,1000}); //D=400
	auto envRB2 = Bond(BD_IN, {Qs(1),Qs(3),Qs(5),Qs(7),Qs(-7),Qs(-5),Qs(-3),Qs(-1)}, {1000,250,150,100,100,150,250,1000}); //D=400
	auto phyB = Bond(BD_IN, {Qs(-1),Qs(1)}, {1,1}); 
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
		auto out = L_.contract(M1_.contract(M2_.contract(psi_.contract(R_, true, true), true, true), true, true), true, true);
    }
}

static void itensor_Hpsi_dense_(benchmark::State& state)
{
	malloc_trim(0);
	int D = state.range(0);
	auto Mb1 = Index(4, "Mb1");
	auto Mb2 = Index(4, "Mb2");
	auto phy1 = Index(2,"phy1");
	auto phy2 = Index(2,"phy2");
	auto Lb = Index(D, "Lb");
	auto Rb = Index(D, "Rb");

	auto M1 = randomITensor(Mb1, prime(Mb1), phy1, prime(phy1));
	auto M2 = randomITensor(prime(Mb1), Mb2, phy2, prime(phy2));
	auto L = randomITensor(Mb1, Lb, prime(Lb));
	auto R = randomITensor(Mb2, Rb, prime(Rb));
	auto psi = randomITensor(Lb, phy1, phy2, Rb);


	for (auto _: state) {
		benchmark::DoNotOptimize(L*(M1*(M2*(psi*R))));
		// auto out = L*(M1*(M2*(psi*R)));
	}
}

static void itensor_Hpsi_U1_D64(benchmark::State& state)
{

	malloc_trim(0);
	// auto envB1 = Bond(BD_IN, {Qs(-2),Qs(0),Qs(2)}, {1, 2, 1});
	// auto envLB2 = Bond(BD_IN, {Qs(-6),Qs(-4),Qs(-2),Qs(0),Qs(2),Qs(4),Qs(6)}, {1,6,15,20,15,6,1}); //D=64
	// auto envRB2 = Bond(BD_IN, {Qs(-6),Qs(-4),Qs(-2),Qs(0),Qs(2),Qs(4),Qs(6)}, {1,6,15,20,15,6,1}); //D=64
	// auto phyB = Bond(BD_IN, {Qs(-1),Qs(1)}, {1,1}); 
	// auto MB = Bond(BD_IN, {Qs(-2),Qs(0),Qs(2)}, {1, 2, 1});
	int D = 64;
	auto Mb1 = Index(QN(-2), 1, QN(0), 2, QN(2), 1, In, "Mb1");
	auto Mb2 = Index(QN(-2), 1, QN(0), 2, QN(2), 1, In, "Mb2");
	auto phy1 = Index(QN(-1), 1, QN(1), 1, In,"phy1");
	auto phy2 = Index(QN(-1), 1, QN(1), 1, In,"phy2");
	auto Lb = Index(QN(-6),1,QN(-4),6,QN(-2),15,QN(0),20,QN(2),15,QN(4),6,QN(6),1,In,"Lb");
	auto Rb = Index(QN(-6),1,QN(-4),6,QN(-2),15,QN(0),20,QN(2),15,QN(4),6,QN(6),1,In,"Rb");

	auto M1 = randomITensor(QN(0), Mb1, dag(prime(Mb1)), phy1, dag(prime(phy1)));
	auto M2 = randomITensor(QN(0),prime(Mb1), dag(Mb2), phy2, dag(prime(phy2)));
	auto L = randomITensor(QN(0),dag(Mb1), dag(Lb), prime(Lb));
	auto R = randomITensor(QN(0),Mb2, Rb, dag(prime(Rb)));
	auto psi = randomITensor(QN(0),Lb, dag(phy1), dag(phy2), dag(Rb));

	for (auto _: state) {
		auto out = L*(M1*(M2*(psi*R)));
	}
}

static void itensor_Hpsi_U1_D100(benchmark::State& state)
{

	// auto envB1 = Bond(BD_IN, {Qs(-2),Qs(0),Qs(2)}, {1, 2, 1});
	// auto envLB2 = Bond(BD_IN, {Qs(-6),Qs(-4),Qs(-2),Qs(0),Qs(2),Qs(4),Qs(6)}, {2,9,22,31,24,10,2}); //D=100
	// auto envRB2 = Bond(BD_IN, {Qs(-6),Qs(-4),Qs(-2),Qs(0),Qs(2),Qs(4),Qs(6)}, {1,8,23,31,24,11,2}); //D=100
	// auto phyB = Bond(BD_IN, {Qs(-1),Qs(1)}, {1,1}); 
	// auto MB = Bond(BD_IN, {Qs(-2),Qs(0),Qs(2)}, {1, 2, 1});

	malloc_trim(0);
	auto Mb1 = Index(QN(-2), 1, QN(0), 2, QN(2), 1, In, "Mb1");
	auto Mb2 = Index(QN(-2), 1, QN(0), 2, QN(2), 1, In, "Mb2");
	auto phy1 = Index(QN(-1), 1, QN(1), 1, In,"phy1");
	auto phy2 = Index(QN(-1), 1, QN(1), 1, In,"phy2");
	auto Lb = Index(QN(-6),2,QN(-4),9,QN(-2),22,QN(0),31,QN(2),24,QN(4),10,QN(6),2,In,"Lb");
	auto Rb = Index(QN(-6),2,QN(-4),9,QN(-2),22,QN(0),31,QN(2),24,QN(4),10,QN(6),2,In,"Rb");

	auto M1 = randomITensor(QN(0), Mb1, dag(prime(Mb1)), phy1, dag(prime(phy1)));
	auto M2 = randomITensor(QN(0),prime(Mb1), dag(Mb2), phy2, dag(prime(phy2)));
	auto L = randomITensor(QN(0),dag(Mb1), dag(Lb), prime(Lb));
	auto R = randomITensor(QN(0),Mb2, Rb, dag(prime(Rb)));
	auto psi = randomITensor(QN(0),Lb, dag(phy1), dag(phy2), dag(Rb));

	for (auto _: state) {
		auto out = L*(M1*(M2*(psi*R)));
	}
}

static void itensor_Hpsi_U1_D200(benchmark::State& state)
{

	// ( 0 , 61) ( 2 , 50) ( 4 , 23) ( 6 , 5) ( -6 , 3) ( -4 , 15) ( -2 , 43) L:  None
	// ( 0 , 61) ( 2 , 50) ( 4 , 23) ( 6 , 5) ( -6 , 3) ( -4 , 15) ( -2 , 43) L:  None
	// ( 0 , 62) ( 2 , 48) ( 4 , 21) ( 6 , 4) ( 8 , 1) ( -6 , 3) ( -4 , 17) ( -2 , 44) R:  None
	// ( 0 , 62) ( 2 , 48) ( 4 , 21) ( 6 , 4) ( 8 , 1) ( -6 , 3) ( -4 , 17) ( -2 , 44) R:  None

	malloc_trim(0);
	auto Mb1 = Index(QN(-2), 1, QN(0), 2, QN(2), 1, In, "Mb1");
	auto Mb2 = Index(QN(-2), 1, QN(0), 2, QN(2), 1, In, "Mb2");
	auto phy1 = Index(QN(-1), 1, QN(1), 1, In,"phy1");
	auto phy2 = Index(QN(-1), 1, QN(1), 1, In,"phy2");
	auto Lb = Index(QN(-6),3,QN(-4),15,QN(-2),43,QN(0),61,QN(2),50,QN(4),23,QN(6),5,In,"Lb");
	auto Rb = Index(QN(-6),3,QN(-4),17,QN(-2),44,QN(0),62,QN(2),48,QN(4),21,QN(6),4,In,"Rb");

	auto M1 = randomITensor(QN(0), Mb1, dag(prime(Mb1)), phy1, dag(prime(phy1)));
	auto M2 = randomITensor(QN(0),prime(Mb1), dag(Mb2), phy2, dag(prime(phy2)));
	auto L = randomITensor(QN(0),dag(Mb1), dag(Lb), prime(Lb));
	auto R = randomITensor(QN(0),Mb2, Rb, dag(prime(Rb)));
	auto psi = randomITensor(QN(0),Lb, dag(phy1), dag(phy2), dag(Rb));

	for (auto _: state) {
		auto out = L*(M1*(M2*(psi*R)));
	}
}

static void itensor_Hpsi_U1_D300(benchmark::State& state)
{


	// auto envLB2 = Bond(BD_IN, {Qs(1),Qs(3),Qs(5),Qs(7),Qs(-7),Qs(-5),Qs(-3),Qs(-1)}, {86,52,16,3,2,14,45,82}); //D=300
	// auto envRB2 = Bond(BD_IN, {Qs(1),Qs(3),Qs(5),Qs(7),Qs(-7),Qs(-5),Qs(-3),Qs(-1)}, {87,54,18,3,2,13,43,80}); //D=300

	malloc_trim(0);
	auto Mb1 = Index(QN(-2), 1, QN(0), 2, QN(2), 1, In, "Mb1");
	auto Mb2 = Index(QN(-2), 1, QN(0), 2, QN(2), 1, In, "Mb2");
	auto phy1 = Index(QN(-1), 1, QN(1), 1, In,"phy1");
	auto phy2 = Index(QN(-1), 1, QN(1), 1, In,"phy2");
	auto Lb = Index(QN(1),86,QN(3),52,QN(5),16,QN(7),3,QN(-7),2,QN(-5),14,QN(-3),45,QN(-1),82,In,"Lb");
	auto Rb = Index(QN(1),87,QN(3),54,QN(5),18,QN(7),3,QN(-7),2,QN(-5),13,QN(-3),43,QN(-1),80,In,"Rb");

	auto M1 = randomITensor(QN(0), Mb1, dag(prime(Mb1)), phy1, dag(prime(phy1)));
	auto M2 = randomITensor(QN(0),prime(Mb1), dag(Mb2), phy2, dag(prime(phy2)));
	auto L = randomITensor(QN(0),dag(Mb1), dag(Lb), prime(Lb));
	auto R = randomITensor(QN(0),Mb2, Rb, dag(prime(Rb)));
	auto psi = randomITensor(QN(0),Lb, dag(phy1), dag(phy2), dag(Rb));

	for (auto _: state) {
		auto out = L*(M1*(M2*(psi*R)));
	}
}

static void itensor_Hpsi_U1_D400(benchmark::State& state)
{


	// auto envLB2 = Bond(BD_IN, {Qs(1),Qs(3),Qs(5),Qs(7),Qs(-7),Qs(-5),Qs(-3),Qs(-1)}, {113,69,23,4,3,19,61,108}); //D=400
	// auto envRB2 = Bond(BD_IN, {Qs(1),Qs(3),Qs(5),Qs(7),Qs(-7),Qs(-5),Qs(-3),Qs(-1)}, {112,72,26,4,3,19,59,105}); //D=400

	malloc_trim(0);
	auto Mb1 = Index(QN(-2), 1, QN(0), 2, QN(2), 1, In, "Mb1");
	auto Mb2 = Index(QN(-2), 1, QN(0), 2, QN(2), 1, In, "Mb2");
	auto phy1 = Index(QN(-1), 1, QN(1), 1, In,"phy1");
	auto phy2 = Index(QN(-1), 1, QN(1), 1, In,"phy2");
	auto Lb = Index(QN(1),113,QN(3),69,QN(5),23,QN(7),4,QN(-7),3,QN(-5),19,QN(-3),61,QN(-1),108,In,"Lb");
	auto Rb = Index(QN(1),112,QN(3),72,QN(5),26,QN(7),4,QN(-7),3,QN(-5),19,QN(-3),59,QN(-1),105,In,"Rb");

	auto M1 = randomITensor(QN(0), Mb1, dag(prime(Mb1)), phy1, dag(prime(phy1)));
	auto M2 = randomITensor(QN(0),prime(Mb1), dag(Mb2), phy2, dag(prime(phy2)));
	auto L = randomITensor(QN(0),dag(Mb1), dag(Lb), prime(Lb));
	auto R = randomITensor(QN(0),Mb2, Rb, dag(prime(Rb)));
	auto psi = randomITensor(QN(0),Lb, dag(phy1), dag(phy2), dag(Rb));

	for (auto _: state) {
		auto out = L*(M1*(M2*(psi*R)));
	}
}

static void itensor_Hpsi_U1_D1000(benchmark::State& state)
{

	// auto envLB2 = Bond(BD_IN, {Qs(1),Qs(3),Qs(5),Qs(7),Qs(-7),Qs(-5),Qs(-3),Qs(-1)}, {113,69,23,4,3,19,61,108}); //D=400
	// auto envRB2 = Bond(BD_IN, {Qs(1),Qs(3),Qs(5),Qs(7),Qs(-7),Qs(-5),Qs(-3),Qs(-1)}, {112,72,26,4,3,19,59,105}); //D=400

	malloc_trim(0);
	auto Mb1 = Index(QN(-2), 1, QN(0), 2, QN(2), 1, In, "Mb1");
	auto Mb2 = Index(QN(-2), 1, QN(0), 2, QN(2), 1, In, "Mb2");
	auto phy1 = Index(QN(-1), 1, QN(1), 1, In,"phy1");
	auto phy2 = Index(QN(-1), 1, QN(1), 1, In,"phy2");
	auto Lb = Index(QN(1),350,QN(3),100,QN(5),30,QN(7),20,QN(-7),20,QN(-5),30,QN(-3),100,QN(-1),350,In,"Lb");
	auto Rb = Index(QN(1),350,QN(3),100,QN(5),30,QN(7),20,QN(-7),20,QN(-5),30,QN(-3),100,QN(-1),350,In,"Rb");

	auto M1 = randomITensor(QN(0), Mb1, dag(prime(Mb1)), phy1, dag(prime(phy1)));
	auto M2 = randomITensor(QN(0),prime(Mb1), dag(Mb2), phy2, dag(prime(phy2)));
	auto L = randomITensor(QN(0),dag(Mb1), dag(Lb), prime(Lb));
	auto R = randomITensor(QN(0),Mb2, Rb, dag(prime(Rb)));
	auto psi = randomITensor(QN(0),Lb, dag(phy1), dag(phy2), dag(Rb));

	for (auto _: state) {
		auto out = L*(M1*(M2*(psi*R)));
	}
}
// 350,100,30,20,20,30,100,350
// 600,300,50,50,50,50,300,600
// 1000,250,150,100,100,150,250,1000

static void itensor_Hpsi_U1_D2000(benchmark::State& state)
{

	malloc_trim(0);
	auto Mb1 = Index(QN(-2), 1, QN(0), 2, QN(2), 1, In, "Mb1");
	auto Mb2 = Index(QN(-2), 1, QN(0), 2, QN(2), 1, In, "Mb2");
	auto phy1 = Index(QN(-1), 1, QN(1), 1, In,"phy1");
	auto phy2 = Index(QN(-1), 1, QN(1), 1, In,"phy2");
	auto Lb = Index(QN(1),600,QN(3),300,QN(5),50,QN(7),50,QN(-7),50,QN(-5),50,QN(-3),300,QN(-1),600,In,"Lb");
	auto Rb = Index(QN(1),600,QN(3),300,QN(5),50,QN(7),50,QN(-7),50,QN(-5),50,QN(-3),300,QN(-1),600,In,"Rb");

	auto M1 = randomITensor(QN(0), Mb1, dag(prime(Mb1)), phy1, dag(prime(phy1)));
	auto M2 = randomITensor(QN(0),prime(Mb1), dag(Mb2), phy2, dag(prime(phy2)));
	auto L = randomITensor(QN(0),dag(Mb1), dag(Lb), prime(Lb));
	auto R = randomITensor(QN(0),Mb2, Rb, dag(prime(Rb)));
	auto psi = randomITensor(QN(0),Lb, dag(phy1), dag(phy2), dag(Rb));

	for (auto _: state) {
		auto out = L*(M1*(M2*(psi*R)));
	}
}

//1000,250,150,100,100,150,250,1000
static void itensor_Hpsi_U1_D3000(benchmark::State& state)
{

	malloc_trim(0);
	auto Mb1 = Index(QN(-2), 1, QN(0), 2, QN(2), 1, In, "Mb1");
	auto Mb2 = Index(QN(-2), 1, QN(0), 2, QN(2), 1, In, "Mb2");
	auto phy1 = Index(QN(-1), 1, QN(1), 1, In,"phy1");
	auto phy2 = Index(QN(-1), 1, QN(1), 1, In,"phy2");
	auto Lb = Index(QN(1),1000,QN(3),250,QN(5),150,QN(7),100,QN(-7),100,QN(-5),150,QN(-3),250,QN(-1),1000,In,"Lb");
	auto Rb = Index(QN(1),1000,QN(3),250,QN(5),150,QN(7),100,QN(-7),100,QN(-5),150,QN(-3),250,QN(-1),1000,In,"Rb");

	auto M1 = randomITensor(QN(0), Mb1, dag(prime(Mb1)), phy1, dag(prime(phy1)));
	auto M2 = randomITensor(QN(0),prime(Mb1), dag(Mb2), phy2, dag(prime(phy2)));
	auto L = randomITensor(QN(0),dag(Mb1), dag(Lb), prime(Lb));
	auto R = randomITensor(QN(0),Mb2, Rb, dag(prime(Rb)));
	auto psi = randomITensor(QN(0),Lb, dag(phy1), dag(phy2), dag(Rb));

	for (auto _: state) {
		auto out = L*(M1*(M2*(psi*R)));
	}
}

static void itensor_permute(benchmark::State& state){
	// malloc_trim(0);
	// int D = state.range(0);
	// auto i = Index(D,"i");
	// auto j = Index(D,"j");
	// auto k = Index(D,"k");
	// auto l = Index(D,"l");
	// auto m = Index(D,"m");
	// auto n = Index(D,"n");
	// auto T = randomITensor(i,j,k,l,m,n);
	for(auto _ : state){
		malloc_trim(0);
		int D = state.range(0);
		auto i = Index(D,"i");
		auto j = Index(D,"j");
		auto k = Index(D,"k");
		auto l = Index(D,"l");
		auto m = Index(D,"m");
		auto n = Index(D,"n");
		auto T = randomITensor(i,j,k,l,m,n);
		// auto T = ITensor(i,j,k,l,m,n);
		T = permute(T,{k,j,m,l,n,i});
	}
}

static void Cytnx_Movemem(benchmark::State& state){
	// malloc_trim(0);
	cytnx_int64 D = state.range(0);
	cytnx::UniTensor T = cytnx::UniTensor(zeros({D,D,D,D,D,D}));
	for(auto _ : state){
		malloc_trim(0);
		// cytnx_int64 D = state.range(0);
		// cytnx::UniTensor T = cytnx::UniTensor(zeros({D,D,D,D,D,D}));
		// cytnx::UniTensor T = cytnx::UniTensor(cytnx::random::rand({D,D,D,D,D,D}));
		T.permute({2,1,4,3,5,0}).contiguous();
	}
}
BENCHMARK(Cytnx_Hpsi_U1_D64);
BENCHMARK(Cytnx_Hpsi_U1_D64);
BENCHMARK(Cytnx_Hpsi_U1_D64);
BENCHMARK(Cytnx_Hpsi_U1_D64);
BENCHMARK(Cytnx_Hpsi_U1_D64);
BENCHMARK(Cytnx_Hpsi_U1_D64);
BENCHMARK(Cytnx_Hpsi_U1_D64);
BENCHMARK(Cytnx_Hpsi_U1_D64);
BENCHMARK(Cytnx_Hpsi_U1_D64);

// BENCHMARK(Cytnx_Hpsi_dense)->Arg(64);
// BENCHMARK(Cytnx_Hpsi_dense)->Arg(100);
// BENCHMARK(Cytnx_Hpsi_dense)->Arg(200);
// BENCHMARK(Cytnx_Hpsi_dense)->Arg(300);
// BENCHMARK(Cytnx_Hpsi_dense)->Arg(600);

// BENCHMARK(itensor_Hpsi_dense)->Arg(64);
// BENCHMARK(itensor_Hpsi_dense)->Arg(100);
// BENCHMARK(itensor_Hpsi_dense)->Arg(200);
// BENCHMARK(itensor_Hpsi_dense)->Arg(300);
// BENCHMARK(itensor_Hpsi_dense)->Arg(600);

// BENCHMARK(Cytnx_Hpsi_dense)->Arg(64);
// BENCHMARK(Cytnx_Hpsi_dense)->Arg(100);
// BENCHMARK(Cytnx_Hpsi_dense)->Arg(200);
// BENCHMARK(Cytnx_Hpsi_dense)->Arg(300);
// BENCHMARK(Cytnx_Hpsi_dense)->Arg(600);

// BENCHMARK(Cytnx_Hpsi_U1_D64);
// BENCHMARK(Cytnx_Hpsi_U1_D100);
// BENCHMARK(Cytnx_Hpsi_U1_D200);
// BENCHMARK(Cytnx_Hpsi_U1_D300);
// BENCHMARK(Cytnx_Hpsi_U1_D400);

// BENCHMARK(itensor_Hpsi_U1_D64);
// BENCHMARK(itensor_Hpsi_U1_D100);
// BENCHMARK(itensor_Hpsi_U1_D200);
// BENCHMARK(itensor_Hpsi_U1_D300);
// BENCHMARK(itensor_Hpsi_U1_D400);

// BENCHMARK(Cytnx_Hpsi_U1_D64);
// BENCHMARK(Cytnx_Hpsi_U1_D100);
// BENCHMARK(Cytnx_Hpsi_U1_D200);
// BENCHMARK(Cytnx_Hpsi_U1_D300);
// BENCHMARK(Cytnx_Hpsi_U1_D400);

// BENCHMARK(itensor_permute)->Arg(2);
// BENCHMARK(Cytnx_Movemem)->Arg(2);
// BENCHMARK(itensor_permute)->Arg(4);
// BENCHMARK(Cytnx_Movemem)->Arg(4);
// BENCHMARK(itensor_permute)->Arg(6);
// BENCHMARK(Cytnx_Movemem)->Arg(6);
// BENCHMARK(itensor_permute)->Arg(8);
// BENCHMARK(Cytnx_Movemem)->Arg(8);
// BENCHMARK(itensor_permute)->Arg(16);
// BENCHMARK(Cytnx_Movemem)->Arg(16);
// BENCHMARK(itensor_permute)->Arg(20);
// BENCHMARK(Cytnx_Movemem)->Arg(20);

// BENCHMARK(itensor_permute)->Arg(2);
// BENCHMARK(Cytnx_Movemem)->Arg(2);
// BENCHMARK(itensor_permute)->Arg(3);
// BENCHMARK(Cytnx_Movemem)->Arg(3);
// BENCHMARK(itensor_permute)->Arg(4);
// BENCHMARK(Cytnx_Movemem)->Arg(4);
// BENCHMARK(itensor_permute)->Arg(5);
// BENCHMARK(Cytnx_Movemem)->Arg(5);
// BENCHMARK(itensor_permute)->Arg(6);
// BENCHMARK(Cytnx_Movemem)->Arg(6);
// BENCHMARK(itensor_permute)->Arg(8);
// BENCHMARK(Cytnx_Movemem)->Arg(8);
// BENCHMARK(itensor_permute)->Arg(10);
// BENCHMARK(Cytnx_Movemem)->Arg(10);
// BENCHMARK(itensor_permute)->Arg(16);
// BENCHMARK(Cytnx_Movemem)->Arg(16);
// BENCHMARK(itensor_permute)->Arg(20);
// BENCHMARK(Cytnx_Movemem)->Arg(20);

BENCHMARK_MAIN();
// int main(){
// 	malloc_trim(0);
// 	for(int i=0;i<1;i++){
// 		STANDALONEitensor_Hpsi_dense();
// 	}
// 	for(int i=0;i<1;i++){
// 		STANDALONECytnx_Hpsi_dense();
// 	}
// 	// for(int i=0;i<1;i++){
// 	// 	STANDALONEitensor_Hpsi_dense();
// 	// }
// }