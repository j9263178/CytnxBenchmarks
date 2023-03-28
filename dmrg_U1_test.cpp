#include <benchmark/benchmark.h>
#include <array>
#include <itensor/all.h>
#include <cytnx.hpp>
#include <malloc.h>
// #include "dmrg.h"

using namespace cytnx;
using namespace itensor;

class Hxx: public LinOp{
    public:
        Network anet;
        UniTensor L;
        UniTensor R;
        UniTensor M1;
        UniTensor M2;

    Hxx(Network anet, UniTensor L, UniTensor M1, UniTensor M2, UniTensor R):
        LinOp("mv", 0, Type.Double, Device.cpu){
        this->anet = anet;
        this->L = L;
        this->R = R;
        this->M1 = M1;
        this->M2 = M2;
    }
    UniTensor matvec(const UniTensor &v) override{
        auto lbl = v.labels(); 
        // this->anet.PutUniTensor("psi",v);
        // auto out = this->anet.Launch(false);
        auto L_ = this->L.relabels({"-5","-1","0"});
        auto R_ = this->R.relabels({"-7","-4","3"});
        auto M1_ = this->M1.relabels({"-5","-6","-2","1"});
        auto M2_ = this->M2.relabels({"-6","-7","-3","2"});
        auto psi_ = v.relabels({"-1","-2","-3","-4"});
        auto out = L_.contract(M1_.contract(M2_.contract(psi_.contract(R_, true, true), true, true), true, true), true, true);

        // out.contiguous_();
        out.set_labels(lbl);
        return out;
    }
};

std::vector<UniTensor> optimize_psi(UniTensor psivec,UniTensor L, UniTensor M1,UniTensor M2, UniTensor R, int maxit, int krydim){
    auto anet = Network();
    // anet.FromString({"psi: -1,-2,-3,-4",\
    //                 "L: -5,-1,0",\
    //                 "R: -7,-4,3",\
    //                 "M1: -5,-6,-2,1",\
    //                 "M2: -6,-7,-3,2",\
    //                 "TOUT: 0,1;2,3",\
    //                 "ORDER: (L,(M1,(M2,(psi,R))))"});
    // anet.PutUniTensors({"L","M1","M2","R"},{L,M1,M2,R});
    auto H = Hxx(anet,L,M1,M2,R);
    auto res = linalg::Lanczos(&H, psivec, "Gnd", 999, maxit, 1, true, false, 0, false);
    return res;
};


static void cytnx_dmrg_U1(benchmark::State& state){

	malloc_trim(0);

    int chi = state.range(0); // Maximum allowed bondim (Perform truncation if exceeds)
    int Nsites = state.range(1); // Number of sites
    int numsweeps = state.range(2); // number of DMRG sweeps
    int maxit = 2; // iterations of Lanczos method
    int krydim = 4; // dimension of Krylov subspace (Will not be used in Lanczos "Gnd" method)
    int d = 2; // 1/2-spin

    //// set-up ////

    Bond bd_inner = Bond(BD_KET,{{0},{-2},{2},{0}},{1,1,1,1});
    Bond bd_phys = Bond(BD_KET,{{1},{-1}},{1,1});
    UniTensor M = UniTensor({bd_inner,bd_inner.redirect(),bd_phys, bd_phys.redirect()}); M.set_rowrank(2);
    // I
    M.set_elem({0,0,0,0},1);
    M.set_elem({0,0,1,1},1);
    M.set_elem({3,3,0,0},1);
    M.set_elem({3,3,1,1},1);
    // S-
    M.set_elem({0,1,1,0},sqrt(2));
    // S+ 
    M.set_elem({0,2,0,1},sqrt(2));
    // S+
    M.set_elem({1,3,0,1},sqrt(2));
    // S-
    M.set_elem({2,3,1,0},sqrt(2));
    int q = 0; // conserving glb Qn 
    Bond VbdL = Bond(BD_KET,{{0}},{1});
    Bond VbdR = Bond(BD_KET,{{q}},{1});
    UniTensor L0 = UniTensor({bd_inner.redirect(),VbdL.redirect(),VbdL}); L0.set_rowrank(1);//Left boundary 
    UniTensor R0 = UniTensor({bd_inner,VbdR,VbdR.redirect()}); R0.set_rowrank(1); //Right boundary
    L0.set_elem({0,0,0},1);
    R0.set_elem({3,0,0},1);
    std::vector<UniTensor> A(Nsites);
    int qcntr = 0;
    int cq;
    if(qcntr <= q)
        cq = 1;
    else
        cq = -1;
    qcntr+=cq;

    A[0] = UniTensor({VbdL,bd_phys.redirect(),Bond(BD_BRA,{{qcntr}},{1})}); A[0].set_rowrank(2);
    A[0].get_block_(0).at({0,0,0})= 1;

    for(int k = 1; k<Nsites;k++){
        auto B1 = A[k-1].bonds()[2].redirect();
        auto B2 = A[k-1].bonds()[1];
        if(qcntr <= q)
            cq = 1;
        else
            cq = -1;
        qcntr+=cq;
        auto B3 = Bond(BD_BRA,{{qcntr}},{1});

        A[k] = UniTensor({B1,B2,B3}); A[k].set_rowrank(2);
        A[k].set_labels({2*k,2*k+1,2*k+2});

        A[k].get_block_(0).at({0,0,0})= 1;
    }
    std::vector<UniTensor> LR(Nsites+1);
    LR[0]  = L0;
    LR[Nsites] = R0;

    // auto anet = Network();
    // anet.FromString({"L: -2,-1,-3",\
    //                 "A: -1,-4,1",\
    //                 "M: -2,0,-4,-5",\
    //                 "A_Conj: -3,-5,2",\
    //                 "TOUT: 0;1,2"});
    for(int p = 0; p<Nsites-1;p++){
        auto LR_ = LR[p].relabels({"-2","-1","-3"});
        auto A_ = A[p].relabels({"-1","-4","1"});
        auto Ad_ = A[p].Dagger().relabels({"-3","-5","2"});
        auto M_= M.relabels({"-2","0","-4","-5"});
        LR[p+1] = Ad_.contract(M_.contract(A_.contract(LR_,true),true),true).permute({1,2,0});

        // anet.PutUniTensors({"L","A","A_Conj","M"},{LR[p],A[p],A[p].Dagger(),M});
        // LR[p+1] = anet.Launch(true);
    }            

    for(int k = 0; k<numsweeps;k++){
        for(int p = Nsites-2; p>=0; p--){
            auto dim_l = A[p].shape()[0];
            auto dim_r = A[p+1].shape()[2];
            auto psi = cytnx::Contract(A[p],A[p+1]);
            auto optres = optimize_psi(psi, LR[p],M,M,LR[p+2], maxit, krydim);
            psi = optres[1];
            auto lbl1 = A[p].labels();
            auto lbl2 = A[p+1].labels();
            psi.set_rowrank(2);  
            auto svdres = linalg::Svd_truncate(psi, chi);
            auto s = svdres[0];
            A[p] = svdres[1];
            A[p+1] = svdres[2];
            A[p+1].set_labels(lbl2);
            A[p] = cytnx::Contract(A[p],s); //// absorb s into next neighbor
            A[p].set_labels(lbl1);

            // anet.FromString({"R: -2,-1,-3",\
            //                 "B: 1,-4,-1",\
            //                 "M: 0,-2,-4,-5",\
            //                 "B_Conj: 2,-5,-3",\
            //                 "TOUT: 0;1,2"});
            // anet.PutUniTensors({"R","B","M","B_Conj"},{LR[p+2],A[p+1],M,A[p+1].Dagger()});
            // LR[p+1] = anet.Launch(true);

            auto LR_ = LR[p+2].relabels({"-2","-1","-3"});
            auto B_ = A[p+1].relabels({"1","-4","-1"});
            auto Bd_ = A[p+1].Dagger().relabels({"2","-5","-3"});
            auto M_= M.relabels({"0","-2","-4","-5"});
            LR[p+1] = Bd_.contract(M_.contract(B_.contract(LR_,true),true),true).permute({1,2,0});;

            // std::cout<<"Sweep r->l "<<k<<"/"<<numsweeps<<" loc:"<<p<<" Energy:"<<double(optres[0].item().real())<<std::endl;
        }
        
        auto lbl = A[0].labels();
        A[0].set_rowrank(1);
        auto res = linalg::Svd(A[0], false, true);
        A[0] = res[1];
        A[0].set_labels(lbl);

        for(int p = 0; p<Nsites-1; p++){    
            auto dim_l = A[p].shape()[0];
            auto dim_r = A[p+1].shape()[2];
            auto psi = cytnx::Contract(A[p],A[p+1]); //// cytnx::Contract
            auto optres = optimize_psi(psi, LR[p],M,M,LR[p+2], maxit, krydim);
            psi = optres[1];
            auto lbl1 = A[p].labels();
            auto lbl2 = A[p+1].labels();
            psi.set_rowrank(2);
            auto svdres = linalg::Svd_truncate(psi, chi);
            auto s = svdres[0];
            A[p] = svdres[1];
            A[p+1] = svdres[2];
            A[p].set_labels(lbl1);
            A[p+1] = cytnx::Contract(s,A[p+1]); //// absorb s into next neighbor.
            A[p+1].set_labels(lbl2);
            // anet = Network();
            // anet.FromString({"L: -2,-1,-3",\
            //                 "A: -1,-4,1",\
            //                 "M: -2,0,-4,-5",\
            //                 "A_Conj: -3,-5,2",\
            //                 "TOUT: 0;1,2"});
            // anet.PutUniTensors({"L","A","A_Conj","M"},{LR[p],A[p],A[p].Dagger(),M});
            // LR[p+1] = anet.Launch(true);
            auto LR_ = LR[p].relabels({"-2","-1","-3"});
            auto A_ = A[p].relabels({"-1","-4","1"});
            auto Ad_ = A[p].Dagger().relabels({"-3","-5","2"});
            auto M_= M.relabels({"-2","0","-4","-5"});
            LR[p+1] = Ad_.contract(M_.contract(A_.contract(LR_,true),true),true).permute({1,2,0});;
            // std::cout<<"Sweep l->r "<<k<<"/"<<numsweeps<<" loc:"<<p<<" Energy:"<<double(optres[0].item().real())<<std::endl;
        }

        lbl = A[Nsites-1].labels();
        A[Nsites-1].set_rowrank(2);
        res = linalg::Svd(A[Nsites-1],true,false); //// last one.
        A[Nsites-1] = res[1];
        A[Nsites-1].set_labels(lbl);
        // std::cout<<"Done : "<<k<<std::endl;
    }

    // std::cout<<A[Nsites/2].shape()<<std::endl;

	for (auto _: state) {
        for(int p = Nsites-2; p>=0; p--){
            auto dim_l = A[p].shape()[0];
            auto dim_r = A[p+1].shape()[2];
            auto psi = cytnx::Contract(A[p],A[p+1],true,true);
            auto optres = optimize_psi(psi, LR[p],M,M,LR[p+2], maxit, krydim);
            psi = optres[1];
            auto lbl1 = A[p].labels();
            auto lbl2 = A[p+1].labels();
            psi.set_rowrank(2);  
            auto svdres = linalg::Svd_truncate(psi, chi);
            auto s = svdres[0];
            A[p] = svdres[1];
            A[p+1] = svdres[2];
            A[p+1].set_labels(lbl2);
            A[p] = cytnx::Contract(A[p],s,true,true); //// absorb s into next neighbor
            A[p].set_labels(lbl1);
            // anet.FromString({"R: -2,-1,-3",\
            //                 "B: 1,-4,-1",\
            //                 "M: 0,-2,-4,-5",\
            //                 "B_Conj: 2,-5,-3",\
            //                 "TOUT: 0;1,2"});
            // anet.PutUniTensors({"R","B","M","B_Conj"},{LR[p+2],A[p+1],M,A[p+1].Dagger()});
            // LR[p+1] = anet.Launch(true);
            auto LR_ = LR[p+2].relabels({"-2","-1","-3"});
            auto B_ = A[p+1].relabels({"1","-4","-1"});
            auto Bd_ = A[p+1].Dagger().relabels({"2","-5","-3"});
            auto M_= M.relabels({"0","-2","-4","-5"});
            LR[p+1] = Bd_.contract(M_.contract(B_.contract(LR_,true),true),true).permute({1,2,0});;
            // std::cout<<"Sweep r->l "<<k<<"/"<<numsweeps<<" loc:"<<p<<" Energy:"<<double(optres[0].item().real())<<std::endl;
        }  
        
        auto lbl = A[0].labels();
        A[0].set_rowrank(1);
        auto res = linalg::Svd(A[0], false, true);
        A[0] = res[1];
        A[0].set_labels(lbl);

        for(int p = 0; p<Nsites-1; p++){    
            auto dim_l = A[p].shape()[0];
            auto dim_r = A[p+1].shape()[2];
            auto psi = cytnx::Contract(A[p],A[p+1],true,true); //// cytnx::Contract
            auto optres = optimize_psi(psi, LR[p],M,M,LR[p+2], maxit, krydim);
            psi = optres[1];
            auto lbl1 = A[p].labels();
            auto lbl2 = A[p+1].labels();
            psi.set_rowrank(2);
            auto svdres = linalg::Svd_truncate(psi, chi);
            auto s = svdres[0];
            A[p] = svdres[1];
            A[p+1] = svdres[2];
            A[p].set_labels(lbl1);
            A[p+1] = cytnx::Contract(s,A[p+1],true,true); //// absorb s into next neighbor.
            A[p+1].set_labels(lbl2);
            // anet = Network();
            // anet.FromString({"L: -2,-1,-3",\
            //                 "A: -1,-4,1",\
            //                 "M: -2,0,-4,-5",\
            //                 "A_Conj: -3,-5,2",\
            //                 "TOUT: 0;1,2"});
            // anet.PutUniTensors({"L","A","A_Conj","M"},{LR[p],A[p],A[p].Dagger(),M});
            // LR[p+1] = anet.Launch(true);
            auto LR_ = LR[p].relabels({"-2","-1","-3"});
            auto A_ = A[p].relabels({"-1","-4","1"});
            auto Ad_ = A[p].Dagger().relabels({"-3","-5","2"});
            auto M_= M.relabels({"-2","0","-4","-5"});
            LR[p+1] = Ad_.contract(M_.contract(A_.contract(LR_,true),true),true).permute({1,2,0});;
            // std::cout<<"Sweep l->r "<<k<<"/"<<numsweeps<<" loc:"<<p<<" Energy:"<<double(optres[0].item().real())<<std::endl;
        }

        lbl = A[Nsites-1].labels();
        A[Nsites-1].set_rowrank(2);
        res = linalg::Svd(A[Nsites-1],true,false); //// last one.
        A[Nsites-1] = res[1];
        A[Nsites-1].set_labels(lbl);
        // std::cout<<"Done : "<<k<<std::endl;
    }
}

static void itensor_dmrg_U1(benchmark::State& state){
    
    // string infile = argv[1];
    // InputGroup input (infile,"basic");
    // auto qn      = input.getYesNo("quantum_number");
    // auto dims    = read_vector<int> (infile, "bond_dim");
	malloc_trim(0);

    int chi = state.range(0);
    int N = state.range(1);
    int Nsweeps = state.range(2);
    auto qn = true;
    auto sites = SpinHalf(N, {"ConserveQNs",qn}); //make a chain of N spin 1/2's
    auto ampo = AutoMPO(sites);
    for(auto j : range1(N-1))
    {
        ampo += 0.5,"S+",j,"S-",j+1;
        ampo += 0.5,"S-",j,"S+",j+1;
    }
    auto H = toMPO(ampo);
    auto state_ = InitState(sites);
    for(auto i : range1(N))
    {
        if(i%2 == 1) state_.set(i,"Up");
        else         state_.set(i,"Dn");
    }
    auto psi = MPS(state_);
    Real energy;

    auto sweeps = Sweeps(Nsweeps);
    sweeps.maxdim() = chi;
    sweeps.mindim() = chi;
    sweeps.cutoff() = 1E-12;
    sweeps.niter() = 2;
    std::tie(energy,psi) = dmrg(H,psi,sweeps,"Silent");
    auto psit = psi;

    sweeps = Sweeps(1);
    sweeps.maxdim() = chi;
    sweeps.mindim() = chi;
    sweeps.cutoff() = 1E-12;
    sweeps.niter() = 2;
	for (auto _: state) {
        std::tie(energy,psit) = dmrg(H,psit,sweeps,"Silent");
    }
}


// BENCHMARK(cytnx_dmrg_U1)->Args({64,32,5});
BENCHMARK(cytnx_dmrg_U1)->Args({100,32,5});
BENCHMARK(cytnx_dmrg_U1)->Args({200,32,5});
BENCHMARK(cytnx_dmrg_U1)->Args({300,32,7});
BENCHMARK(cytnx_dmrg_U1)->Args({400,32,10});
// BENCHMARK(cytnx_dmrg_U1)->Args({500,32,18});
// BENCHMARK(itensor_dmrg_U1)->Args({64,32,2});
BENCHMARK(itensor_dmrg_U1)->Args({100,32,5});
BENCHMARK(itensor_dmrg_U1)->Args({200,32,5});
BENCHMARK(itensor_dmrg_U1)->Args({300,32,7});
BENCHMARK(itensor_dmrg_U1)->Args({400,32,10});
// BENCHMARK(itensor_dmrg_U1)->Args({500,32,18});
BENCHMARK_MAIN();