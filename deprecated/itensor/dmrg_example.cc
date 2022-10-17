#include "itensor/all.h"
#include <iostream>

using namespace itensor;

int main(){
    int N = 16;
    auto sites = SpinOne(N);

    auto ampo = AutoMPO(sites);
    for(int j = 1; j < N; ++j)
        {
        // ampo += "Sz",j,"Sz",j+1;
        ampo += 0.5,"S+",j,"S-",j+1;
        ampo += 0.5,"S-",j,"S+",j+1;
        }
    auto H = toMPO(ampo);

    auto sweeps = Sweeps(5);
    sweeps.maxdim() = 10,40,100,200,200;
    sweeps.cutoff() = 1E-8;

    // Create a random starting state
    auto state = InitState(sites);
    for(auto i : range1(N))
        {
        if(i%2 == 1) state.set(i,"Up");
        else         state.set(i,"Dn");
        }
    auto psi0 = randomMPS(state);

    auto [energy,psi] = dmrg(H,psi0,sweeps,{"Quiet",true});

    printfln("Ground state energy = %.20f",energy);
    return 0;
}
