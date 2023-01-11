//
// 2019 Many Electron Collaboration Summer School
// ITensor Tutorial
//

#include "itensor/all.h"
#include "itensor/util/print_macro.h"
#include <ctime>
#include <time.h> // clock_gettime 函數所需之標頭檔
#include "ReadInput.h"
using namespace std;
using namespace itensor;

struct timespec t_start, t_end;

struct timespec diff(struct timespec start, struct timespec end) {
  struct timespec temp;

  if (end.tv_sec - start.tv_sec == 0) {
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  } else {
    temp.tv_nsec = ((end.tv_sec - start.tv_sec) * 1000000000) + end.tv_nsec - start.tv_nsec;
  }

  return temp;
};

void print_dim (std::string name, const ITensor& A)
{
    cout << "dim " << name << " = ";
    for(auto ii : A.inds())
        cout << dim(ii) << " ";
    cout << endl;
}
#include "dmrg.h"

int main(int argc, char* argv[])
{



    string infile = argv[1];
    InputGroup input (infile,"basic");
    auto qn      = input.getYesNo("quantum_number");
    auto dims    = read_vector<int> (infile, "bond_dim");

    int N = 32;
    auto sites = SpinHalf(N, {"ConserveQNs",qn}); //make a chain of N spin 1/2's

    auto ampo = AutoMPO(sites);
    for(auto j : range1(N-1))
    {
        ampo += 0.5,"S+",j,"S-",j+1;
        ampo += 0.5,"S-",j,"S+",j+1;
        //ampo +=     "Sz",j,"Sz",j+1;
    }
    auto H = toMPO(ampo);

    auto state = InitState(sites);
    for(auto i : range1(N))
    {
        if(i%2 == 1) state.set(i,"Up");
        else         state.set(i,"Dn");
    }
    auto psi = MPS(state);
    Real energy;

    //
    // Begin the DMRG calculation
    //
    vector<ITensor> A1s, A2s;
    for(int dim : dims)
    {
        auto sweeps = Sweeps(5);
        sweeps.maxdim() = dim;
        sweeps.mindim() = dim;
        sweeps.cutoff() = 1E-12;
        sweeps.niter() = 2;
        tie (energy,psi) = dmrg(H,psi,sweeps,"Quiet");


        // ===============================================================

        auto A1 = psi(N/2);
        auto A2 = psi(N/2+1);
        // Check the dimension
        int dim1 = maxDim (A1);
        int dim2 = maxDim (A2);
        if (dim1 != dim or dim2 != dim)
        {
            cout << "dim not match: " << dim << ", " << dim1 << " " << dim2 << endl;
        }
        else
        {
            cout << "dim = " << dim << endl;
        }
        A1s.push_back (A1);
        A2s.push_back (A2);

        auto psit = psi;
        tie (energy,psit) = my::dmrg(H,psit,sweeps,"Quiet");
    }

    for(int i = 0; i < A1s.size(); i++)
    {
        auto A1 = A1s.at(i);
        auto A2 = A2s.at(i);

        // print_dim ("A1", A1);
        // print_dim ("A2", A2);

        // // Contract
        // clock_t t1 = clock();
        // auto AA = A1 * A2;
        // clock_t t2 = clock();
        // auto dt = double(t2 - t1) / CLOCKS_PER_SEC;
        // cout << "===========================================" << endl;
        // cout << "contract time (sec) = " << dt << endl;
        // cout << "===========================================" << endl;

        // //

        // Contract 
        clock_gettime(CLOCK_MONOTONIC, &t_start);
        auto AA = A1 * A2;
        // 計算結束時間
        clock_gettime(CLOCK_MONOTONIC, &t_end);

        // 計算實際花費時間
        auto temp = diff(t_start, t_end);
        cout << "===========================================" << endl;
        cout << "contract time (sec) = " <<  (double) temp.tv_nsec / 1000000000.0 << endl;
        cout << "===========================================" << endl;
        
        // Print tensor elements
        // PrintData(A1);
        // PrintData(A2);
    }

    return 0;
}
