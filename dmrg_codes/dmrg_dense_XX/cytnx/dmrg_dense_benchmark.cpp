#include "cytnx.hpp"
#include <cfloat>
#include <ctime>
#include <chrono>
#include <time.h> // clock_gettime 函數所需之標頭檔

#define min(a, b) (a < b ? a : b)

using namespace cytnx;

/*
Reference: https://www.tensors.net
Author: j9263178
*/

struct timespec start, end, start_sweep, end_sweep;
struct timespec temp;
struct timespec diff(struct timespec start, struct timespec end) {
  struct timespec temp;

  if (end.tv_sec - start.tv_sec == 0) {
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  } else {
    temp.tv_nsec = ((end.tv_sec - start.tv_sec) * 1000000000) + end.tv_nsec - start.tv_nsec;
  }

  return temp;
};

class Hxx : public LinOp {
 public:
  Network projector;
  UniTensor L, M1, M2, R;
  int verbose;
  Hxx(Network& projector, int verbose, UniTensor& L, UniTensor& M1, UniTensor& M2, UniTensor& R, int dim)
      : LinOp("mv", dim, Type.Double, Device.cpu) {
    this->projector = projector;
    this->verbose = verbose;
    this->L = L;
    this->M1 = M1;
    this->M2 = M2;
    this->R = R;
  }
  // psi: ;-1,-2,-3,-4
  // L: ;-5,-1,0
  // R: ;-7,-4,3
  // M1: ;-5,-6,-2,1
  // M2: ;-6,-7,-3,2
  // TOUT: ;0,1,2,3
  // ORDER : (L,(M1,(M2,(psi,R))))
  UniTensor matvec(const UniTensor& psi) override {
    struct timespec mv_start1, mv_end1, mv_start2, mv_end2;
    clock_gettime(CLOCK_MONOTONIC, &mv_start1);
    auto lbl = psi.labels();
    // projector.PutUniTensor("psi", psi);
    // std::cout << projector.getOptimalOrder() << std::endl;

    auto L_ = L.relabels({-5,-1,0});
    auto R_ = R.relabels({-7,-4,3});
    auto M1_ = M1.relabels({-5,-6,-2,1});
    auto M2_ = M2.relabels({-6,-7,-3,2});
    auto psi_ = psi.relabels({-1,-2,-3,-4});   
    clock_gettime(CLOCK_MONOTONIC, &mv_start2); 
    auto out = Contract(L_, Contract(M1_, Contract(M2_,Contract(psi_,R_))));
    // std::cout<< out.labels() <<std::endl;
    clock_gettime(CLOCK_MONOTONIC, &mv_end2);
    out.set_labels(lbl);
    clock_gettime(CLOCK_MONOTONIC, &mv_end1);
    if (verbose==1){
      temp = diff(mv_start1, mv_end1);
      std::cout << " mv time 1 : "<< (double) temp.tv_nsec / 1000000000.0<<std::endl;
      temp = diff(mv_start2, mv_end2);
      std::cout << " mv time 2 : "<< (double) temp.tv_nsec / 1000000000.0<<std::endl;
    }
    return out;
  }
};

std::vector<UniTensor> DMRG(std::vector<UniTensor>& A, UniTensor& ML, UniTensor& M, UniTensor& MR,
                            int chi, int Nsweeps, int maxit, int krydim) {
  std::vector<UniTensor> out, svdtemp;
  UniTensor s, u, vT;
  int chid = M.shape()[2];
  int Nsites = A.size();
  int chil, chir;

  Network L_AMAH, R_AMAH, projector;
  projector.Fromfile("projector.net");
  L_AMAH.Fromfile("L_AMAH.net");
  R_AMAH.Fromfile("R_AMAH.net");

  std::vector<UniTensor> LR(Nsites + 1);
  LR[0] = ML;
  LR[Nsites] = MR;

  // Setup : put MPS into right/left? othogonal form
  for (int p = 0; p < Nsites - 1; p++) {
    // SVD on A[p]
    svdtemp = linalg::Svd(A[p]);
    s = svdtemp[0];
    u = svdtemp[1];
    vT = svdtemp[2];

    // A[p+1] absorbs s and vT from A[p]
    A[p] = u;
    A[p + 1] = Contract(Contract(s, vT), A[p + 1]);

    // Calculate and store all the Ls for the right-to-left sweep

    L_AMAH.PutUniTensors({"L", "A", "A_Conj", "M"}, {LR[p], A[p], A[p].Conj(), M});
    LR[p + 1] = L_AMAH.Launch(true);
  }

  // SVD for the right boundary tensor A[Nsites-1], only save U (?)
  A[Nsites - 1] = linalg::Svd(A[Nsites - 1], true, false)[1];

  // std::vector<cytnx_double> Ekeep(0);

  std::vector<Scalar> Ekeep(0);

  for (int k = 1; k < Nsweeps + 2; k++) {
    // Optimization sweep: right-to-left
    printf("\n L <- R \n");
    clock_gettime(CLOCK_MONOTONIC, &start_sweep);
    for (int p = Nsites - 2; p > -1; p--) {

      // A[p] is absorbed to make a two-site update
      chil = A[p].shape()[0];
      chir = A[p + 1].shape()[2];

      int verbose = 1 ? (p==Nsites/2 && chil == chi && chir == chi) : 0;

      clock_gettime(CLOCK_MONOTONIC, &start);
      auto psi = Contract(A[p], A[p + 1]);
      clock_gettime(CLOCK_MONOTONIC, &end);

      if (verbose){
        temp = diff(start, end);
        // std::cout << "contraction time_1 : "<<temp.tv_sec<<std::endl;
        std::cout << "contraction time_2 : "<< (double) temp.tv_nsec / 1000000000.0<<std::endl;
      }

      projector.PutUniTensors({"L", "M1", "M2", "R"}, {LR[p], M, M, LR[p+2]});


      auto H = Hxx(projector, verbose,LR[p], M, M, LR[p + 2], psi.shape()[0]);

      psi.set_rowrank(0);

      clock_gettime(CLOCK_MONOTONIC, &start);
      auto res = linalg::Lanczos_Gnd_Ut(&H, psi, 1e-17, true, true, maxit);
      clock_gettime(CLOCK_MONOTONIC, &end);

      if (verbose){
        temp = diff(start, end);
        // std::cout << "Lanczos time_1 : "<<temp.tv_sec<<std::endl;
        std::cout << "Lanczos time_2 : "<< (double) temp.tv_nsec / 1000000000.0<<std::endl;
      }

      psi = res[1];
      psi.set_rowrank(2);
      Ekeep.push_back(Scalar(res[0].item()));

      // restore MPS via SVD
      int newdim = min(min(chil * chid, chir * chid), chi);
      svdtemp = linalg::Svd_truncate(psi, newdim);
      s = svdtemp[0];
      s.Div_(s.get_block_().Norm().item());
      u = svdtemp[1];
      vT = svdtemp[2];
      A[p] = Contract(u, s);
      A[p + 1] = vT;

      // get the new block Hamiltonian
      R_AMAH.PutUniTensors({"R", "B", "M", "B_Conj"}, {LR[p + 2], A[p + 1], M, A[p + 1].Conj()});
      LR[p + 1] = R_AMAH.Launch(true);

      if(verbose){
        printf("Sweep: %d of %d, Loc: %d, Energy: ", k, Nsweeps, p);
        std::cout << Ekeep[Ekeep.size() - 1] << std::endl;
      }


    }  // end of sweep for

    // SVD for the left boundary tensor, only save vT (?)
    A[0].set_rowrank(1);
    A[0] = linalg::Svd(A[0], false, true)[1];  // shape[1,2,2], rowrank = 1

    clock_gettime(CLOCK_MONOTONIC, &end_sweep);
    auto temp_sweep = diff(start_sweep, end_sweep);
    // std::cout << "Lanczos time_1 : "<<temp.tv_sec<<std::endl;
    std::cout << "sweep time : "<< (double) temp_sweep.tv_nsec / 1000000000.0<<std::endl;

    // Optimization sweep: left-to-right
    printf("\n L -> R \n");
    for (int p = 0; p < Nsites - 1; p++) {
      chil = A[p].shape()[0];
      chir = A[p + 1].shape()[2];
      auto psi = Contract(A[p], A[p + 1]);
      projector.PutUniTensors({"L", "M1", "M2", "R"}, {LR[p], M, M, LR[p+2]});

      auto H = Hxx(projector, 0,LR[p], M, M, LR[p + 2], psi.shape()[0]);
      psi.set_rowrank(0);
      auto res = linalg::Lanczos_Gnd_Ut(&H, psi, 1e-17, true, true, maxit);
      Ekeep.push_back(Scalar(res[0].item()));

      psi = res[1];
      psi.set_rowrank(2);

      int newdim = min(min(chil * chid, chir * chid), chi);
      svdtemp = linalg::Svd_truncate(psi, newdim);

      s = svdtemp[0];
      s.Div_(s.get_block_().Norm().item());
      u = svdtemp[1];
      vT = svdtemp[2];
      A[p] = u;
      A[p + 1] = Contract(s, vT);

      L_AMAH.PutUniTensors({"L", "A", "A_Conj", "M"}, {LR[p], A[p], A[p].Conj(), M});
      LR[p + 1] = L_AMAH.Launch(true);

      //printf("Sweep: %d of %d, Loc: %d, Energy: ", k, Nsweeps, p);
      // std::cout << Ekeep[Ekeep.size() - 1] << std::endl;

    }  // end of iteration for

    // SVD for the right boundary tensor, only save U (?)
    A[Nsites - 1].set_rowrank(2);
    A[Nsites - 1] = linalg::Svd(A[Nsites - 1], true, false)[1];  // shape[1,2,2], rowrank = 2

  }  // end of iteration for

  return out;
}

int main() {
  int Nsites = 32;  // system size
  int chid = 2;  // physical local dimension
  int chi = 64;  // bond dimension
  int Nsweeps = 2;  // number of DMRG sweeps
  int krydim = 4;  // dimension of Krylov subspace
  int maxit = 3;  // iterations of Lanczos method

  // ?
  // std::complex<double> j = (0, 1);
  // auto Sx = physics::spin(0.5,'x');
  // auto Sy = physics::spin(0.5,'y');
  // auto Sp = Sx + j*Sy;
  // auto Sm = Sx - j*Sy;

  auto Sp = zeros({2, 2});
  Sp.at<cytnx_double>(0, 1) = 1;
  auto Sm = zeros({2, 2});
  Sm.at<cytnx_double>(1, 0) = 1;

  auto Si = eye(2);

  auto M = zeros({4, 4, chid, chid}, Type.Double);
  M(0, 0, ":", ":") = Si;
  M(0, 1, ":", ":") = sqrt(2) * Sm;
  M(0, 2, ":", ":") = sqrt(2) * Sp;
  M(1, 3, ":", ":") = sqrt(2) * Sp;
  M(2, 3, ":", ":") = sqrt(2) * Sm;
  M(3, 3, ":", ":") = Si;


  auto M_ = UniTensor(M, false, 0);

  auto ML = UniTensor(zeros({4, 1, 1}, Type.Double), false, 0);  // left MPO boundary
  auto MR = UniTensor(zeros({4, 1, 1}, Type.Double), false, 0);  // right MPO boundary
  ML.get_block_()(0, 0, 0) = 1;
  MR.get_block_()(3, 0, 0) = 1;
  std::vector<UniTensor> A(Nsites);

  Tensor tempAk = zeros({1, chid, min(chi, chid)});
  int spin = (0 % 2); // 0 for spin up and 1 for spin down
  tempAk(0, spin, 0) = 1;
  A[0] = UniTensor(tempAk, false, 2);

  // A[0] = UniTensor(zeros({1, chid, min(chi, chid)}, Type.Double), false, 2);
  // random::Make_normal(A[0].get_block_(), 0, 1);
  // for (int k = 1; k < Nsites; k++) {
  //   int pre = A[k - 1].shape()[2];
  //   int nxt = min(min(chi, A[k - 1].shape()[2] * chid), pow(chid, (Nsites - k - 1)));
  //   A[k] = UniTensor(zeros({pre, chid, nxt}), false, 2);
  //   random::Make_normal(A[k].get_block_(), 0, 1);
  //   A[k].set_labels({2 * k, 2 * k + 1, 2 * k + 2});
  // }

 for (int k = 1; k < Nsites; k++) {
    int pre = A[k - 1].shape()[2];
    int nxt = min(min(chi, A[k - 1].shape()[2] * chid), pow(chid, (Nsites - k - 1)));
    Tensor tempAk = zeros({pre, chid, nxt});
    int spin = (k % 2); // 0 for spin up and 1 for spin down
    tempAk(0, spin, 0) = 1;
    A[k] = UniTensor(tempAk, false, 2);
    A[k].set_labels({2 * k, 2 * k + 1, 2 * k + 2});
  }
  DMRG(A, ML, M_, MR, chi, Nsweeps, maxit, krydim);

  printf("\nDone.\n");
}