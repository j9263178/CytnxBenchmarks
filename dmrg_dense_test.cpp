#include <benchmark/benchmark.h>
#include <array>
#include <itensor/all.h>
#include <cytnx.hpp>
#include <malloc.h>
#define min(a, b) (a < b ? a : b)

using namespace cytnx;
// using namespace itensor;

class Hxx : public LinOp {
 public:
  Network projector;
  UniTensor L, M1, M2, R;
  Hxx(Network projector, UniTensor& L, UniTensor& M1, UniTensor& M2, UniTensor& R)
      : LinOp("mv", 0, Type.Double, Device.cpu) {
    this->projector = projector;
    this->L = L;
    this->M1 = M1;
    this->M2 = M2;
    this->R = R;
  }
  UniTensor matvec(const UniTensor& psi) override {
    auto lbl = psi.labels();
    auto L_ = L.relabels({"-5", "-1", "0"});
    auto R_ = R.relabels({"-7", "-4", "3"});
    auto M1_ = M1.relabels({"-5", "-6", "-2", "1"});
    auto M2_ = M2.relabels({"-6", "-7", "-3", "2"});
    auto psi_ = psi.relabels({"-1", "-2", "-3", "-4"});
    auto out = cytnx::Contract(
      L_,
      cytnx::Contract(M1_, cytnx::Contract(M2_, cytnx::Contract(psi_, R_, true, true), true, true),
                      true, true),
      true, true);
    out.set_labels(lbl);
    return out;
  }
};

static void cytnx_dmrg_dense(benchmark::State& state) {
  malloc_trim(0);

  int chi = state.range(0);  // Maximum allowed bondim (Perform truncation if exceeds)
  int Nsites = state.range(1);  // Number of sites
  int Nsweeps = state.range(2);  // number of DMRG sweeps
  int maxit = 2;  // iterations of Lanczos method
  int krydim = 4;  // dimension of Krylov subspace (Will not be used in Lanczos "Gnd" method)
  int chid = 2;  // 1/2-spin

  auto Sp = zeros({2, 2});
  Sp.at<cytnx_double>(0, 1) = 1;
  auto Sm = zeros({2, 2});
  Sm.at<cytnx_double>(1, 0) = 1;

  auto Si = eye(2);
  auto M_ = zeros({4, 4, chid, chid}, Type.Double);
  M_(0, 0, ":", ":") = Si;
  M_(0, 1, ":", ":") = sqrt(2) * Sm;
  M_(0, 2, ":", ":") = sqrt(2) * Sp;
  M_(1, 3, ":", ":") = sqrt(2) * Sp;
  M_(2, 3, ":", ":") = sqrt(2) * Sm;
  M_(3, 3, ":", ":") = Si;
  auto M = UniTensor(M_, false, 0);
  auto ML = UniTensor(zeros({4, 1, 1}, Type.Double), false, 0);  // left MPO boundary
  auto MR = UniTensor(zeros({4, 1, 1}, Type.Double), false, 0);  // right MPO boundary
  ML.get_block_()(0, 0, 0) = 1;
  MR.get_block_()(3, 0, 0) = 1;
  std::vector<UniTensor> A(Nsites);
  Tensor tempAk = zeros({1, chid, min(chi, chid)});
  int spin = (0 % 2);  // 0 for spin up and 1 for spin down
  tempAk(0, spin, 0) = 1;
  A[0] = UniTensor(tempAk, false, 2);
  for (int k = 1; k < Nsites; k++) {
    int pre = A[k - 1].shape()[2];
    int nxt = min(min(chi, A[k - 1].shape()[2] * chid), pow(chid, (Nsites - k - 1)));
    Tensor tempAk = zeros({pre, chid, nxt});
    int spin = (k % 2);  // 0 for spin up and 1 for spin down
    tempAk(0, spin, 0) = 1;
    A[k] = UniTensor(tempAk, false, 2);
    A[k].set_labels({std::to_string(2 * k), std::to_string(2 * k + 1), std::to_string(2 * k + 2)});
  }
  std::vector<UniTensor> out, svdtemp;
  UniTensor s, u, vT;
  int chil, chir;

  Network L_AMAH, R_AMAH, projector;
  // projector.Fromfile("projector.net");
  // L_AMAH.Fromfile("L_AMAH.net");
  // R_AMAH.Fromfile("R_AMAH.net");

  std::vector<UniTensor> LR(Nsites + 1);
  LR[0] = ML;
  LR[Nsites] = MR;

  // Setup : put MPS into right/left? othogonal form
  for (int p = 0; p < Nsites - 1; p++) {
    // SVD on A[p]
    auto Albl = A[p].labels();
    auto Albl_ = A[p + 1].labels();
    svdtemp = linalg::Svd(A[p]);
    s = svdtemp[0];
    u = svdtemp[1];
    vT = svdtemp[2];
    A[p] = u;
    A[p].set_labels(Albl);
    A[p + 1] = cytnx::Contract(cytnx::Contract(s, vT), A[p + 1]);
    A[p + 1].set_labels(Albl_);
    // L_AMAH.PutUniTensors({"L", "A", "A_Conj", "M"}, {LR[p], A[p], A[p].Conj(), M});
    // LR[p + 1] = L_AMAH.Launch(true);

    auto LR_ = LR[p].relabels({"-2", "-1", "-3"});
    auto A_ = A[p].relabels({"-1", "-4", "1"});
    auto Ad_ = A[p].Conj().relabels({"-3", "-5", "2"});
    auto M_ = M.relabels({"-2", "0", "-4", "-5"});
    LR[p + 1] = Ad_.contract(M_.contract(A_.contract(LR_, true), true), true).permute({1, 2, 0});
  }
  auto Albl = A[Nsites - 1].labels();
  A[Nsites - 1] = linalg::Svd(A[Nsites - 1], true)[1];
  A[Nsites - 1].set_labels(Albl);

  std::vector<cytnx::Scalar> Ekeep(0);
  for (int k = 1; k < Nsweeps + 1; k++) {
    for (int p = Nsites - 2; p > -1; p--) {
      auto psi = cytnx::Contract(A[p], A[p + 1]);
      chil = A[p].shape()[0];
      chir = A[p + 1].shape()[2];
      // projector.PutUniTensors({"L", "M1", "M2", "R"}, {LR[p], M, M, LR[p+2]});
      auto H = Hxx(projector, LR[p], M, M, LR[p + 2]);
      psi.set_rowrank_(0);
      auto res = linalg::Lanczos(&H, psi, "Gnd", 999, maxit, 1, true, false, 0, false);
      Ekeep.push_back(cytnx::Scalar(res[0].item()));
      psi = res[1];
      psi.set_rowrank_(2);
      // int newdim = min(min(chil * chid, chir * chid), chi);
      int newdim = chi;
      svdtemp = linalg::Svd_truncate(psi, newdim);
      s = svdtemp[0];
      s.Div_(s.get_block_().Norm().item());
      u = svdtemp[1];
      vT = svdtemp[2];

      auto Albl = A[p].labels();
      auto Albl_ = A[p + 1].labels();
      A[p] = cytnx::Contract(u, s);
      A[p].set_labels(Albl);
      A[p + 1] = vT;
      A[p + 1].set_labels(Albl_);
      // R_AMAH.PutUniTensors({"R", "B", "M", "B_Conj"}, {LR[p + 2], A[p + 1], M, A[p + 1].Conj()});
      // LR[p + 1] = R_AMAH.Launch(true);
      auto LR_ = LR[p + 2].relabels({"-2", "-1", "-3"});
      auto B_ = A[p + 1].relabels({"1", "-4", "-1"});
      auto Bd_ = A[p + 1].Conj().relabels({"2", "-5", "-3"});
      auto M_ = M.relabels({"0", "-2", "-4", "-5"});
      LR[p + 1] = Bd_.contract(M_.contract(B_.contract(LR_, true), true), true).permute({1, 2, 0});
    }  // end of sweep for
    A[0].set_rowrank_(1);
    Albl = A[0].labels();
    A[0] = linalg::Svd(A[0])[2];  // shape[1,2,2], rowrank = 1
    A[0].set_labels(Albl);
    for (int p = 0; p < Nsites - 1; p++) {
      chil = A[p].shape()[0];
      chir = A[p + 1].shape()[2];
      auto psi = cytnx::Contract(A[p], A[p + 1]);
      // projector.PutUniTensors({"L", "M1", "M2", "R"}, {LR[p], M, M, LR[p+2]});
      auto H = Hxx(projector, LR[p], M, M, LR[p + 2]);
      psi.set_rowrank_(0);
      auto res = linalg::Lanczos(&H, psi, "Gnd", 999, maxit, 1, true, false, 0, false);
      Ekeep.push_back(cytnx::Scalar(res[0].item()));
      psi = res[1];
      psi.set_rowrank_(2);
      // int newdim = min(min(chil * chid, chir * chid), chi);
      int newdim = chi;
      svdtemp = linalg::Svd_truncate(psi, newdim);
      s = svdtemp[0];  // s.Div_(s.get_block_().Norm().item());
      u = svdtemp[1];
      vT = svdtemp[2];
      auto Albl = A[p].labels();
      auto Albl_ = A[p + 1].labels();
      A[p] = u;
      A[p].set_labels(Albl);
      A[p + 1] = cytnx::Contract(s, vT);
      A[p + 1].set_labels(Albl_);
      // L_AMAH.PutUniTensors({"L", "A", "A_Conj", "M"}, {LR[p], A[p], A[p].Conj(), M});
      // LR[p + 1] = L_AMAH.Launch(true);
      auto LR_ = LR[p].relabels({"-2", "-1", "-3"});
      auto A_ = A[p].relabels({"-1", "-4", "1"});
      auto Ad_ = A[p].Conj().relabels({"-3", "-5", "2"});
      auto M_ = M.relabels({"-2", "0", "-4", "-5"});
      LR[p + 1] = Ad_.contract(M_.contract(A_.contract(LR_, true), true), true).permute({1, 2, 0});
    }  // end of iteration for
    Albl = A[Nsites - 1].labels();
    A[Nsites - 1].set_rowrank_(2);
    A[Nsites - 1] = linalg::Svd(A[Nsites - 1], true)[1];  // shape[1,2,2], rowrank = 2
    A[Nsites - 1].set_labels(Albl);
  }  // end of iteration for
  for (auto _ : state) {
    for (int p = Nsites - 2; p > -1; p--) {
      auto psi = cytnx::Contract(A[p], A[p + 1], true, true);
      chil = A[p].shape()[0];
      chir = A[p + 1].shape()[2];
      // projector.PutUniTensors({"L", "M1", "M2", "R"}, {LR[p], M, M, LR[p+2]});
      auto H = Hxx(projector, LR[p], M, M, LR[p + 2]);
      psi.set_rowrank_(0);
      auto res = linalg::Lanczos(&H, psi, "Gnd", 999, maxit, 1, true, false, 0, false);
      Ekeep.push_back(cytnx::Scalar(res[0].item()));
      psi = res[1];
      psi.set_rowrank_(2);
      // int newdim = min(min(chil * chid, chir * chid), chi);
      int newdim = chi;
      svdtemp = linalg::Svd_truncate(psi, newdim);
      s = svdtemp[0];  // s.Div_(s.get_block_().Norm().item());
      u = svdtemp[1];
      vT = svdtemp[2];
      auto Albl = A[p].labels();
      auto Albl_ = A[p + 1].labels();
      A[p] = cytnx::Contract(u, s, true, true);
      A[p].set_labels(Albl);
      A[p + 1] = vT;
      A[p + 1].set_labels(Albl_);
      // R_AMAH.PutUniTensors({"R", "B", "M", "B_Conj"}, {LR[p + 2], A[p + 1], M, A[p + 1].Conj()});
      // LR[p + 1] = R_AMAH.Launch(true);
      auto LR_ = LR[p + 2].relabels({"-2", "-1", "-3"});
      auto B_ = A[p + 1].relabels({"1", "-4", "-1"});
      auto Bd_ = A[p + 1].Conj().relabels({"2", "-5", "-3"});
      auto M_ = M.relabels({"0", "-2", "-4", "-5"});
      LR[p + 1] = Bd_.contract(M_.contract(B_.contract(LR_, true), true), true).permute({1, 2, 0});
    }  // end of sweep for
    A[0].set_rowrank_(1);
    Albl = A[0].labels();
    A[0] = linalg::Svd(A[0], true)[2];  // shape[1,2,2], rowrank = 1
    A[0].set_labels(Albl);
    for (int p = 0; p < Nsites - 1; p++) {
      chil = A[p].shape()[0];
      chir = A[p + 1].shape()[2];
      auto psi = cytnx::Contract(A[p], A[p + 1], true, true);
      // projector.PutUniTensors({"L", "M1", "M2", "R"}, {LR[p], M, M, LR[p+2]});
      auto H = Hxx(projector, LR[p], M, M, LR[p + 2]);
      psi.set_rowrank_(0);
      auto res = linalg::Lanczos(&H, psi, "Gnd", 999, maxit, 1, true, false, 0, false);
      Ekeep.push_back(cytnx::Scalar(res[0].item()));
      psi = res[1];
      psi.set_rowrank_(2);
      // int newdim = min(min(chil * chid, chir * chid), chi);
      int newdim = chi;
      svdtemp = linalg::Svd_truncate(psi, newdim);
      s = svdtemp[0];  // s.Div_(s.get_block_().Norm().item());
      u = svdtemp[1];
      vT = svdtemp[2];
      auto Albl = A[p].labels();
      auto Albl_ = A[p + 1].labels();
      A[p] = u;
      A[p].set_labels(Albl);
      A[p + 1] = cytnx::Contract(s, vT, true, true);
      A[p + 1].set_labels(Albl_);
      // L_AMAH.PutUniTensors({"L", "A", "A_Conj", "M"}, {LR[p], A[p], A[p].Conj(), M});
      // LR[p + 1] = L_AMAH.Launch(true);
      auto LR_ = LR[p].relabels({"-2", "-1", "-3"});
      auto A_ = A[p].relabels({"-1", "-4", "1"});
      auto Ad_ = A[p].Conj().relabels({"-3", "-5", "2"});
      auto M_ = M.relabels({"-2", "0", "-4", "-5"});
      LR[p + 1] = Ad_.contract(M_.contract(A_.contract(LR_, true), true), true).permute({1, 2, 0});
    }  // end of iteration for
    Albl = A[Nsites - 1].labels();
    A[Nsites - 1].set_rowrank_(2);
    A[Nsites - 1] = linalg::Svd(A[Nsites - 1], true)[1];  // shape[1,2,2], rowrank = 2
    A[Nsites - 1].set_labels(Albl);
  }
  cytnx::vec_print(std::cout, Ekeep);
}

static void itensor_dmrg_dense(benchmark::State& state) {
  // string infile = argv[1];
  // InputGroup input (infile,"basic");
  // auto qn      = input.getYesNo("quantum_number");
  // auto dims    = read_vector<int> (infile, "bond_dim");
  malloc_trim(0);
  int chi = state.range(0);
  int N = state.range(1);
  int Nsweeps = state.range(2);
  auto qn = false;
  auto sites = itensor::SpinHalf(N, {"ConserveQNs", qn});  // make a chain of N spin 1/2's
  auto ampo = itensor::AutoMPO(sites);
  for (auto j : itensor::range1(N - 1)) {
    ampo += 0.5, "S+", j, "S-", j + 1;
    ampo += 0.5, "S-", j, "S+", j + 1;
  }
  auto H = itensor::toMPO(ampo);
  auto state_ = itensor::InitState(sites);
  for (auto i : itensor::range1(N)) {
    if (i % 2 == 1)
      state_.set(i, "Up");
    else
      state_.set(i, "Dn");
  }
  auto psi = itensor::MPS(state_);
  itensor::Real energy;

  auto sweeps = itensor::Sweeps(Nsweeps);
  sweeps.maxdim() = chi;
  sweeps.mindim() = chi;
  sweeps.cutoff() = 1E-15;
  sweeps.niter() = 2;
  std::tie(energy, psi) = itensor::dmrg(H, psi, sweeps, {"Silent=", true, "UseSVD=", true});
  auto psit = psi;

  sweeps = itensor::Sweeps(1);
  sweeps.maxdim() = chi;
  sweeps.mindim() = chi;
  sweeps.cutoff() = 1E-15;
  sweeps.niter() = 2;
  for (auto _ : state) {
    std::tie(energy, psit) = itensor::dmrg(H, psit, sweeps, {"Silent=", true, "UseSVD=", true});
  }
}

// BENCHMARK(itensor_dmrg_dense)->Args({200,32,10});
// BENCHMARK(cytnx_dmrg_dense)->Args({100,32,5});
// BENCHMARK(cytnx_dmrg_dense)->Args({200,32,10});

BENCHMARK(cytnx_dmrg_dense)->Args({100, 32, 5});
BENCHMARK(cytnx_dmrg_dense)->Args({200, 32, 5});
BENCHMARK(cytnx_dmrg_dense)->Args({300, 32, 7});
BENCHMARK(cytnx_dmrg_dense)->Args({400, 32, 10});
// BENCHMARK(cytnx_dmrg_dense)->Args({500,32,10});

// BENCHMARK(cytnx_dmrg_dense)->Args({1000,32,10});
// BENCHMARK(cytnx_dmrg_dense)->Args({2000,32,10});
// BENCHMARK(cytnx_dmrg_dense)->Args({3000,32,10});

// BENCHMARK(itensor_dmrg_dense)->Args({100,32,5});
// BENCHMARK(itensor_dmrg_dense)->Args({200,32,5});
// BENCHMARK(itensor_dmrg_dense)->Args({300,32,7});
// BENCHMARK(itensor_dmrg_dense)->Args({400,32,10});
// BENCHMARK(itensor_dmrg_dense)->Args({500,32,10});

BENCHMARK_MAIN();