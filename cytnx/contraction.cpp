#include "cytnx.hpp"

using namespace cytnx;

int main(){
    int dim = 1000;
    std::vector<std::vector<cytnx_int64>> qns;
    for(int i = 0; i < dim; i++){
        qns.push_back({i});
    }   
    Bond b1 = Bond(dim, BD_KET, qns);
    Bond b2 = Bond(3, BD_KET, {{1},{2},{3}});
    Bond b3 = Bond(3, BD_BRA, {{1},{2},{3}});
    UniTensor T1 = UniTensor({b1, b2, b3}, {0, 1, 2}, 1);
    UniTensor T2 = UniTensor({b2, b3}, {2, 3}, 1);
    Contract(T1, T2);

    return 0;
}
