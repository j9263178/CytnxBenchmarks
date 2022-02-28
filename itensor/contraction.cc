#include "itensor/all.h"
#include <stdio.h>
#include <iostream>
#include <chrono>

using namespace itensor;

int main(){

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    int dim[11] = {100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000};

    std::vector<double> dense;
    std::vector<double> sparse;

    for (int i = 0; i < 11; i++){
        double sum = 0;
        for (int k = 0; k < 10; k++){
            auto I1 = Index(QN({"q1", 1}),dim[i],
                    QN({"q2", -1}),dim[i],In);
            auto I2 = Index(QN({"q1", 1}),dim[i],
                        QN({"q2", -1}),dim[i],Out);

            auto T1 = randomITensor(QN({"flux", 0}), I1, I2);
            auto T2 = randomITensor(QN({"flux", 0}), I2.dag(), I1.dag());

            auto t1 = high_resolution_clock::now();
            auto T3 = T1*prime(T2, I1);
            auto t2 = high_resolution_clock::now();

            duration<double, std::milli> ms_double = t2 - t1;
            std::cout << "Sparse contraction time : " << ms_double.count() << "ms\n";
            sum += ms_double.count();
        }
        sparse.push_back(sum/10);

        sum = 0;
        for (int k = 0; k < 10; k++){
            auto I1 = Index(2*dim[i]);
            auto I2 = Index(2*dim[i]);
            auto I3 = Index(2*dim[i]);

            auto T1 = randomITensor(I1, I2);
            auto T2 = randomITensor(I2, I3);

            auto t1 = high_resolution_clock::now();
            auto T3 = T1*T2;
            auto t2 = high_resolution_clock::now();

            duration<double, std::milli> ms_double = t2 - t1;
            std::cout << "Dense contraction time : " << ms_double.count() << "ms\n";
            sum += ms_double.count();
        }
        dense.push_back(sum/10);
        
        puts("====================================");
    }
    puts("Sparse contration times:");
    for (auto ele:sparse) std::cout<<ele<<" ,";
    puts("Dense contration times:");
    for (auto ele:dense) std::cout<<ele<<" ,";

    return 0;
}
