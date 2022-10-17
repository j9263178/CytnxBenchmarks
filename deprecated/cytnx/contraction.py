import cytnx
import time

dims = [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
sparse, dense = [], []
for dim in dims:
    qns = []
    for i in range(dim):
        qns.append([1])
        qns.append([-1])
        
    b1 = cytnx.Bond(2*dim, cytnx.BD_KET, qns)
    b2 = cytnx.Bond(2*dim, cytnx.BD_KET, qns)

    T1 = cytnx.UniTensor([b1, b2.redirect()], labels = [0, 1], rowrank = 1)
    T2 = cytnx.UniTensor([b1, b2.redirect()], labels = [1, 2], rowrank = 1)

    sumt = 0
    for k in range(10):
        st = time.time()
        _ = cytnx.Contract(T1, T2)
        en = time.time()
        print("Sparse contraction time : ", en-st)
        sumt += (en-st)
    sparse.append(sumt/10)
    
    T3 = cytnx.UniTensor(cytnx.zeros([2*dim, 2*dim]), rowrank=1)
    T3 = T3.relabels([0, 1])
    T4 = cytnx.UniTensor(cytnx.zeros([2*dim, 2*dim]), rowrank=1)
    T4 = T4.relabels([1, 2])

    sumt = 0
    for k in range(10):
        st = time.time()
        _ = cytnx.Contract(T3, T4)
        en = time.time()
        print("Dense contraction time : ", en-st)
        sumt += (en-st)
    dense.append(sumt/10)

    print("="*20)

print("Sparse contraction times : ", sparse)
print("Dense contraction times : ", dense)
