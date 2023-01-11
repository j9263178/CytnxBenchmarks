import sys
import numpy as np

def get_inds (indstr):
    inds = list(map(int,indstr.strip('()').split(',')))
    inds = [i-1 for i in inds]
    inds = tuple(inds)
    return inds

def get_inds2 (line):
    return tuple(map(int,line.split('=')[-1].split()))

if __name__ == '__main__':
    fname = sys.argv[1]
    A1s, A2s, Ls, Rs, W1s, W2s, phis = [],[],[],[],[],[],[]
    with open(fname) as f:
        for line in f:
            if line.startswith('dim A1'):
                dims = get_inds2 (line)
                A1s.append (np.zeros(dims))
            elif line.startswith('dim A2'):
                dims = get_inds2 (line)
                A2s.append (np.zeros(dims))
            elif line.startswith('dim L'):
                dims = get_inds2 (line)
                Ls.append (np.zeros(dims))
            elif line.startswith('dim R'):
                dims = get_inds2 (line)
                Rs.append (np.zeros(dims))
            elif line.startswith('dim W1'):
                dims = get_inds2 (line)
                W1s.append (np.zeros(dims))
            elif line.startswith('dim W2'):
                dims = get_inds2 (line)
                W2s.append (np.zeros(dims))
            elif line.startswith('dim psi'):
                dims = get_inds2 (line)
                phis.append (np.zeros(dims))

            elif line.startswith('A1 ='):
                A = A1s[-1]
            elif line.startswith('A2 ='):
                A = A2s[-1]
            elif line.startswith('L() ='):
                A = Ls[-1]
            elif line.startswith('R() ='):
                A = Rs[-1]
            elif line.startswith('(*Op1_) ='):
                A = W1s[-1]
            elif line.startswith('(*Op2_) ='):
                A = W2s[-1]
            elif line.startswith('phi ='):
                A = phis[-1]

            elif line[0] == '(' and line[1].isdigit():
                inds, val = line.split()
                inds = get_inds(inds)
                val = float(val)
                A[inds] = val

            elif 'number of dav' in line or 'davison time' in line or 'H|psi> time' in line or 'contract tim' in line or line.startswith('dim ='):
                print (line.strip())
                if line.startswith('dav'):
                    print()

    print('number of (A1,A2):',len(A1s),len(A2s),len(W1s))
    print(A1s[0])
