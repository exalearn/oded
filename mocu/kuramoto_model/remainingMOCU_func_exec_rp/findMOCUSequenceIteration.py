import time
import numpy as np
from MOCU import *
import argparse

parser = argparse.ArgumentParser(description="MOCU")
parser.add_argument("--arg1", "-a1", help="1st argument")
parser.add_argument("--arg2", "-a2", help="2nd argument")
parser.add_argument("--arg3", "-a3", help="3rd argument")
parser.add_argument("--arg4", "-a4", help="4th argument")
parser.add_argument("--arg5", "-a5", help="5th argument")
parser.add_argument("--arg6", "-a6", help="6th argument")
parser.add_argument("--arg7", "-a7", help="7th argument")
parser.add_argument("--arg8", "-a8", help="8th argument")
parser.add_argument("--arg9", "-a9", help="9th argument")
parser.add_argument("--arg10", "-a10", help="10th argument")
parser.add_argument("--arg11", "-a11", help="11th argument")
parser.add_argument("--arg12", "-a12", help="12th argument")
args = parser.parse_args()
i                    = arg1
j                    = arg2
K_max                = arg3
w                    = arg4
N                    = arg5
h                    = arg6
MVirtual             = arg7
TVirtual             = arg8
aUpperBoundUpdated   = arg9
aLowerBoundUpdated   = arg10
it_idx               = arg11
pseudoRandomSequence = arg12

if (not isInitiallyComputed) or iterative:
    # Computing the expected remaining MOCU
    for i in range(N):
        for j in range(i+1,N):
            isInitiallyComputed = True
            if (i, j) not in optimalExperiments:
                aUpper = aUpperBoundUpdated.copy()
                aLower = aLowerBoundUpdated.copy()

                w_i = w[i]
                w_j = w[j]
                f_inv = 0.5*np.abs(w_i - w_j)

                aLower[i,j] = max(f_inv, aLower[i,j])
                aLower[j,i] = aLower[i,j]

                a_tilde = min(max(f_inv, aLowerBoundUpdated[i, j]), aUpperBoundUpdated[i, j])
                P_syn = (aUpperBoundUpdated[i, j] - a_tilde)/(aUpperBoundUpdated[i, j] - aLowerBoundUpdated[i, j])

                it_temp_val = np.zeros(it_idx)
                for l in range(it_idx):
                    # it_temp_val[l] = MOCU(K_max, w, N, h , MVirtual, TVirtual, aLower, aUpper, ((iteration * N * N * l) + (i*N) + j + 3))
                    it_temp_val[l] = MOCU(K_max, w, N, h , MVirtual, TVirtual, aLower, aUpper, 0)
                # print("     Computation time for the expected remaining MOCU (P_syn): ", i, j, time.time() - ttMOCU)
                MOCU_matrix_syn = np.mean(it_temp_val)

                aUpper = aUpperBoundUpdated.copy()
                aLower = aLowerBoundUpdated.copy()

                aUpper[i, j] = min(f_inv, aUpper[i, j])
                aUpper[j, i] = aUpper[i, j]

                P_nonsyn = (a_tilde - aLowerBoundUpdated[i,j])/(aUpperBoundUpdated[i,j] - aLowerBoundUpdated[i,j])

                it_temp_val = np.zeros(it_idx)
                for l in range(it_idx):
                    # it_temp_val[l] = MOCU(K_max, w, N, h , MVirtual, TVirtual, aLower, aUpper, ((2 * iteration * N * N * l) + (i*N) + j + 2))
                    it_temp_val[l] = MOCU(K_max, w, N, h , MVirtual, TVirtual, aLower, aUpper, 0)
                # print("     Computation time for the expected remaining MOCU (P_nonsyn): ", i, j, time.time() - ttMOCU)
                MOCU_matrix_nonsyn = np.mean(it_temp_val)
                # print(P_syn, MOCU_matrix_syn, P_nonsyn, MOCU_matrix_nonsyn)
    
                # print("i = ",i)
                # print("R = ",R)
                R[i, j] = (P_syn*MOCU_matrix_syn + P_nonsyn*MOCU_matrix_nonsyn)
