import os
import sys
import time

sys.path.append("./src")

from findEntropySequence  import *
from findRandomSequence import *
from findIdealSequence import *
from determineSyncTwo import *
from determineSyncN import *

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import radical.pilot as rp
import radical.utils as ru

from radical.pilot import PythonTask
pythontask = PythonTask.pythontask

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

mod = SourceModule("""

// This should be manually changed due to the technical issue in the PyCUDA.
// Well, yes, I am lazy...
#include <stdio.h>

#define N_global 8
#define NUMBER_FEATURES (N_global * N_global)

__device__ int mocu_comp(double *w, double h, int N, int M, double* a)
{
    int D = 0;
    double tol,max_temp,min_temp;
    max_temp = -100.0;
    min_temp = 100.0;
    double pi_n = 3.14159265358979323846;

    double theta[N_global];
    double theta_old[N_global];
    double F[N_global],k1[N_global],k2[N_global],k3[N_global],k4[N_global];
    double diff_t[N_global];
    int i,j,k;
    double t = 0.0;
    double sum_temp;


    for (i=0;i<N;i++){
        theta[i] = 0.0;
        theta_old[i] = 0.0;
        F[i] = 0.0;
        k1[i] = 0.0;
        k2[i] = 0.0;
        k3[i] = 0.0;
        k4[i] = 0.0;
        diff_t[i] = 0.0;
    }

    for (k=0;k<M;k++){


        for (i=0;i<N;i++){

            sum_temp = 0.0;
            for (j=0;j<N;j++){
              sum_temp += a[j*N+i]*sin(theta[j] - theta[i]);

            }
            F[i] = w[i] + sum_temp;
        }

        for(i=0;i<N;i++){
            k1[i] = h*F[i];
            theta[i] = theta_old[i] + k1[i]/2.0;
          }



        for (i=0;i<N;i++){
            sum_temp = 0.0;
            for (j=0;j<N;j++){
              sum_temp += a[j*N+i]*sin(theta[j] - theta[i]);
            }
            F[i] = w[i] + sum_temp;
        }

        for(i=0;i<N;i++){
            k2[i] = h*F[i];
            theta[i] = theta_old[i] + k2[i]/2.0;
          }


        for (i=0;i<N;i++){
            sum_temp = 0.0;
            for (j=0;j<N;j++){
              sum_temp += a[j*N+i]*sin(theta[j] - theta[i]);
            }
            F[i] = w[i] + sum_temp;
         }
        for(i=0;i<N;i++){
            k3[i] = h*F[i];
            theta[i] = theta_old[i] + k3[i];
          }



        for (i=0;i<N;i++){
            sum_temp = 0.0;
            for (j=0;j<N;j++){
              sum_temp += a[j*N+i]*sin(theta[j] - theta[i]);
            }
            F[i] = w[i] + sum_temp;
        }


        for(i=0;i<N;i++){        
            k4[i] = h*F[i];
            theta[i] = theta_old[i] + 1.0/6.0*(k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
          }


        for (i=0;i<N;i++){
            if ((M/2) < k)
            {
             diff_t[i] = (theta[i] - theta_old[i]);
            }

             if (theta[i] > 2.0*pi_n)
             {
          		theta[i] = theta[i] - 2.0*pi_n;
            }

             theta_old[i] = theta[i];  
        }

        if ((M/2) < k){
            for(i=0;i<N;i++){
                if (diff_t[i] > max_temp)
                {
                    max_temp  = diff_t[i];
                }

                if (diff_t[i] < min_temp)
                {
                    min_temp  = diff_t[i];
                }
            }

        }

        t = t+h;

    }


    tol = max_temp-min_temp;
    if (tol <= 0.001){
        D = 1;
    }

    return D;
}

__global__ void task(double *a, double *random_data, double *a_save, double *w, \
                     double h , int N, int M, double *a_lower_bound_update, \
                    double *a_upper_bound_update)
{
    const int i_c = blockDim.x*blockIdx.x + threadIdx.x;
    int i,j;
    int observeIndex = 10000000000;

    double a_new[N_global*N_global];
    for (i=0;i<N_global*N_global;i++){
            a_new[i] = 0.0;
    }
    if (i_c == observeIndex) {
        printf("find minimum cost %d", i_c);
            for (i=0;i<N_global*N_global;i++){
            if ((i%N_global) == 0) {
                printf("\\n");
            }
            printf("a_new[%d]=%.10f\\t", i, a_new[i]);
        }
        printf("\\n");
    }
    int rand_ind, cnt0, cnt1;

    cnt0 = (i_c*(N-1)*N/2);
    cnt1 = 0;

    for (i=0;i<N;i++){
        for (j=i+1;j<N;j++)
        {
            rand_ind = cnt0 + cnt1;
            a_new[j*(N+1)+i] = a_lower_bound_update[(j*N)+i]+ (a_upper_bound_update[(j*N)+i]-a_lower_bound_update[(j*N)+i])*random_data[rand_ind];
            a_new[i*(N+1)+j] = a_new[j*(N+1)+i];
            cnt1++;
        }
    }

    if (i_c == observeIndex) {
        printf("Initialization of a_new", i_c);
            for (i=0;i<N_global*N_global;i++){
                            if ((i%N_global) == 0) {
                printf("\\n");
            }
            printf("a_new[%d]=%.10f\\t", i, a_new[i]);
        }
        printf("\\n");
    }
    bool isFound = 0;
    int D;
    int iteration;
    double initialC = 0;

    for (iteration = 1; iteration < 100; iteration++) {
        initialC = 2 * iteration;
        for (i=0;i<N;i++){
            a_new[(i*(N+1))+N] = initialC;
            a_new[(N*(N+1))+i] = initialC;
        }

        if (i_c == observeIndex) {
        printf("Find upper bound, iteration: %d, upperbound: %.10f", iteration, initialC);
            for (i=0;i<N_global*N_global;i++){
                            if ((i%N_global) == 0) {
                printf("\\n");
            }
            printf("a_new[%d]=%.10f\\t", i, a_new[i]);
            }
            printf("\\n");
        }
        D = mocu_comp(w, h, N+1, M, a_new);

        if (D > 0) {
            isFound = 1;
            break;
        }
    }

    double c_lower = 0.0;
    double c_upper = initialC;
    double midPoint = 0;
    int iterationOffset = iteration - 1;

    if (isFound > 0) {
        for (iteration = 0; iteration < (14 + iterationOffset); iteration++) {
            midPoint = (c_upper + c_lower) / 2.0;

            for (i=0;i<N;i++){
                a_new[(i*(N+1))+N] = midPoint;
                a_new[(N*(N+1))+i] = midPoint;
            }
            if (i_c == observeIndex) {
            printf("binary serach, iteration: %d, upper bound: %.10f, lower bound: %.10f", iteration, c_upper, c_lower);
                for (i=0;i<N_global*N_global;i++){
                    if ((i%N_global) == 0) {
                        printf("\\n");
                    }
                    printf("a_new[%d]=%.10f\\t", i, a_new[i]);
                }
                printf("\\n");
            }
            D = mocu_comp(w, h, N+1, M, a_new);

            if (D > 0) {  
                c_upper = midPoint;
            }
            else {  
                c_lower = midPoint;
            }

            if ((c_upper - c_lower) < 0.00025) {
                //printf("Upper - Lower is less than 0.00025\\n");
                break;
            }
        }
        a_save[i_c] = c_upper; 
    }
    else {
        printf("Can't find a! i_c: %d\\n", i_c);
        a_save[i_c] = -1; 
    }    
    if (i_c == observeIndex) {
        printf("binary serach end, iteration: %d, upper bound: %.10f, lower bound: %.10f", iteration, c_upper, c_lower);
        for (i=0;i<N_global*N_global;i++){
            if ((i%N_global) == 0) {
                printf("\\n");
            }
            printf("a_new[%d]=%.10f\\t", i, a_new[i]);
        }
        printf("\\n");
    }
}
"""
                   )

task = mod.get_function("task")

def MOCU_orig(K_max, w, N, h , M, T, aLowerBoundIn, aUpperBoundIn, seed):
    # seed = 0
    blocks = 128
    block_size = np.int(K_max/blocks)

    w = np.append(w, np.mean(w))

    a_save = np.zeros(K_max).astype(np.float64)

    vec_a_lower = np.zeros(N*N).astype(np.float64)
    vec_a_upper = np.zeros(N*N).astype(np.float64)

    vec_a_lower = np.reshape(aLowerBoundIn.copy(), N*N)
    vec_a_upper = np.reshape(aUpperBoundIn.copy(), N*N)

    a = np.zeros((N+1)*(N+1)).astype(np.float64)

    if (int(seed) == 0):
        rand_data = np.random.random(int((N-1)*N/2.0*K_max)).astype(np.float64)
    else:
        rand_data = np.random.RandomState(int(seed)).uniform(size = int((N-1)*N/2.0*K_max))

    task(drv.In(a), drv.In(rand_data), drv.Out(a_save), drv.In(w), 
        np.float64(h), np.intc(N), np.intc(M), drv.In(vec_a_lower), 
        drv.In(vec_a_upper), grid=(blocks,1), block=(block_size,1,1))

    # print("a_save")
    # print(a_save)

    if min(a_save) == -1:
        print("Non sync case exists")
        return -1
    
    if K_max >= 1000:
        temp = np.sort(a_save)
        ll = int(K_max*0.005)
        uu = int(K_max*0.995)
        a_save = temp[ll-1:uu]
        a_star = max(a_save)
        MOCU_val = sum(a_star - a_save)/(K_max*0.99)

    else:
        a_star = max(a_save)
        MOCU_val = sum(a_star - a_save)/(K_max)

    return MOCU_val

'''@pythontask
def MOCU_RP(K_max, w, N, h, M, T, aLowerBoundIn, aUpperBoundIn, seed):

    import numpy as np
    sys.path.append("/gpfs/alpine/csc299/scratch/litan/MOCU/Byung-Jun/new/ExaLearn-ODED-Kuramoto-main/N7")
    from MOCU import task#, mocu_comp
    import pycuda.driver as drv

    # seed = 0
    blocks = 128
    block_size = np.int(K_max / blocks)

    w = np.append(w, np.mean(w))

    a_save = np.zeros(K_max).astype(np.float64)

    vec_a_lower = np.zeros(N * N).astype(np.float64)
    vec_a_upper = np.zeros(N * N).astype(np.float64)

    vec_a_lower = np.reshape(aLowerBoundIn.copy(), N * N)
    vec_a_upper = np.reshape(aUpperBoundIn.copy(), N * N)

    a = np.zeros((N + 1) * (N + 1)).astype(np.float64)

    if (int(seed) == 0):
        rand_data = np.random.random(int((N - 1) * N / 2.0 * K_max)).astype(np.float64)
    else:
        rand_data = np.random.RandomState(int(seed)).uniform(size=int((N - 1) * N / 2.0 * K_max))

    task(drv.In(a), drv.In(rand_data), drv.Out(a_save), drv.In(w),
         np.float64(h), np.intc(N), np.intc(M), drv.In(vec_a_lower),
         drv.In(vec_a_upper), grid=(blocks, 1), block=(block_size, 1, 1))

    # print("a_save")
    # print(a_save)

    if min(a_save) == -1:
        print("Non sync case exists")
        return -1

    if K_max >= 1000:
        temp = np.sort(a_save)
        ll = int(K_max * 0.005)
        uu = int(K_max * 0.995)
        a_save = temp[ll - 1:uu]
        a_star = max(a_save)
        MOCU_val = sum(a_star - a_save) / (K_max * 0.99)

    else:
        a_star = max(a_save)
        MOCU_val = sum(a_star - a_save) / (K_max)

    return MOCU_val'''

@pythontask
def computeExpectedRemainingMOCU(i, j, K_max, w, N, h, MVirtual, TVirtual, aUpperBoundUpdated, aLowerBoundUpdated,
                                 it_idx, pseudoRandomSequence):
    import numpy as np
    sys.path.append("/gpfs/alpine/csc299/scratch/litan/MOCU/Byung-Jun/new/ExaLearn-ODED-Kuramoto-main/N7")
    from MOCU import MOCU, task#, mocu_comp

    aUpper = aUpperBoundUpdated.copy()
    aLower = aLowerBoundUpdated.copy()

    w_i = w[i]
    w_j = w[j]
    f_inv = 0.5 * np.abs(w_i - w_j)

    aLower[i, j] = max(f_inv, aLower[i, j])
    aLower[j, i] = aLower[i, j]

    a_tilde = min(max(f_inv, aLowerBoundUpdated[i, j]), aUpperBoundUpdated[i, j])
    P_syn = (aUpperBoundUpdated[i, j] - a_tilde) / (aUpperBoundUpdated[i, j] - aLowerBoundUpdated[i, j])

    it_temp_val = np.zeros(it_idx)

    for l in range(it_idx):
        # it_temp_val[l] = MOCU(K_max, w, N, h , MVirtual, TVirtual, aLower, aUpper, ((iteration * N * N * l) + (i*N) + j + 3))
        it_temp_val[l] = MOCU(K_max, w, N, h, MVirtual, TVirtual, aLower, aUpper, 0)

    # print("     Computation time for the expected remaining MOCU (P_syn): ", i, j, time.time() - ttMOCU)
    MOCU_matrix_syn = np.mean(it_temp_val)

    aUpper = aUpperBoundUpdated.copy()
    aLower = aLowerBoundUpdated.copy()

    aUpper[i, j] = min(f_inv, aUpper[i, j])
    aUpper[j, i] = aUpper[i, j]

    P_nonsyn = (a_tilde - aLowerBoundUpdated[i, j]) / (aUpperBoundUpdated[i, j] - aLowerBoundUpdated[i, j])

    it_temp_val = np.zeros(it_idx)

    for l in range(it_idx):
        # it_temp_val[l] = MOCU(K_max, w, N, h , MVirtual, TVirtual, aLower, aUpper, ((2 * iteration * N * N * l) + (i*N) + j + 2))
        it_temp_val[l] = MOCU(K_max, w, N, h, MVirtual, TVirtual, aLower, aUpper, 0)

    # print("     Computation time for the expected remaining MOCU (P_nonsyn): ", i, j, time.time() - ttMOCU)
    MOCU_matrix_nonsyn = np.mean(it_temp_val)
    # print(P_syn, MOCU_matrix_syn, P_nonsyn, MOCU_matrix_nonsyn)

    # print("i = ",i)
    # print("R = ",R)
    return (P_syn * MOCU_matrix_syn + P_nonsyn * MOCU_matrix_nonsyn)

def findMOCUSequence(syncThresholds, isSynchronized, MOCUInitial, K_max, w, N, h, MVirtual, MReal, TVirtual, TReal,
                     aLowerBoundIn, aUpperBoundIn, it_idx, update_cnt, iterative=True):
    pseudoRandomSequence = True

    MOCUCurve = np.ones(update_cnt + 1) * 50.0
    MOCUCurve[0] = MOCUInitial
    timeComplexity = np.ones(update_cnt)

    aUpperBoundUpdated = aUpperBoundIn.copy()
    aLowerBoundUpdated = aLowerBoundIn.copy()

    optimalExperiments = []
    isInitiallyComputed = False
    isInitiallyComputed2 = False
    R = np.zeros((N, N))

    for iteration in range(1, update_cnt+1):

        tds = list()

        ExprCount = 0
        iterationStartTime = time.time()
        if (not isInitiallyComputed) or iterative:
            # Computing the expected remaining MOCU
            for i in range(N):
                for j in range(i + 1, N):
                    isInitiallyComputed = True
                    if (i, j) not in optimalExperiments:
                        '''MOCU_RP_starttime = time.time()
                        R[i, j] = computeExpectedRemainingMOCU(i, j, K_max, w, N, h, MVirtual, TVirtual, aUpperBoundUpdated, aLowerBoundUpdated, it_idx, pseudoRandomSequence)
                        MOCU_RP_endtime = time.time() - MOCU_RP_starttime
                        print("FUNC_runtime: ", MOCU_RP_endtime)'''
                        td = rp.TaskDescription()
                        td.pre_exec = []
                        td.executable = computeExpectedRemainingMOCU(i, j, K_max, w, N, h, MVirtual, TVirtual, aUpperBoundUpdated, aLowerBoundUpdated, it_idx, pseudoRandomSequence)
                        td.arguments = []
                        td.gpu_processes = 1
                        td.cpu_processes = 1
                        td.cpu_threads = 1
                        td.cpu_process_type = rp.FUNC
                        tds.append(td)
                        report.progress()
                        ExprCount += 1
        # print("Computed erMOCU")
        # print(R)

        if (not isInitiallyComputed) or (not isInitiallyComputed2) or iterative:
            report.progress_done()
            tasks = tmgr.submit_tasks(tds)
            report.header('gather results')
            tmgr.wait_tasks()

        print("Num_remaining_MOCU: ", ExprCount)
        ExprCountConst = ExprCount
        if (not isInitiallyComputed2) or iterative:
            # Computing the expected remaining MOCU
            for i in range(N):
                for j in range(i + 1, N):
                    isInitiallyComputed2 = True
                    if (i, j) not in optimalExperiments:
                        R[i, j] = tasks[ExprCountConst-ExprCount].stdout
                        ##print(tasks[ExprCountConst-ExprCount].stdout, i, j)
                        ExprCount -= 1

        min_ind = np.where(R == np.min(R[np.nonzero(R)]))

        if len(min_ind[0]) == 1:
            min_i_MOCU = int(min_ind[0])
            min_j_MOCU = int(min_ind[1])
        else:
            min_i_MOCU = int(min_ind[0][0])
            min_j_MOCU = int(min_ind[1][0])

        iterationTime = time.time() - iterationStartTime
        timeComplexity[iteration - 1] = iterationTime

        optimalExperiments.append((min_i_MOCU, min_j_MOCU))
        # print("selected experiment: ", min_i_MOCU, min_j_MOCU, "R: ", R[min_i_MOCU, min_j_MOCU])
        R[min_i_MOCU, min_j_MOCU] = 0.0
        f_inv = syncThresholds[min_i_MOCU, min_j_MOCU]

        if isSynchronized[min_i_MOCU, min_j_MOCU] == 0.0:
            aUpperBoundUpdated[min_i_MOCU, min_j_MOCU] \
                = min(aUpperBoundUpdated[min_i_MOCU, min_j_MOCU], f_inv)
            aUpperBoundUpdated[min_j_MOCU, min_i_MOCU] \
                = min(aUpperBoundUpdated[min_i_MOCU, min_j_MOCU], f_inv)
        else:
            aLowerBoundUpdated[min_i_MOCU, min_j_MOCU] \
                = max(aLowerBoundUpdated[min_i_MOCU, min_j_MOCU], f_inv)
            aLowerBoundUpdated[min_j_MOCU, min_i_MOCU] \
                = max(aLowerBoundUpdated[min_i_MOCU, min_j_MOCU], f_inv)

        print("Iteration: ", iteration, " (", update_cnt, "), selected: (", min_i_MOCU, min_j_MOCU, ")", iterationTime,
              "seconds")
        # print("aUpperBoundUpdated")
        # print(aUpperBoundUpdated)
        # print("aLowerBoundUpdated")
        # print(aLowerBoundUpdated)

        # cnt = 0
        # while MOCUCurve[iteration] > MOCUCurve[iteration - 1]:
        #     if cnt == 5:
        #         MOCUCurve[iteration] = MOCUCurve[iteration - 1]
        #         break

        #     it_temp_val = np.zeros(it_idx)
        #     for l in range(it_idx):
        #         it_temp_val[l] = MOCU(K_max, w, N, h , MReal, TReal, aLowerBoundUpdated, aUpperBoundUpdated, ((iteration * N * N * N) * cnt + l))
        #     MOCUCurve[iteration] = np.mean(it_temp_val)
        #     cnt = cnt + 1
        it_temp_val = np.zeros(it_idx)

        for l in range(it_idx):
            # it_temp_val[l] = MOCU(K_max, w, N, h , MReal, TReal, aLowerBoundUpdated, aUpperBoundUpdated, ((iteration * N * N * N) + l))
            it_temp_val[l] = MOCU_orig(K_max, w, N, h, MReal, TReal, aLowerBoundUpdated, aUpperBoundUpdated, 0)

        MOCUCurve[iteration] = np.mean(it_temp_val)
        print("before adjusting")
        print(MOCUCurve[iteration])
        if MOCUCurve[iteration] > MOCUCurve[iteration - 1]:
            MOCUCurve[iteration] = MOCUCurve[iteration - 1]
        print("The end of iteration: actual MOCU", MOCUCurve[iteration])

    print(optimalExperiments)
    return MOCUCurve, optimalExperiments, timeComplexity

if __name__ == '__main__':

    it_idx = 10
    N = 7
    update_cnt = int((N * (N-1))/2)
    K_max = 20480

    deltaT = 1.0/160.0
    TVirtual = 5
    MVirtual = int(TVirtual/deltaT)
    TReal = 5
    MReal = int(TReal/deltaT)

    inputPath = './uncertaintyClass/'
    w = np.loadtxt(inputPath + 'naturalFrequencies.txt')

    listMethods = ['iODE', 'ODE', 'RANDOM', 'ENTROPY']
    numberOfSimulationsPerMethod = 1#100
    numberOfVaildSimulations = 0
    numberOfSimulations = 0

    aInitialUpper = np.loadtxt(inputPath + 'upper.txt')
    aInitialLower = np.loadtxt(inputPath + 'lower.txt')

    np.savetxt('./results/paramNaturalFrequencies.txt', w, fmt='%.64e')
    np.savetxt('./results/paramInitialUpper.txt', aInitialUpper, fmt='%.64e')
    np.savetxt('./results/paramInitialLower.txt', aInitialLower, fmt='%.64e')

    while (numberOfSimulationsPerMethod > numberOfVaildSimulations):
        randomState = np.random.RandomState(int(numberOfSimulations))
        a = np.zeros((N,N))
        for i in range(N):
            for j in range(i+1,N):
                randomNumber = randomState.uniform()
                a[i,j] = aInitialLower[i,j] + randomNumber*(aInitialUpper[i,j] - aInitialLower[i,j])
                a[j,i] = a[i,j]

        numberOfSimulations += 1

        init_sync_check = determineSyncN(w, deltaT, N, MReal, a)

        if init_sync_check == 1:
            print('             The system has been already stable.')
            continue
        else:
            print('             Unstable system has been found')

        isSynchronized = np.zeros((N,N))
        criticalK = np.zeros((N,N))

        for i in range(N):
            for j in range(i+1,N):
                w_i = w[i]
                w_j = w[j]
                a_ij = a[i,j]
                syncThreshold = 0.5*np.abs(w_i - w_j)
                criticalK[i, j] = syncThreshold
                criticalK[j, i] = syncThreshold
                isSynchronized[i,j] = determineSyncTwo(w_i, w_j, deltaT, 2, MReal, a_ij)

        np.savetxt('./results/paramCouplingStrength' + str(numberOfVaildSimulations) + '.txt', a, fmt='%.64e')
        test = np.loadtxt('./results/paramCouplingStrength' + str(numberOfVaildSimulations) + '.txt')

        report = ru.Reporter(name='radical.pilot')
        report.title('An HPC Workflow for Finding MOCU Sequence on GPU')
        session = rp.Session()
        report.header('submit pilots')
        pmgr = rp.PilotManager(session=session)
        n_nodes = math.ceil(float(int(update_cnt)/6))
        pd_init = {'resource'     : 'ornl.summit',
                   'runtime'      : 120,
                   'exit_on_error': True,
                   'project'      : 'CSC299',
                   'queue'        : 'debug',
                   'acess_scheme' : 'local',
                   'cores'        : 168 * n_nodes,
                   'gpus'         : 6 * n_nodes
                  }
        pdesc = rp.PilotDescription(pd_init)
        pilot = pmgr.submit_pilots(pdesc)
        report.header('submit tasks')
        tmgr = rp.TaskManager(session=session)
        tmgr.add_pilots(pilot)
        report.progress_tgt(update_cnt, label='create')
        ##tds = list()

        for indexMethod in range(len(listMethods)):     

            timeMOCU = time.time()

            MOCUInitial = MOCU_orig(K_max, w, N, deltaT, MReal, TReal, aInitialLower.copy(), aInitialUpper.copy(), 0)

            print("Round: ", numberOfVaildSimulations, "/", numberOfSimulationsPerMethod, "-", listMethods[indexMethod], "Iteration: ", numberOfVaildSimulations, " Initial MOCU: ", MOCUInitial, " Computation time: ", time.time() - timeMOCU)
            aUpperUpdated = aInitialUpper.copy()
            aLowerUpdated = aInitialLower.copy()
            if listMethods[indexMethod] == 'RANDOM':
                MOCUCurve, experimentSequence, timeComplexity = findRandomSequence(criticalK, isSynchronized, MOCUInitial, K_max, w, N, deltaT, MReal, TReal, aLowerUpdated, aUpperUpdated, it_idx, update_cnt)
            elif listMethods[indexMethod] == 'ENTROPY':
                MOCUCurve, experimentSequence, timeComplexity = findEntropySequence(criticalK, isSynchronized, MOCUInitial, K_max, w, N, deltaT, MReal, TReal, aLowerUpdated, aUpperUpdated, it_idx, update_cnt)
            elif listMethods[indexMethod] == 'Ideal':
                MOCUCurve, experimentSequence, timeComplexity = findIdealSequence(criticalK, isSynchronized, MOCUInitial, K_max, w, N, deltaT, MVirtual, MReal, TVirtual, TReal, aLowerUpdated, aUpperUpdated, it_idx, update_cnt, iterative = True)
            else:
                if listMethods[indexMethod] == 'iODE':
                    iterative = True
                else:
                    iterative = False
                print("iterative: ", iterative)
                MOCUCurve, experimentSequence, timeComplexity = findMOCUSequence(criticalK, isSynchronized, MOCUInitial, K_max, w, N, deltaT, MVirtual, MReal, TVirtual, TReal, aLowerUpdated, aUpperUpdated, it_idx, update_cnt, iterative = iterative)

            outMOCUFile = open('./results/' + listMethods[indexMethod] + '_MOCU.txt', 'a')
            outTimeFile = open('./results/' + listMethods[indexMethod] + '_timeComplexity.txt', 'a')
            outSequenceFile = open('./results/' + listMethods[indexMethod] + '_sequence.txt', 'a')
            np.savetxt(outMOCUFile, MOCUCurve.reshape(1, MOCUCurve.shape[0]), delimiter = "\t")
            np.savetxt(outTimeFile, timeComplexity.reshape(1, timeComplexity.shape[0]), delimiter = "\t")
            np.savetxt(outSequenceFile, experimentSequence, delimiter = "\t")
            outMOCUFile.close()
            outTimeFile.close()
            outSequenceFile.close()

        report.header('finalize')
        session.close(download=True)

        numberOfVaildSimulations += 1
