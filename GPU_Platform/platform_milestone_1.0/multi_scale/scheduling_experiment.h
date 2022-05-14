#include <fstream>
#include <iostream>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <vector>
using namespace std;

#define cudaDebugCall(F)                                                                                               \
  if ((F) != cudaSuccess) {                                                                                            \
    printf("Error at line %d: %s\n", __LINE__, cudaGetErrorString(cudaGetLastError()));                                \
    exit(1);                                                                                                           \
  };

#define debugCall(F)                                                                                                   \
  if ((F) != 0) {                                                                                                      \
    printf("Error at line %d\n", __LINE__);                                                                            \
    exit(1);                                                                                                           \
  };

#define CPU_UNIT_TASK (60500) // parameter to generate a unit cpu task of 1 ms, worst case: 60000 / 1.1
#define GPU_UNIT_TASK (59900) // parameter to generate a unit gpu task of 1 ms, worst case: 46200 / 1.1
#define MAX_CPU_TASK_NUM (51) // max number of cpu tasks in a batch
#define MAX_GPU_TASK_NUM (50) // max number of gpu tasks in a batch
#define PTHREAD_NUM (5)       // number of pthreads

int cpuTaskNum[PTHREAD_NUM] = {9, 12, 20, 23, 51}; // number of cpu tasks in a batch
int gpuTaskNum[PTHREAD_NUM] = {8, 11, 19, 22, 50}; // number of cpu tasks in a batch
float cpuTaskLens[PTHREAD_NUM][MAX_CPU_TASK_NUM];  // lengths of cpu tasks, in ms
float gpuTaskLens[PTHREAD_NUM][MAX_GPU_TASK_NUM];  // lengths of gpu tasks, in ms
float utilRates[PTHREAD_NUM];                      // utility rates of each pthread
float ddls[PTHREAD_NUM];                           // deadline of batch on each pthread, in ms

__device__ float deviceData[PTHREAD_NUM][2048]; // gpu task data preparation
cudaStream_t cudaStreams[PTHREAD_NUM];          // each pthreads has its own cuda stream

int prios[PTHREAD_NUM];                    // priority of each pthread: 0 ~ 99
pthread_t mainThreads[PTHREAD_NUM];        // main threads
pthread_t syncThreads[PTHREAD_NUM];        // threads for cuda synchronization
int timeExceeded = 0;                      // whether there's a batch missing its deadline
pthread_mutex_t syncStartMut[PTHREAD_NUM]; // used to activate cuda synchronization
pthread_mutex_t syncEndMut[PTHREAD_NUM];   // used to wake up main threads

void pthreadDataGen(float totalUtilRate) {
  srand((unsigned)time(NULL));

  // UTIL RATE
  float sumUtilRate = 0;
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    // for practical reasons, one util rate is at most twice of another
    sumUtilRate += utilRates[_tid] = rand() % 101 + 100;
  }
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    utilRates[_tid] /= sumUtilRate / totalUtilRate;
  }

  float C[PTHREAD_NUM] = {0}; // sum cpu task lengths
  float S[PTHREAD_NUM] = {0}; // sum gpu task lengths
  // CPU TASK LENGTHS
  for (int i = 0; i < cpuTaskNum[0]; ++i) {
    C[0] += cpuTaskLens[0][i] = rand() % 10 + 1;
  }
  for (int i = 0; i < cpuTaskNum[1]; ++i) {
    C[1] += cpuTaskLens[1][i] = rand() % 10 + 1;
  }
  for (int i = 0; i < cpuTaskNum[2]; ++i) {
    C[2] += cpuTaskLens[2][i] = rand() % 10 + 1;
  }
  for (int i = 0; i < cpuTaskNum[3]; ++i) {
    C[3] += cpuTaskLens[3][i] = rand() % 10 + 1;
  }
  for (int i = 0; i < cpuTaskNum[4]; ++i) {
    C[4] += cpuTaskLens[4][i] = rand() % 10 + 1;
  }
  // GPU TASK LENGTHS
  for (int i = 0; i < gpuTaskNum[0]; ++i) {
    S[0] += gpuTaskLens[0][i] = rand() % 13 + 1;
  }
  for (int i = 0; i < gpuTaskNum[1]; ++i) {
    S[1] += gpuTaskLens[1][i] = rand() % 46 + 1;
  }
  for (int i = 0; i < gpuTaskNum[2]; ++i) {
    S[2] += gpuTaskLens[2][i] = rand() % 46 + 1;
  }
  for (int i = 0; i < gpuTaskNum[3]; ++i) {
    S[3] += gpuTaskLens[3][i] = rand() % 36 + 1;
  }
  for (int i = 0; i < gpuTaskNum[4]; ++i) {
    S[4] += gpuTaskLens[4][i] = rand() % 19 + 2;
  }

  // DDL
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    ddls[_tid] = (C[_tid] + S[_tid]) / utilRates[_tid];
  }
}

void pthreadDataWrite() {
  ofstream ofstrm;
  ofstrm.open("pthreadData.dat");
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    cout << ddls[_tid] << endl;
    ofstrm << utilRates[_tid] << " " << ddls[_tid] << " ";
    for (int i = 0; i < cpuTaskNum[_tid]; ++i) {
      ofstrm << cpuTaskLens[_tid][i] << " ";
    }
    for (int i = 0; i < gpuTaskNum[_tid]; ++i) {
      ofstrm << gpuTaskLens[_tid][i] << " ";
    }
  }
  ofstrm << endl;
  ofstrm.close();
}

void pthreadDataRead() {
  ifstream ifstrm;
  ifstrm.open("pthreadData.dat");
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    ifstrm >> utilRates[_tid] >> ddls[_tid];
    for (int i = 0; i < cpuTaskNum[_tid]; ++i) {
      ifstrm >> cpuTaskLens[_tid][i];
    }
    for (int i = 0; i < gpuTaskNum[_tid]; ++i) {
      ifstrm >> gpuTaskLens[_tid][i];
    }
  }
  ifstrm.close();
}

void pthreadDataPrint() {
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    cout << "Thread " << _tid << '\n';
    cout << "util = " << utilRates[_tid] << ", ddl = " << ddls[_tid] << '\n';
    cout << "cpu: ";
    for (int i = 0; i < cpuTaskNum[_tid]; ++i) {
      cout << cpuTaskLens[_tid][i] << " ";
    }
    cout << '\n';
    cout << "gpu: ";
    for (int i = 0; i < gpuTaskNum[_tid]; ++i) {
      cout << gpuTaskLens[_tid][i] << " ";
    }
    cout << endl;
  }
}

void prioGen(int nth) {
  vector<vector<int>> prioAllPermu(120);
  vector<int> C(PTHREAD_NUM, 0);
  vector<int> A = {95, 96, 97, 98, 99};
  int i = 0;
  int row = 1;
  prioAllPermu[0] = A;
  while (i < PTHREAD_NUM) {
    if (C[i] < i) {
      if (i / 2 * 2 == i) {
        swap(A[0], A[i]);
      } else {
        swap(A[C[i]], A[i]);
      }
      prioAllPermu[row] = A;
      ++row;
      ++C[i];
      i = 0;
    } else {
      C[i] = 0;
      ++i;
    }
  }
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    prios[_tid] = prioAllPermu[nth][_tid];
  }
}
