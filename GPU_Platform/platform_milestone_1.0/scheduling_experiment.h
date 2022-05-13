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

/* === A program with multiple cpu & gpu tasks activated in each pthread === */

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
#define MAX_CPU_TASK_NUM (10) // max number of cpu tasks in a batch
#define MAX_GPU_TASK_NUM (10) // max number of gpu tasks in a batch
#define PTHREAD_NUM (5)       // number of pthreads

int cpuTaskNum;                                   // number of cpu tasks in a batch
int gpuTaskNum;                                   // number of gpu tasks in a batch
float cpuTaskLens[PTHREAD_NUM][MAX_CPU_TASK_NUM]; // lengths of cpu tasks, in ms
float gpuTaskLens[PTHREAD_NUM][MAX_GPU_TASK_NUM]; // lengths of gpu tasks, in ms
float utilRates[PTHREAD_NUM];                     // utility rates of each pthread
float ddls[PTHREAD_NUM];                          // deadline of batch on each pthread, in ms

__device__ float deviceData[PTHREAD_NUM][2048]; // gpu task data preparation
cudaStream_t cudaStreams[PTHREAD_NUM];          // each pthreads has its own cuda stream

int prios[PTHREAD_NUM];                    // priority of each pthread: 0 ~ 99
pthread_t mainThreads[PTHREAD_NUM];        // main threads
pthread_t syncThreads[PTHREAD_NUM];        // threads for cuda synchronization
int timeExceeded = 0;                      // whether there's a batch missing its deadline
pthread_mutex_t syncStartMut[PTHREAD_NUM]; // used to activate cuda synchronization
pthread_mutex_t syncEndMut[PTHREAD_NUM];   // used to wake up main threads

void pthreadDataGen(float totalUtilRate, int level, int scale) {
  srand((unsigned)time(NULL));

  switch (scale) {
  case 1:
    cpuTaskNum = 3;
    gpuTaskNum = 2;
    break;
  case 2:
    cpuTaskNum = 5;
    gpuTaskNum = 4;
    break;
  case 3:
    cpuTaskNum = 10;
    gpuTaskNum = 9;
    break;
  }

  // utilization rate for each pthread
  float sumUtilRate = 0;
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    // for practical reasons, one util rate is at most twice of another
    sumUtilRate += utilRates[_tid] = rand() % 101 + 100;
  }
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    utilRates[_tid] /= sumUtilRate / totalUtilRate;
  }

  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    float C = 0; // C: sum cpu task lengths
    float S = 0; // S: sum gpu task lengths
    // cpuTaskLens
    for (int i = 0; i < cpuTaskNum; ++i) {
      C += cpuTaskLens[_tid][i] = rand() % 10 + 1;
    }
    // gpuTaskLens, ddls
    if (level == 0) {
      for (int i = 0; i < gpuTaskNum; ++i) {
        S += gpuTaskLens[_tid][i] = rand() % 10 + 1;
      }
      ddls[_tid] = (C + S) / utilRates[_tid];
    } else {
      ddls[_tid] = C / utilRates[_tid];
      // for practical reasons, length of one gpu task is at most twice of the other's
      switch (level) {
      case 1:
        S = (rand() % 91 + 10) / 1000.0 * (ddls[_tid] - C);
        break;
      case 2:
        S = (rand() % 101 + 20) / 200.0 * (ddls[_tid] - C);
        break;
      case 3:
        S = (rand() % 101 + 150) / 250.0 * (ddls[_tid] - C);
        break;
      default:
        break;
      }
      float tmp = 0;
      for (int i = 0; i < gpuTaskNum; ++i) {
        tmp += gpuTaskLens[_tid][i] = rand() % 101 + 100;
      }
      for (int i = 0; i < gpuTaskNum; ++i) {
        gpuTaskLens[_tid][i] /= tmp / S;
      }
    }
  }
}

void pthreadDataWrite() {
  ofstream ofstrm;
  ofstrm.open("pthreadData.dat");
  ofstrm << cpuTaskNum << " " << gpuTaskNum << " ";
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    ofstrm << utilRates[_tid] << " " << ddls[_tid] << " ";
    for (int i = 0; i < cpuTaskNum; ++i) {
      ofstrm << cpuTaskLens[_tid][i] << " ";
    }
    for (int i = 0; i < gpuTaskNum; ++i) {
      ofstrm << gpuTaskLens[_tid][i] << " ";
    }
  }
  ofstrm << endl;
  ofstrm.close();
}

void pthreadDataRead() {
  ifstream ifstrm;
  ifstrm.open("pthreadData.dat");
  ifstrm >> cpuTaskNum >> gpuTaskNum;
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    ifstrm >> utilRates[_tid] >> ddls[_tid];
    for (int i = 0; i < cpuTaskNum; ++i) {
      ifstrm >> cpuTaskLens[_tid][i];
    }
    for (int i = 0; i < gpuTaskNum; ++i) {
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
    for (int i = 0; i < cpuTaskNum; ++i) {
      cout << cpuTaskLens[_tid][i] << " ";
    }
    cout << '\n';
    cout << "gpu: ";
    for (int i = 0; i < gpuTaskNum; ++i) {
      cout << gpuTaskLens[_tid][i] << " ";
    }
    cout << endl;
  }
}

void prioGen(int nth) {
  vector<vector<int>> prioAllPermu(120);

  int i = 0;
  int row = 0;
  vector<int> C(PTHREAD_NUM, 0);
  vector<int> A = {95, 96, 97, 98, 99};
  prioAllPermu[row] = A;
  ++row;
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
