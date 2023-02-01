#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <sched.h>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
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

#define CPU_UNIT_TASK (10000) // parameter to generate a unit cpu task of 1 ms
#define GPU_UNIT_TASK (10000) // parameter to generate a unit gpu task of 1 ms
#define MAX_CPU_TASK_NUM (51) // max number of cpu tasks in a batch
#define MAX_GPU_TASK_NUM (51) // max number of gpu tasks in a batch
#define PTHREAD_NUM (5)       // number of pthreads
int cpuTaskNum[PTHREAD_NUM];                     // number of cpu tasks in a batch
int gpuTaskNum[PTHREAD_NUM];                      // number of cpu tasks in a batch
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


/* ----------------------------------------
  Three types of task generation
  ---------------------------------------- */
void pthreadDataGen_Benchmark(float totalUtilRate) {
  srand((unsigned)time(NULL));

  cpuTaskNum[0] = 9;
  cpuTaskNum[1] = 12;
  cpuTaskNum[2] = 20;
  cpuTaskNum[3] = 23;
  cpuTaskNum[4] = 51;
  gpuTaskNum[0] = 8;
  gpuTaskNum[1] = 11;
  gpuTaskNum[2] = 19;
  gpuTaskNum[3] = 22;
  gpuTaskNum[4] = 50;

  // UTIL RATE
  float sumUtilRate = 0;
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    // for practical reasons, one util rate is at most twice of another
    sumUtilRate += utilRates[_tid] = rand() % 101 + 100;
  }
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    utilRates[_tid] /= sumUtilRate / totalUtilRate;
  }

  // CPU TASK LENGTHS
  float C[PTHREAD_NUM] = {0}; // sum cpu task lengths
  for (int i = 0; i < cpuTaskNum[0]; ++i) {
    C[0] += cpuTaskLens[0][i] = rand() % 50 + 1;
  }
  for (int i = 0; i < cpuTaskNum[1]; ++i) {
    C[1] += cpuTaskLens[1][i] = rand() % 50 + 1;
  }
  for (int i = 0; i < cpuTaskNum[2]; ++i) {
    C[2] += cpuTaskLens[2][i] = rand() % 50 + 1;
  }
  for (int i = 0; i < cpuTaskNum[3]; ++i) {
    C[3] += cpuTaskLens[3][i] = rand() % 50 + 1;
  }
  for (int i = 0; i < cpuTaskNum[4]; ++i) {
    C[4] += cpuTaskLens[4][i] = rand() % 50 + 1;
  }

  // GPU TASK LENGTHS
  float S[PTHREAD_NUM] = {0}; // sum gpu task lengths
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

void pthreadDataGen_Scaled(float totalUtilRate, int scale) {
  srand((unsigned)time(NULL));

  switch (scale) {
  case 1:
    cpuTaskNum[0] = 3;
    cpuTaskNum[1] = 3;
    cpuTaskNum[2] = 3;
    cpuTaskNum[3] = 3;
    cpuTaskNum[4] = 3;
    gpuTaskNum[0] = 2;
    gpuTaskNum[1] = 2;
    gpuTaskNum[2] = 2;
    gpuTaskNum[3] = 2;
    gpuTaskNum[4] = 2;
    break;
  case 2:
    cpuTaskNum[0] = 5;
    cpuTaskNum[1] = 5;
    cpuTaskNum[2] = 5;
    cpuTaskNum[3] = 5;
    cpuTaskNum[4] = 5;
    gpuTaskNum[0] = 4;
    gpuTaskNum[1] = 4;
    gpuTaskNum[2] = 4;
    gpuTaskNum[3] = 4;
    gpuTaskNum[4] = 4;
    break;
  case 3:
    cpuTaskNum[0] = 10;
    cpuTaskNum[1] = 10;
    cpuTaskNum[2] = 10;
    cpuTaskNum[3] = 10;
    cpuTaskNum[4] = 10;
    gpuTaskNum[0] = 9;
    gpuTaskNum[1] = 9;
    gpuTaskNum[2] = 9;
    gpuTaskNum[3] = 9;
    gpuTaskNum[4] = 9;
    break;
  }

  // UTIL RATE
  float sumUtilRate = 0;
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    // for practical reasons, one util rate is at most twice of another
    sumUtilRate += utilRates[_tid] = rand() % 101 + 100;
  }
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    utilRates[_tid] /= sumUtilRate / totalUtilRate;
  }

  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    float C = 0; // sum cpu task lengths
    float S = 0; // sum gpu task lengths
    // CPU TASK LENGTHS
    for (int i = 0; i < cpuTaskNum[_tid]; ++i) {
      C += cpuTaskLens[_tid][i] = rand() % 50 + 1;
    }
    // GPU TASK LENGTHS
    for (int i = 0; i < gpuTaskNum[_tid]; ++i) {
      S += gpuTaskLens[_tid][i] = rand() % 50 + 1;
    }
    // DDL
    ddls[_tid] = (C + S) / utilRates[_tid];
  }
}

void pthreadDataGen_Scaled_GLenLeveled(float totalUtilRate, int scale, int level) {
  srand((unsigned)time(NULL));

  switch (scale) {
  case 1:
    cpuTaskNum[0] = 3;
    cpuTaskNum[1] = 3;
    cpuTaskNum[2] = 3;
    cpuTaskNum[3] = 3;
    cpuTaskNum[4] = 3;
    gpuTaskNum[0] = 2;
    gpuTaskNum[1] = 2;
    gpuTaskNum[2] = 2;
    gpuTaskNum[3] = 2;
    gpuTaskNum[4] = 2;
    break;
  case 2:
    cpuTaskNum[0] = 5;
    cpuTaskNum[1] = 5;
    cpuTaskNum[2] = 5;
    cpuTaskNum[3] = 5;
    cpuTaskNum[4] = 5;
    gpuTaskNum[0] = 4;
    gpuTaskNum[1] = 4;
    gpuTaskNum[2] = 4;
    gpuTaskNum[3] = 4;
    gpuTaskNum[4] = 4;
    break;
  case 3:
    cpuTaskNum[0] = 10;
    cpuTaskNum[1] = 10;
    cpuTaskNum[2] = 10;
    cpuTaskNum[3] = 10;
    cpuTaskNum[4] = 10;
    gpuTaskNum[0] = 9;
    gpuTaskNum[1] = 9;
    gpuTaskNum[2] = 9;
    gpuTaskNum[3] = 9;
    gpuTaskNum[4] = 9;
    break;
  }

  // UTIL RATE
  float sumUtilRate = 0;
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    // for practical reasons, one util rate is at most twice of another
    sumUtilRate += utilRates[_tid] = rand() % 101 + 100;
  }
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    utilRates[_tid] /= sumUtilRate / totalUtilRate;
  }

  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    float C = 0; // sum cpu task lengths
    float S = 0; // sum gpu task lengths
    // CPU TASK LENGTHS
    for (int i = 0; i < cpuTaskNum[_tid]; ++i) {
      C += cpuTaskLens[_tid][i] = rand() % 10 + 1;
    }
    // DDL
    ddls[_tid] = C / utilRates[_tid];
    // CPU TASK LENGTHS
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
    for (int i = 0; i < gpuTaskNum[_tid]; ++i) {
      tmp += gpuTaskLens[_tid][i] = rand() % 101 + 100;
    }
    for (int i = 0; i < gpuTaskNum[_tid]; ++i) {
      gpuTaskLens[_tid][i] /= tmp / S;
    }
  }
}

void pthreadDataWrite() {
  ofstream ofstrm;
  ofstrm.open("pthreadData.dat");
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    ofstrm << cpuTaskNum[_tid] << " " << gpuTaskNum[_tid] << " " << utilRates[_tid] << " " << ddls[_tid] << " ";
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
    ifstrm >> cpuTaskNum[_tid] >> gpuTaskNum[_tid] >> utilRates[_tid] >> ddls[_tid];
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
    cout << cpuTaskNum[_tid] << " cpu tasks: ";
    for (int i = 0; i < cpuTaskNum[_tid]; ++i) {
      cout << cpuTaskLens[_tid][i] << " ";
    }
    cout << '\n';
    cout << gpuTaskNum[_tid] << " gpu tasks: ";
    for (int i = 0; i < gpuTaskNum[_tid]; ++i) {
      cout << gpuTaskLens[_tid][i] << " ";
    }
    cout << '\n' << endl;
  }
}

/* ----------------------------------------
  Priority assignment
   ---------------------------------------- */
void prioGen(int nth) {
  // Fixed priority
  prios[0] = 95;
  prios[1] = 96;
  prios[2] = 97;
  prios[3] = 98;
  prios[4] = 99;
  
  // priority traversal
  /*
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
  */
}
