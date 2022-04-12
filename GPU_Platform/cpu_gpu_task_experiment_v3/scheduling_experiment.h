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

#define debugCall(F)                                                                                                   \
  if ((F) != cudaSuccess) {                                                                                            \
    printf("Error at line %d: %s\n", __LINE__, cudaGetErrorString(cudaGetLastError()));                                \
    exit(-1);                                                                                                          \
  };

// worst case is 1.5 times slower than average case
#define CPU_UNIT_TASK (60000) / (1.2) // parameter to generate a unit cpu task of 1 ms
#define GPU_UNIT_TASK (46200) / (1.5) // parameter to generate a unit gpu task of 1 ms

#define MAX_CPU_TASK_NUM (10) // max number of cpu tasks in a job
#define MAX_GPU_TASK_NUM (10) // max number of gpu tasks in a job
#define PTHREAD_NUM (5)       // number of pthreads

float utilRates[PTHREAD_NUM];                     // utility rates of each pthread
float cpuTaskLens[PTHREAD_NUM][MAX_CPU_TASK_NUM]; // lengths of cpu tasks: (int) 1 ~ 10 ms
float gpuTaskLens[PTHREAD_NUM][MAX_GPU_TASK_NUM]; // lengths of gpu tasks: ms
float ddls[PTHREAD_NUM];                          // deadline: ms
int prios[PTHREAD_NUM];                           // priority of each pthread: 0 ~ 99
float hostData[PTHREAD_NUM * 2048] = {0};         // gpu task data preparation
__device__ float deviceData[PTHREAD_NUM][2048];   // gpu task data preparation
int cpuTaskNum;                                   // number of cpu tasks in a job
int gpuTaskNum;                                   // number of gpu tasks in a job
pthread_t pthreads[PTHREAD_NUM];                  // pthreads
cudaStream_t streams[PTHREAD_NUM];                // each pthreads has its own cuda stream
int timeExceeded = 0;                             // whether there's a job missing its deadline