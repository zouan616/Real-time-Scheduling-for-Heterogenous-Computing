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

#define CPU_UNIT_TASK (50000) // parameter to generate a unit cpu task of 1 ms
#define GPU_UNIT_TASK (28500) // parameter to generate a unit gpu task of 1 ms

#define CPU_TASK_NUM (3) // number of cpu tasks in a job
#define GPU_TASK_NUM (2) // number of gpu tasks in a job
#define PTHREAD_NUM (5)  // number of pthreads

float cpuTaskLens[PTHREAD_NUM][CPU_TASK_NUM]; // lengths of cpu tasks: (int) 1 ~ 10 ms
float gpuTaskLens[PTHREAD_NUM][GPU_TASK_NUM]; // lengths of gpu tasks: ms
float ddls[PTHREAD_NUM];                      // deadline: ms
int prios[PTHREAD_NUM];                       // priority of each thread: 0 ~ 99
float *hostData[PTHREAD_NUM];                 // gpu task data preparation
float *deviceData[PTHREAD_NUM];               // gpu task data preparation
pthread_t pthreads[PTHREAD_NUM];
cudaStream_t streams[PTHREAD_NUM];
pthread_mutex_t pthrdMuts[PTHREAD_NUM];
pthread_mutex_t exitMut;
