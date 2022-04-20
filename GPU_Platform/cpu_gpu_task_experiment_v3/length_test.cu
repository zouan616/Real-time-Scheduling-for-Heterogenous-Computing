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

// USAGE: ./length_test [taskType] [length] [unitTask]
// taskType: 0 -> cpu, 1 -> gpu
// length: length of a single task, in ms, float
// unitTask: number of loops to make up a 1ms task

// ctrl + c to terminate
// nvprof to inspect gpu test

long cpuUnitTask;
__device__ long gpuUnitTask;
__device__ float testDeviceData[2048];
int taskType;
float length;

void cpuTaskFunc(float cpuTaskLen) {
  float c = 0;
  long i = cpuTaskLen * cpuUnitTask;
  for (long j = 0; j < i; ++j) {
    c += 98765.4321 / 654.321;
    c -= 98765.4321 / 654.321;
    c += 98765.4321 / 654.321;
    c -= 98765.4321 / 654.321;
  }
}

__global__ void gpuTaskFunc(float gpuTaskLen) {
  long i = threadIdx.x + blockIdx.x * blockDim.x;
  long j = gpuTaskLen * gpuUnitTask;
  float a = 9876.54321, b = 543.21;
  for (long k = 0; k < j; ++k) {
    testDeviceData[i] += a / b;
    testDeviceData[i] -= a / b;
  }
}

void *threadFunc(void *pd) {
  // pin to either core 4 or 5
  cpu_set_t cpuSet;
  CPU_ZERO(&cpuSet);
  CPU_SET(4, &cpuSet);
  CPU_SET(5, &cpuSet);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuSet);

  // set scheduling config
  struct sched_param schedParam;
  schedParam.sched_priority = 99;
  sched_setscheduler(getpid(), SCHED_FIFO, &schedParam);

  struct timeval startTime;
  struct timeval endTime;
  long duration;

  // gpu preparation
  float testHostData[2048] = {0};
  cudaMemcpyToSymbol(testDeviceData, testHostData, 2048 * sizeof(float));
  cudaStream_t strm;
  cudaStreamCreate(&strm);
  
  // infinitely launch tasks
  if (taskType == 0) {
    while (1) {
      gettimeofday(&startTime, NULL);
      cpuTaskFunc(length);
      gettimeofday(&endTime, NULL);
      duration = endTime.tv_sec * 1000000 + endTime.tv_usec - (startTime.tv_sec * 1000000 + startTime.tv_usec);
      cout << duration << endl;
    }
  } else {
    while (1) {
      gpuTaskFunc<<<2, 1024, 0, strm>>>(length);
      cudaStreamSynchronize(strm);
    }
  }
  return NULL;
}

int main(int argc, char **argv) {
  // process input parameters
  taskType = atoi(argv[1]);
  if (taskType == 0) {
    cpuUnitTask = (long)atoi(argv[3]);
  } else {
    long tmp = (long)atoi(argv[3]);
    cudaMemcpyToSymbol(gpuUnitTask, &tmp, sizeof(long));
  }
  length = atof(argv[2]);
  // test on a pthread
  pthread_t pthread;
  pthread_create(&pthread, NULL, threadFunc, NULL);
  pthread_join(pthread, NULL);
  return 0;
}
