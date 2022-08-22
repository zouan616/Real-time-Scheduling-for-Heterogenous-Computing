/*
USAGE:
  sudo ./length_test [taskType] [taskLen] [unitTask]
  ctrl + c to terminate
  nvprof to inspect gpu tasks test results
EFFECT:
  test the time cost of CPU/GPU tasks
ARGS:
  [taskType]: 0 -> cpu, 1 -> gpu
  [taskLen]: (float) task length of a single task, in ms
  [unitTask]: (int) number of iterations to make up a 1 ms task
*/

#include <stdio.h>
#include <sys/time.h>

#define cudaDebugCall(F)                                                                                               \
  if ((F) != cudaSuccess) {                                                                                            \
    printf("Error at line %d: %s\n", __LINE__, cudaGetErrorString(cudaGetLastError()));                                \
    exit(1);                                                                                                           \
  };

int taskType;
float taskLen;
int cpuUnitTask;
__device__ int gpuUnitTask;
__device__ float deviceData[2048];

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
    deviceData[i] += a / b;
    deviceData[i] -= a / b;
  }
}

int main(int argc, char **argv) {
  taskType = atoi(argv[1]);
  taskLen = atof(argv[2]);

  if (taskType == 0) {
    cpuUnitTask = atoi(argv[3]);
    while (1) {
      struct timeval startTime;
      struct timeval endTime;
      long duration;
      gettimeofday(&startTime, NULL);
      cpuTaskFunc(taskLen);
      gettimeofday(&endTime, NULL);
      duration = endTime.tv_sec * 1000000 + endTime.tv_usec - (startTime.tv_sec * 1000000 + startTime.tv_usec);
      printf("%ld\n", duration);
    }
  } else {
    int tmp = atoi(argv[3]);
    cudaDebugCall(cudaMemcpyToSymbol(gpuUnitTask, &tmp, sizeof(int)));
    gpuTaskFunc<<<2, 1024, 0, 0>>>(250); // gpu warm up
    cudaDebugCall(cudaDeviceSynchronize());
    while (1) {
      gpuTaskFunc<<<2, 1024, 0, 0>>>(taskLen);
      cudaDebugCall(cudaDeviceSynchronize());
    }
  }

  return 0;
}
