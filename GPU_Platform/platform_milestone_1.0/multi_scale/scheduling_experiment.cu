// sudo ./scheduling_experiment [priorityOrder]

#include "scheduling_experiment.h"

void cpuTaskFunc(float cpuTaskLen) {
  float c = 0;
  long i = cpuTaskLen * CPU_UNIT_TASK;
  for (long j = 0; j < i; ++j) {
    c += 98765.4321 / 654.321;
    c -= 98765.4321 / 654.321;
    c += 98765.4321 / 654.321;
    c -= 98765.4321 / 654.321;
  }
}

__global__ void gpuTaskFunc(int _tid, float gpuTaskLen) {
  long i = threadIdx.x + blockIdx.x * blockDim.x;
  long j = gpuTaskLen * GPU_UNIT_TASK;
  float a = 9876.54321, b = 543.21;
  for (long k = 0; k < j; ++k) {
    deviceData[_tid][i] += a / b;
    deviceData[_tid][i] -= a / b;
  }
}

void *threadFunc(void *_tidPtr) {
  int _tid = *(int *)_tidPtr;
  long ddlusec = ddls[_tid] * 1000;

  cpu_set_t cpuSet;
  CPU_ZERO(&cpuSet);
  CPU_SET(6, &cpuSet);
  CPU_SET(7, &cpuSet);
  debugCall(pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuSet));

  struct sched_param schedParam;
  schedParam.sched_priority = prios[_tid];
  debugCall(pthread_setschedparam(pthread_self(), SCHED_FIFO, &schedParam));

  struct timeval startTime;
  struct timeval endTime;
  long duration; // us

  // MAIN LOOP
  for (int i = 0; i < 100; ++i) {
    gettimeofday(&startTime, NULL);
    for (int j = 0; j < cpuTaskNum[_tid] - 1; ++j) {
      cpuTaskFunc(cpuTaskLens[_tid][j]);
      gpuTaskFunc<<<2, 1024, 0, cudaStreams[_tid]>>>(_tid, gpuTaskLens[_tid][j]);
      debugCall(pthread_mutex_unlock(&syncStartMut[_tid]));
      debugCall(pthread_mutex_lock(&syncEndMut[_tid]));
    }
    cpuTaskFunc(cpuTaskLens[_tid][cpuTaskNum[_tid] - 1]);
    gettimeofday(&endTime, NULL);

    // some other pthreads time exceeded
    if (timeExceeded) {
      return NULL;
    }
    // current pthread time exceeded
    duration = endTime.tv_sec * 1000000 + endTime.tv_usec - (startTime.tv_sec * 1000000 + startTime.tv_usec);
    if (duration > ddlusec) {
      timeExceeded = 1;
      return NULL;
    }
    // current pthread not time exceeded, sleep until deadline
    usleep(ddlusec - duration);
  }
  // current pthread successfully scheduled
  return NULL;
}

void *syncFunc(void *_tidPtr) {
  int _tid = *(int *)_tidPtr;

  cpu_set_t cpuSet;
  CPU_ZERO(&cpuSet);
  CPU_SET(1 + _tid, &cpuSet);
  debugCall(pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuSet));

  struct sched_param schedParam;
  schedParam.sched_priority = 99;
  debugCall(pthread_setschedparam(pthread_self(), SCHED_FIFO, &schedParam));

  while (1) {
    debugCall(pthread_mutex_lock(&syncStartMut[_tid]));
    cudaDebugCall(cudaStreamSynchronize(cudaStreams[_tid]));
    debugCall(pthread_mutex_unlock(&syncEndMut[_tid]));
  }
  return NULL;
}

int main(int argc, char **argv) {
  cpu_set_t cpuSet;
  CPU_ZERO(&cpuSet);
  CPU_SET(0, &cpuSet);
  debugCall(pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuSet));

  // init setup
  cudaDebugCall(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    cudaDebugCall(cudaStreamCreate(&cudaStreams[_tid]));
    syncStartMut[_tid] = PTHREAD_MUTEX_INITIALIZER;
    debugCall(pthread_mutex_lock(&syncStartMut[_tid]));
    syncEndMut[_tid] = PTHREAD_MUTEX_INITIALIZER;
    debugCall(pthread_mutex_lock(&syncEndMut[_tid]));
  }

  pthreadDataRead();
  // pthreadDataPrint();
  prioGen(atoi(argv[1]));

  // gpu warm up
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    gpuTaskFunc<<<2, 1024, 0, cudaStreams[_tid]>>>(_tid, 250);
  }
  usleep(250000);
  cudaDebugCall(cudaDeviceSynchronize());

  // START SCHEDULING
  int _tids[PTHREAD_NUM];
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    _tids[_tid] = _tid;
    debugCall(pthread_create(&mainThreads[_tid], NULL, threadFunc, (void *)&_tids[_tid]));
    debugCall(pthread_create(&syncThreads[_tid], NULL, syncFunc, (void *)&_tids[_tid]));
  }
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    debugCall(pthread_join(mainThreads[_tid], NULL));
  }
  cudaDebugCall(cudaDeviceReset());
  exit(timeExceeded);
  return 0;
}
