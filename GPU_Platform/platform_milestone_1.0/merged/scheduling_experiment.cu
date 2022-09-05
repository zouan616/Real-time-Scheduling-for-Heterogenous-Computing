// sudo ./scheduling_experiment [priorityOrder]

#include "scheduling_experiment.h"

int fdMinGpuFreq;
int fdMaxGpuFreq;
set<float> gpuRatioTree;
unsigned long gpuAvaiFreqs[] = {
    114750000, 216750000, 318750000,  420750000,  522750000,  624750000, 726750000,
    854250000, 930750000, 1032750000, 1122000000, 1236750000, 1300500000}; // default = 114750000

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

  struct timeval cpuSegStartTime;
  struct timeval cpuSegEndTime;
  long cpuSegDuration; // us
  long diffDuration;   // us
  float gpuRatio;
  float targetGpuRatio;

  // MAIN LOOP
  for (int i = 0; i < 100; ++i) {
    gettimeofday(&startTime, NULL);
    for (int j = 0; j < cpuTaskNum[_tid] - 1; ++j) {
      gettimeofday(&cpuSegStartTime, NULL);
      cpuTaskFunc(cpuTaskLens[_tid][j]);
      gettimeofday(&cpuSegEndTime, NULL);
      cpuSegDuration = cpuSegEndTime.tv_sec * 1000000 + cpuSegEndTime.tv_usec -
                       (cpuSegStartTime.tv_sec * 1000000 + cpuSegStartTime.tv_usec);
      if ((diffDuration = cpuSegDuration - cpuTaskLens[_tid][j] * 1000) < 0) {
        gpuRatio = (diffDuration + gpuTaskLens[_tid][j]) / gpuTaskLens[_tid][j];
        gpuRatioTree.insert(gpuRatio);
        targetGpuRatio = *gpuRatioTree.begin();
        // TODO find appropriate GPU frequency
        // syscall(SYS_write, fdMaxGpuFreq, "318750000", 9);
        // syscall(SYS_write, fdMinGpuFreq, "318750000", 9);
      }
      gpuTaskFunc<<<2, 1024, 0, cudaStreams[_tid]>>>(_tid, gpuTaskLens[_tid][j]);
      debugCall(pthread_mutex_unlock(&syncStartMut[_tid]));
      debugCall(pthread_mutex_lock(&syncEndMut[_tid]));
      gpuRatioTree.erase(gpuRatio);
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
  prioGen(atoi(argv[1]));

  fdMinGpuFreq = open("/sys/devices/17000000.gp10b/devfreq/17000000.gp10b/min_freq", O_WRONLY | O_TRUNC);
  fdMaxGpuFreq = open("/sys/devices/17000000.gp10b/devfreq/17000000.gp10b/max_freq", O_WRONLY | O_TRUNC);

  // gpu warm up TODO may not need anymore, since static gpu frequencies
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

  close(fdMinGpuFreq);
  close(fdMaxGpuFreq);
  exit(timeExceeded);
  return 0;
}
