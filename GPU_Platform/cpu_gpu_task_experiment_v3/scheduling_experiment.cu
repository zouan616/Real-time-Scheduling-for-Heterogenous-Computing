#include "scheduling_experiment.h"

void cpuTaskFunc(float cpuTaskLen) {
  float a = 9876.54321, b = 654.321, c = 0;
  long i = cpuTaskLen * CPU_UNIT_TASK;
  for (long j = 0; j < i; ++j) {
    c += a / b;
    c -= a / b;
    c += a / b;
    c -= a / b;
    c += a / b;
    c -= a / b;
    c += a / b;
    c -= a / b;
    c += a / b;
    c -= a / b;
  }
}

__global__ void gpuTaskFunc(float *dvcData, float gpuTaskLen) {
  long i = threadIdx.x + blockIdx.x * blockDim.x;
  float a = 9876.54321, b = 654.321;
  long j = gpuTaskLen * GPU_UNIT_TASK;
  for (long k = 0; k < j; ++k) {
    dvcData[i] += a / b;
    dvcData[i] -= a / b;
    dvcData[i] += a / b;
    dvcData[i] -= a / b;
  }
}

void *threadFunc(void *_pidPtr) {
  int _pid = *(int *)_pidPtr;
  long ddlusec = ddls[_pid] * 1000; // convert ddl to microsecond, consistent with duration

  // set scheduling config
  struct sched_param schedParam;
  schedParam.sched_priority = prios[_pid];
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &schedParam);

  // pin to either core 4 or 5
  cpu_set_t cpuSet;
  CPU_ZERO(&cpuSet);
  CPU_SET(4, &cpuSet);
  CPU_SET(5, &cpuSet);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuSet);

  struct timeval startTime;
  struct timeval endTime;
  long duration; // microsecond

  // PERFORM TASKS, 100 TIMES
  for (int i = 0; i < 100; ++i) {
    gettimeofday(&startTime, NULL);

    // lock when performing tasks
    pthread_mutex_lock(&pthrdMuts[_pid]);
    cpuTaskFunc(cpuTaskLens[_pid][0]);
    gpuTaskFunc<<<2, 1024, 0, streams[_pid]>>>(deviceData[i], gpuTaskLens[_pid][0]);
    debugCall(cudaStreamSynchronize(streams[_pid]));
    cpuTaskFunc(cpuTaskLens[_pid][1]);
    gpuTaskFunc<<<2, 1024, 0, streams[_pid]>>>(deviceData[i], gpuTaskLens[_pid][1]);
    debugCall(cudaStreamSynchronize(streams[_pid]));
    cpuTaskFunc(cpuTaskLens[_pid][2]);
    pthread_mutex_unlock(&pthrdMuts[_pid]);

    gettimeofday(&endTime, NULL);

    duration = endTime.tv_sec * 1000000 + endTime.tv_usec - (startTime.tv_sec * 1000000 + startTime.tv_usec);
    if (duration > ddlusec) {
      // only the first thread can enter this region
      pthread_mutex_lock(&exitMut);
      for (int j = 0; j < PTHREAD_NUM; ++j) {
        // wait all threads finishes tasks
        pthread_mutex_lock(&pthrdMuts[j]);
      }
      for (int i = 0; i < PTHREAD_NUM; ++i) {
        free(hostData[i]);
        cudaFree(deviceData[i]);
      }
      exit(10);
    }
    usleep(ddlusec - duration);
  }
  return NULL;
}

int main(int argc, char **argv) {
  // init setup
  for (int i = 0; i < PTHREAD_NUM; ++i) {
    pthread_mutex_init(&pthrdMuts[i], NULL);
    debugCall(cudaStreamCreate(&streams[i]));
  }
  pthread_mutex_init(&exitMut, NULL);
  srand((unsigned)time(NULL));
  // instead of default busy waiting (polling)
  // now cudaStreamSynchronize will release cpu
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

/* TODO, read in these data
cpuTaskLens[PTHREAD_NUM][CPU_TASK_NUM]; // lengths of cpu tasks: (int) 1 ~ 10 ms
gpuTaskLens[PTHREAD_NUM][GPU_TASK_NUM]; // lengths of gpu tasks: ms
ddls[PTHREAD_NUM];                      // deadline: ms
prios[PTHREAD_NUM];                       // priority of each thread: 0 ~ 99
*/

  // prepare data
  for (int i = 0; i < PTHREAD_NUM; ++i) {
    hostData[i] = (float *)malloc(2048 * sizeof(float));
    memset(hostData[i], 0, 2048 * sizeof(float));
    debugCall(cudaMalloc((void **)&deviceData[i], 2048 * sizeof(float)));
    debugCall(cudaMemcpy(deviceData[i], hostData[i], 2048 * sizeof(float), cudaMemcpyHostToDevice));
  }

  // warm up gpu, should be removed in later versions
  float *hd = (float *)malloc(2048 * sizeof(float));
  memset(hd, 0, 2048 * sizeof(float));
  float *dd;
  debugCall(cudaMalloc((void **)&dd, 2048 * sizeof(float)));
  debugCall(cudaMemcpy(dd, hd, 2048 * sizeof(float), cudaMemcpyHostToDevice));
  for (int i = 0; i < 250; ++i) {
    gpuTaskFunc<<<2, 1024, 0, 0>>>(dd, 1);
  }
  free(hd);
  debugCall(cudaFree(dd));
  debugCall(cudaDeviceSynchronize());

  int _pids[PTHREAD_NUM];
  // create pthreads
  for (int i = 0; i < PTHREAD_NUM; ++i) {
    _pids[i] = i;
    pthread_create(&pthreads[i], NULL, threadFunc, (void *)&_pids[i]);
  }
  for (int i = 0; i < PTHREAD_NUM; ++i) {
    pthread_join(pthreads[i], NULL);
  }
  // clean memory
  for (int i = 0; i < PTHREAD_NUM; ++i) {
    free(hostData[i]);
    debugCall(cudaFree(deviceData[i]));
  }
  return 0;
}
