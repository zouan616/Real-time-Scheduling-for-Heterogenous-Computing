#include "scheduling_experiment.h"

void pthreadDataGenerator(pthread_data_t *pd, float util, int lv) {
  srand((unsigned)time(NULL));
  float C = 0, T = 0;

  // cpuTaskLens
  for (int i = 0; i < CPU_TASK_NUM; ++i) {
    C += pd->cpuTaskLens[i] = rand() % 10 + 1;
  }

  // ddl
  T = (pd->ddl = C / util);

  // gpuTaskLens
  // for practical reasons, length of one gpu task is at most twice of the other's
  switch (lv) {
  case 1: // S = rand(0.01, 0.1) * (T - C)
    pd->gpuTaskLens[0] = (rand() % 101 + 100) / 1100.0 * (T - C);
    pd->gpuTaskLens[1] = (rand() % 51 + 50) / 150 * pd->gpuTaskLens[0];
    pd->gpuTaskLens[0] -= pd->gpuTaskLens[1];
    break;
  case 2: // S = rand(0.1, 0.6) * (T - C)
    pd->gpuTaskLens[0] = (rand() % 101 + 20) / 200.0 * (T - C);
    pd->gpuTaskLens[1] = (rand() % 51 + 50) / 150 * pd->gpuTaskLens[0];
    pd->gpuTaskLens[0] -= pd->gpuTaskLens[1];
    break;
  case 3: // S = rand(0.6, 1) * (T - C)
    pd->gpuTaskLens[0] = (rand() % 101 + 150) / 250.0 * (T - C);
    pd->gpuTaskLens[1] = (rand() % 51 + 50) / 150 * pd->gpuTaskLens[0];
    pd->gpuTaskLens[0] -= pd->gpuTaskLens[1];
    break;
  default:
    break;
  }

  printf("util = %f, ddl = %f\n", util, pd->ddl);
  printf("c0 = %f, c1 = %f, c2 = %f\n", pd->cpuTaskLens[0], pd->cpuTaskLens[1], pd->cpuTaskLens[2]);
  printf("g0 = %f, g1 = %f\n\n", pd->gpuTaskLens[0], pd->gpuTaskLens[1]);
}

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

void *threadFunc(void *pd) {
  float *cpuTaskLens = ((pthread_data_t *)pd)->cpuTaskLens;
  float *gpuTaskLens = ((pthread_data_t *)pd)->gpuTaskLens;
  long ddlusec = ((pthread_data_t *)pd)->ddl * 1000; // convert ddl to microsecond, consistent with duration
  unsigned int sleepTime0 = gpuTaskLens[0] * 1000;
  unsigned int sleepTime1 = gpuTaskLens[1] * 1000;

  // set scheduling config
  struct sched_param schedParam;
  schedParam.sched_priority = ((pthread_data_t *)pd)->prio;
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

  // prepare data
  float *hostData = (float *)malloc(2048 * sizeof(float));
  for (int i = 0; i < 2048; ++i) {
    hostData[i] = 0;
  }
  float *deviceData;
  cudaMalloc((void **)&deviceData, 2048 * sizeof(float));
  cudaMemcpy(deviceData, hostData, 2048 * sizeof(float), cudaMemcpyHostToDevice);

  cudaStream_t strm;
  cudaStreamCreate(&strm);

  // warm up gpu stream
  for (int i = 0; i < 100; ++i) {
    gpuTaskFunc<<<2, 1024, 0, strm>>>(deviceData, 0.5);
  }
  cudaStreamSynchronize(strm);

  // PERFORM TASKS, 100 TIMES
  for (int i = 0; i < 100; ++i) {
    gettimeofday(&startTime, NULL);

    cpuTaskFunc(cpuTaskLens[0]);
    gpuTaskFunc<<<2, 1024, 0, strm>>>(deviceData, gpuTaskLens[0]);
    usleep(sleepTime0);
    cudaStreamSynchronize(strm);
    cpuTaskFunc(cpuTaskLens[1]);
    gpuTaskFunc<<<2, 1024, 0, strm>>>(deviceData, gpuTaskLens[1]);
    usleep(sleepTime1);
    cudaStreamSynchronize(strm);
    cpuTaskFunc(cpuTaskLens[2]);

    gettimeofday(&endTime, NULL);

    duration = endTime.tv_sec * 1000000 + endTime.tv_usec - (startTime.tv_sec * 1000000 + startTime.tv_usec);
    if (duration > ddlusec) {
      printf("[%ld > %ld] Thread time exceeded, abort!\n", duration, ddlusec);
      exit(-1);
    }
    usleep(ddlusec - duration);
  }
  return NULL;
}

int main(int argc, char **argv) {
  srand((unsigned)time(NULL));
  // S = level * (T - C)
  // level 1: rand(0.01, 0.1)
  // level 2: rand(0.1, 0.6)
  // level 3: rand(0.6, 1)
  int level = atoi(argv[2]);

  float totalUtilRate = atoi(argv[1]) / 100.0, utilRates[5];
  float sumUtilRate = 0;
  printf("level = %d, totalUtilRate = %f\n\n", level, totalUtilRate);
  for (int i = 0; i < 5; ++i) {
    // for practical reasons, one utilrate is at most twice of another
    sumUtilRate += utilRates[i] = (rand() % 101 + 50) / 350.0 * totalUtilRate;
  }
  if (sumUtilRate > totalUtilRate) {
    for (int i = 0; i < 5; ++i) {
      utilRates[i] /= sumUtilRate / totalUtilRate;
    }
  }

  pthread_data_t pthreadDataMain[5];
  for (int i = 0; i < 5; ++i) {
    pthreadDataGenerator(&pthreadDataMain[i], utilRates[i], level);
    pthreadDataMain[i].prio = 99 - i;
  }
  printf("\n");

  pthread_t pthreads[5];
  for (int i = 0; i < 5; ++i)
    pthread_create(&pthreads[i], NULL, threadFunc, (void *)&pthreadDataMain[i]);
  for (int i = 0; i < 5; ++i)
    pthread_join(pthreads[i], NULL);
  return 0;
}
