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

void *threadFunc(void *_tidPtr) {
  int _tid = *(int *)_tidPtr;
  long ddlusec = ddls[_tid] * 1000; // convert ddl to microsecond, consistent with duration

  // set scheduling config
  struct sched_param schedParam;
  schedParam.sched_priority = prios[_tid];
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
    pthread_mutex_lock(&pthrdMuts[_tid]);
    cpuTaskFunc(cpuTaskLens[_tid][0]);
    gpuTaskFunc<<<2, 1024, 0, streams[_tid]>>>(deviceData[_tid], gpuTaskLens[_tid][0]);
    debugCall(cudaStreamSynchronize(streams[_tid]));
    cpuTaskFunc(cpuTaskLens[_tid][1]);
    gpuTaskFunc<<<2, 1024, 0, streams[_tid]>>>(deviceData[_tid], gpuTaskLens[_tid][1]);
    debugCall(cudaStreamSynchronize(streams[_tid]));
    cpuTaskFunc(cpuTaskLens[_tid][2]);
    // unlock after finishing tasks
    pthread_mutex_unlock(&pthrdMuts[_tid]);

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

void prioGen(vector<vector<int>> &array, int n) {

  int row = 0;
  vector<int> c(n, 0);
  vector<int> A = {90, 92, 94, 96, 98};
  array[row] = A;
  ++row;
  int i = 0;
  while (i < n) {
    if (c[i] < i) {
      if (i / 2 * 2 == i) {
        swap(A[0], A[i]);
      } else {
        swap(A[c[i]], A[i]);
      }
      array[row] = A;
      ++row;
      ++c[i];
      i = 0;
    } else {
      c[i] = 0;
      ++i;
    }
  }
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

  // prepare data
  for (int i = 0; i < PTHREAD_NUM; ++i) {
    hostData[i] = (float *)malloc(2048 * sizeof(float));
    memset(hostData[i], 0, 2048 * sizeof(float));
    debugCall(cudaMalloc((void **)&deviceData[i], 2048 * sizeof(float)));
    debugCall(cudaMemcpy(deviceData[i], hostData[i], 2048 * sizeof(float), cudaMemcpyHostToDevice));
  }

  // read in parameters
  ifstream pthreadData;
  pthreadData.open("pthreadData.dat");
  for (int i = 0; i < PTHREAD_NUM; ++i) {
    pthreadData >> utilRates[i] >> ddls[i] >> cpuTaskLens[i][0] >> cpuTaskLens[i][1] >> cpuTaskLens[i][2] >>
        gpuTaskLens[i][0] >> gpuTaskLens[i][1];
  }
  pthreadData.close();

  // set priorities
  vector<vector<int>> array(120);
  prioGen(array, 5);
  int prioLine = atoi(argv[1]);
  prios[0] = array[prioLine][0];
  prios[1] = array[prioLine][1];
  prios[2] = array[prioLine][2];
  prios[3] = array[prioLine][3];
  prios[4] = array[prioLine][4];

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

  // create pthreads
  int _tids[PTHREAD_NUM];
  for (int i = 0; i < PTHREAD_NUM; ++i) {
    _tids[i] = i;
    pthread_create(&pthreads[i], NULL, threadFunc, (void *)&_tids[i]);
  }
  for (int i = 0; i < PTHREAD_NUM; ++i) {
    pthread_join(pthreads[i], NULL);
  }

  // successful scheduling
  // clean memory
  for (int i = 0; i < PTHREAD_NUM; ++i) {
    free(hostData[i]);
    debugCall(cudaFree(deviceData[i]));
  }
  // print info
  for (int i = 0; i < PTHREAD_NUM; ++i) {
    printf("%f %f ", utilRates[i], ddls[i]);
    printf("%f %f %f %f ", cpuTaskLens[i][0], cpuTaskLens[i][1], cpuTaskLens[i][2],
           cpuTaskLens[i][0] + cpuTaskLens[i][1] + cpuTaskLens[i][2]);
    printf("%f %f %f ", gpuTaskLens[i][0], gpuTaskLens[i][1], gpuTaskLens[i][0] + gpuTaskLens[i][1]);
    printf("%d ", prios[i]); // notice, no \n, for cooperation with driver.sh
  }
  return 0;
}
