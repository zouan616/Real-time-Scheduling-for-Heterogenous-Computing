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
  long ddlusec = ddls[_tid] * 1000; // convert ddl to microsecond, consistent with duration

  // pin to either core 4 or 5
  cpu_set_t cpuSet;
  CPU_ZERO(&cpuSet);
  CPU_SET(4, &cpuSet);
  CPU_SET(5, &cpuSet);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuSet);

  // set scheduling config
  struct sched_param schedParam;
  schedParam.sched_priority = prios[_tid];
  cout<< "sched"<<pthread_setschedparam(pthread_self(), SCHED_FIFO, &schedParam)<<endl;

  struct timeval startTime;
  struct timeval endTime;
  long duration; // microsecond

  // PERFORM TASKS, 100 TIMES
  for (int i = 0; i < 100; ++i) {

    gettimeofday(&startTime, NULL);
    for (int j = 0; j < gpuTaskNum; ++j) {
      cpuTaskFunc(cpuTaskLens[_tid][j]);
      // gpuTaskFunc<<<2, 1024, 0, streams[_tid]>>>(_tid, gpuTaskLens[_tid][j]);
      // debugCall(cudaStreamSynchronize(streams[_tid]));
      usleep(gpuTaskLens[_tid][j]*1000);
    }
    cpuTaskFunc(cpuTaskLens[_tid][cpuTaskNum - 1]);
    gettimeofday(&endTime, NULL);

    if (timeExceeded)
      return NULL;

    duration = endTime.tv_sec * 1000000 + endTime.tv_usec - (startTime.tv_sec * 1000000 + startTime.tv_usec);
    if (duration > ddlusec) {
      timeExceeded = 1;
      return NULL;
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
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    debugCall(cudaStreamCreate(&streams[_tid]));
  }
  srand((unsigned)time(NULL));
  // instead of default busy waiting (polling)
  // now cudaStreamSynchronize will release cpu
  // cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync); // TODO

  // prepare device data
  debugCall(cudaMemcpyToSymbol(deviceData, hostData, PTHREAD_NUM * 5 * sizeof(float)));

  // read in parameters
  ifstream pthreadData;
  pthreadData.open("pthreadData.dat");
  pthreadData >> cpuTaskNum >> gpuTaskNum;
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    pthreadData >> utilRates[_tid] >> ddls[_tid];
    for (int i = 0; i < cpuTaskNum; ++i) {
      pthreadData >> cpuTaskLens[_tid][i];
    }
    for (int i = 0; i < gpuTaskNum; ++i) {
      pthreadData >> gpuTaskLens[_tid][i];
    }
  }
  pthreadData.close();

  // print info
  // for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
  //   cout << "util " << _tid << ": " << utilRates[_tid] << endl << "ddl " << _tid << ": " << ddls[_tid] << endl;
  //   for (int i = 0; i < cpuTaskNum; ++i) {
  //     cout << cpuTaskLens[_tid][i] << " ";
  //   }
  //   cout << '\n';
  //   for (int i = 0; i < gpuTaskNum; ++i) {
  //     cout << gpuTaskLens[_tid][i] << " ";
  //   }
  //   cout << '\n' << endl;
  // }

  // set priorities
  vector<vector<int>> array(120);
  prioGen(array, 5);
  int prioLine = atoi(argv[1]);
  for (int i = 0; i < PTHREAD_NUM; ++i) {
    prios[i] = array[prioLine][i];
  }

  // warm up gpu, should be removed in later versions
  // for (int i = 0; i < 250; ++i) {
  //   for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
  //     gpuTaskFunc<<<2, 1024, 0, streams[_tid]>>>(_tid, 1);
  //   }
  // }
  // debugCall(cudaDeviceSynchronize());

  // create pthreads
  int _tids[PTHREAD_NUM];
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    _tids[_tid] = _tid;
    pthread_create(&pthreads[_tid], NULL, threadFunc, (void *)&_tids[_tid]);
  }
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    pthread_join(pthreads[_tid], NULL);
  }
  cudaDeviceReset();
  if (timeExceeded)
    exit(10);
  return 0;
}
