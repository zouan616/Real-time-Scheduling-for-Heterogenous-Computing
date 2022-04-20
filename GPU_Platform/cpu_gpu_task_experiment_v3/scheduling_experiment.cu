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
  // convert ddl to microsecond, consistent with duration
  long ddlusec = ddls[_tid] * 1000;

  // pin to either core 4 or 5
  cpu_set_t cpuSet;
  CPU_ZERO(&cpuSet);
  CPU_SET(4, &cpuSet);
  CPU_SET(5, &cpuSet);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuSet);

  // set scheduling config
  struct sched_param schedParam;
  schedParam.sched_priority = prios[_tid];
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &schedParam);

  struct timeval startTime;
  struct timeval endTime;
  long duration; // microsecond

  // MAIN LOOP
  for (int i = 0; i < 100; ++i) {
    cout<<_tid<<" loop "<< i<<endl;
    // launch a batch of tasks
    gettimeofday(&startTime, NULL);
    for (int j = 0; j < gpuTaskNum; ++j) {
      cpuTaskFunc(cpuTaskLens[_tid][j]);
      gpuTaskFunc<<<2, 1024, 0, streams[_tid]>>>(_tid, gpuTaskLens[_tid][j]);
      usleep(gpuTaskLens[_tid][j] * 1000);
      cudaStreamSynchronize(streams[_tid]);
    }
    cpuTaskFunc(cpuTaskLens[_tid][cpuTaskNum - 1]);
    gettimeofday(&endTime, NULL);
    cout<<_tid<<" batch finish"<<endl;

    // some other pthreads time exceeded
    if (timeExceeded) {
      isDone[_tid] = 1;
      cout<<_tid<<" done"<<endl;
      return NULL;
    }
    // current pthread time exceeded
    duration = endTime.tv_sec * 1000000 + endTime.tv_usec - (startTime.tv_sec * 1000000 + startTime.tv_usec);
    if (duration > ddlusec) {
      timeExceeded = 1;
      isDone[_tid] = 1;
      cout<<_tid<<" done"<<endl;
      return NULL;
    }
    // current pthread not time exceeded, sleep until deadline
    usleep(ddlusec - duration);
  }
  // current pthread successfully schedule
  isDone[_tid] = 1;
  cout<<_tid<<" done"<<endl;
  return NULL;
}

void prioGen(vector<vector<int>> &array, int n) {
  int row = 0;
  vector<int> c(n, 0);
  vector<int> A = {95, 96, 97, 98, 99};
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
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

  // prepare device data
  debugCall(cudaMemcpyToSymbol(deviceData, hostData, PTHREAD_NUM * 5 * sizeof(float)));

  // read parameters from pthreadData.dat
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
  for (int i = 0; i < 250; ++i) {
    for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
      gpuTaskFunc<<<2, 1024, 0, streams[_tid]>>>(_tid, 1);
    }
  }
  debugCall(cudaDeviceSynchronize());
  cout<< "warmed"<<endl;

  // create pthreads
  int _tids[PTHREAD_NUM];
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    _tids[_tid] = _tid;
    pthread_create(&pthreads[_tid], NULL, threadFunc, (void *)&_tids[_tid]);
  }

  // wait until all pthreads are done
  while (1) {
    int done = 1;
    for (int i = 0; i < PTHREAD_NUM; ++i) {
      done *= isDone[i];
    }
    if (done) {
      break;
    }
  }
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    pthread_join(pthreads[_tid], NULL);
  }
  cudaDeviceReset();
  exit(timeExceeded);
}
