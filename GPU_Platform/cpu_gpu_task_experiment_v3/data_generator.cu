#include "scheduling_experiment.h"

// USAGE: ./data_generator [total utility rate] [level] [scale]

// write generated data into pthreadData.dat

// total utility rate: 0.1, 0.2 ... 2.0
// when passed in, multiply by 100, e.g. 10, 20 ... 200
float totalUtilRate;

// level = 1 or 2 or 3
// level 1: S = rand(0.01, 0.1) * (T - C)
// level 2: S = rand(0.1, 0.6) * (T - C)
// level 3: S = rand(0.6, 1) * (T - C)
int level;

// scale = 1 or 2 or 3
// scale 1: 3 cpu + 2 gpu tasks in a batch
// scale 2: 5 cpu + 4 gpu tasks in a batch
// scale 3: 10 cpu + 9 gpu tasks in a batch
int scale;

void pthreadDataGen(int _tid) {
  // C: sum cpu tasks
  // S: sum gpu tasks
  // T: deadline
  float C = 0, S = 0, T = 0;

  // cpuTaskLens
  for (int i = 0; i < cpuTaskNum; ++i) {
    C += cpuTaskLens[_tid][i] = rand() % 10 + 1;
  }

  // ddl
  T = ddls[_tid] = C / utilRates[_tid];

  // gpuTaskLens
  // for practical reasons, length of one gpu task is at most twice of the other's
  switch (level) {
  case 1:
    S = (rand() % 91 + 10) / 1000.0 * (T - C);
    break;
  case 2:
    S = (rand() % 101 + 20) / 200.0 * (T - C);
    break;
  case 3:
    S = (rand() % 101 + 150) / 250.0 * (T - C);
    break;
  default:
    break;
  }
  float tmp = 0;
  for (int i = 0; i < gpuTaskNum; ++i) {
    tmp += gpuTaskLens[_tid][i] = rand() % 101 + 100;
  }
  for (int i = 0; i < gpuTaskNum; ++i) {
    gpuTaskLens[_tid][i] /= tmp / S;
  }
}

int main(int argc, char **argv) {
  ofstream pthreadData;
  pthreadData.open("pthreadData.dat");
  pthreadData.close();
  srand((unsigned)time(NULL));

  totalUtilRate = atoi(argv[1]) / 100.0;
  level = atoi(argv[2]);
  scale = atoi(argv[3]);

  // cpu/gpu tasks
  switch (scale) {
  case 1:
    cpuTaskNum = 3;
    gpuTaskNum = 2;
    break;
  case 2:
    cpuTaskNum = 5;
    gpuTaskNum = 4;
    break;
  case 3:
    cpuTaskNum = 10;
    gpuTaskNum = 9;
    break;
  }

  pthreadData.open("pthreadData.dat", std::ios_base::app);
  pthreadData << cpuTaskNum << " " << gpuTaskNum << " ";

  // util rate
  float sumUtilRate = 0;
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    // for practical reasons, one utilrate is at most twice of another
    sumUtilRate += utilRates[_tid] = rand() % 101 + 100;
  }
  for (int _tid = 0; _tid < PTHREAD_NUM; ++_tid) {
    utilRates[_tid] /= sumUtilRate / totalUtilRate;
    pthreadDataGen(_tid);
    pthreadData << utilRates[_tid] << " " << ddls[_tid] << " ";
    for (int i = 0; i < cpuTaskNum; ++i) {
      pthreadData << cpuTaskLens[_tid][i] << " ";
    }
    for (int i = 0; i < gpuTaskNum; ++i) {
      pthreadData << gpuTaskLens[_tid][i] << " ";
    }
  }

  pthreadData << endl;
  pthreadData.close();
  return 0;
}
