#include "scheduling_experiment.h"

int level;                    // 1, 2, 3
float totalUtilRate;          // total utility rate: 0.1, 0.2 ... 2.0

void pthreadDataGen(int _tid) {
  float C = 0, T = 0;

  // cpuTaskLens
  for (int i = 0; i < CPU_TASK_NUM; ++i) {
    C += cpuTaskLens[_tid][i] = rand() % 10 + 1;
  }

  // ddl
  T = ddls[_tid] = C / utilRates[_tid];

  // gpuTaskLens
  // for practical reasons, length of one gpu task is at most twice of the other's
  switch (level) {
  case 1: // S = rand(0.01, 0.1) * (T - C)
    gpuTaskLens[_tid][0] = (rand() % 91 + 10) / 1000.0 * (T - C);
    gpuTaskLens[_tid][1] = (rand() % 51 + 50) / 150.0 * gpuTaskLens[_tid][0];
    gpuTaskLens[_tid][0] -= gpuTaskLens[_tid][1];
    break;
  case 2: // S = rand(0.1, 0.6) * (T - C)
    gpuTaskLens[_tid][0] = (rand() % 101 + 20) / 200.0 * (T - C);
    gpuTaskLens[_tid][1] = (rand() % 51 + 50) / 150.0 * gpuTaskLens[_tid][0];
    gpuTaskLens[_tid][0] -= gpuTaskLens[_tid][1];
    break;
  case 3: // S = rand(0.6, 1) * (T - C)
    gpuTaskLens[_tid][0] = (rand() % 101 + 150) / 250.0 * (T - C);
    gpuTaskLens[_tid][1] = (rand() % 51 + 50) / 150.0 * gpuTaskLens[_tid][0];
    gpuTaskLens[_tid][0] -= gpuTaskLens[_tid][1];
    break;
  default:
    break;
  }

  // printf("util = %f, ddl = %f\n", utilRates[_tid], ddls[_tid]);
  // printf("c0 = %f, c1 = %f, c2 = %f\n", cpuTaskLens[_tid][0], cpuTaskLens[_tid][1], cpuTaskLens[_tid][2]);
  // printf("g0 = %f, g1 = %f\n\n", gpuTaskLens[_tid][0], gpuTaskLens[_tid][1]);
    ofstream pthreadData;
  pthreadData.open("pthreadData.dat",std::ios_base::app);
      pthreadData << utilRates[_tid] <<" "<< ddls[_tid] << " "<<cpuTaskLens[_tid][0] <<" "<< cpuTaskLens[_tid][1] <<" "<< cpuTaskLens[_tid][2]
                << " "<<gpuTaskLens[_tid][0] << " "<<gpuTaskLens[_tid][1] << endl;
  pthreadData.close();

}

///////////////////////////
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
  ofstream pthreadData;
  pthreadData.open("pthreadData.dat");
  pthreadData.close();
  srand((unsigned)time(NULL));
  totalUtilRate = atoi(argv[1]) / 100.0;
  level = atoi(argv[2]);

  float sumUtilRate = 0;
  // printf("level = %d, totalUtilRate = %f\n\n", level, totalUtilRate);
  for (int i = 0; i < PTHREAD_NUM; ++i) {
    // for practical reasons, one utilrate is at most twice of another
    sumUtilRate += utilRates[i] = rand() % 101 + 100;
  }
  for (int i = 0; i < PTHREAD_NUM; ++i) {
    utilRates[i] /= sumUtilRate / totalUtilRate;
    // printf("util %d = %f\n", i, utilRates[i]);
    pthreadDataGen(i);

  }
  return 0;
}