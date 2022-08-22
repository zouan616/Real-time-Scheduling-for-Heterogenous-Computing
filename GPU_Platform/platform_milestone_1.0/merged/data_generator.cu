/*
sudo ./data_generator 1 [totalUtilRate] [*] [*]
sudo ./data_generator 2 [totalUtilRate] [scale] [*]
sudo ./data_generator 3 [totalUtilRate] [scale] [gpuLenLevel]
*/

#include "scheduling_experiment.h"

int main(int argc, char **argv) {
  if (atoi(argv[1]) == 1) {
    pthreadDataGen_Benchmark(atof(argv[2]) / 100);
  } else if (atoi(argv[1]) == 2) {
    pthreadDataGen_Scaled(atof(argv[2]) / 100, atoi(argv[3]));
  } else if (atoi(argv[1]) == 3) {
    pthreadDataGen_Scaled_GLenLeveled(atof(argv[2]) / 100, atoi(argv[3]), atoi(argv[4]));
  }
  pthreadDataWrite();
  pthreadDataPrint();
  return 0;
}
