/*
sudo ./data_generator [totalUtilRate] [scale]
[totalUtilRate]:
  an integer from 10 to 400,
  will be scaled to 0.1 ~ 4 when passed in
[scale]:
  1 -> 3 cpu + 2 gpu tasks in a batch
  2 -> 5 cpu + 4 gpu tasks in a batch
  3 -> 10 cpu + 9 gpu tasks in a batch
*/

#include "scheduling_experiment.h"

int main(int argc, char **argv) {
  float totalUtilRate = atoi(argv[1]) / 100.0;
  int scale = atoi(argv[2]);
  pthreadDataGen(totalUtilRate, scale);
  pthreadDataWrite();
  return 0;
}
