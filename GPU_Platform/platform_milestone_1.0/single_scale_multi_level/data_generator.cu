/*
sudo ./data_generator [totalUtilRate] [level] [scale]
[totalUtilRate]:
  an integer from 10 to 200,
  will be scaled to 0.1 ~ 2 when passed in
[level]:
  0 -> ddl = (C + S) / u, s = (int) rand(1, 10)
  1 -> ddl = C / u, S = rand(0.01, 0.1) * (ddl - C)
  2 -> ddl = C / u, S = rand(0.1, 0.6) * (ddl - C)
  3 -> ddl = C / u, S = rand(0.6, 1) * (ddl - C)
[scale]:
  1 -> 3 cpu + 2 gpu tasks in a batch
  2 -> 5 cpu + 4 gpu tasks in a batch
  3 -> 10 cpu + 9 gpu tasks in a batch
*/

#include "scheduling_experiment.h"

int main(int argc, char **argv) {
  float totalUtilRate = atoi(argv[1]) / 100.0;
  int level = atoi(argv[2]);
  int scale = atoi(argv[3]);
  pthreadDataGen(totalUtilRate, level, scale);
  pthreadDataWrite();
  return 0;
}
