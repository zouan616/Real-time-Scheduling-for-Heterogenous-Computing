/* 
sudo ./data_generator [totalUtilRate]
[totalUtilRate]:
  an integer from 10 to 400,
  will be scaled to 0.1 ~ 4 when passed in
*/

#include "scheduling_experiment.h"

int main(int argc, char **argv) {
  float totalUtilRate = atoi(argv[1]) / 100.0;
  pthreadDataGen(totalUtilRate);
  pthreadDataWrite();
  return 0;
}
