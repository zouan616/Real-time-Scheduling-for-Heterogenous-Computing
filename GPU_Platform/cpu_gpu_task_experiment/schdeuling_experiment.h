#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

/* === A program with multiple cpu & gpu tasks activated in each pthread ===
All time variables are in unit microsecond
*/

#define debugCall(F)                                                           \
if ((F) != cudaSuccess){                                                    \
    printf("Error at line %d: %s\n", __LINE__,                                 \
            cudaGetErrorString(cudaGetLastError()));                            \
    exit(-1);                                                                  \
};

#define CPU_UNIT_TASK ((float)15872)// the number of iterations for the cpu task to run 1 millisecond
#define GPU_UNIT_TASK ((float)52224)// the number of iterations for the cpu task to run 1 millisecond


#define CPU_TASK_NUM (3)
#define GPU_TASK_NUM (3)
#define UTIL_RATE (0.4)

struct pthread_data_t
{
    int* cpuTaskLen;//time of a cpu task in millisecond
    int* gpuTaskLen;//time of a gpu task in millisecond
    int ddl;//time of ddl in millisecond
};
