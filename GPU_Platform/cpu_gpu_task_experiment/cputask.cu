#include "schdeuling_experiment.h"

void cpuTaskFunc(float* cpu_A, float* cpu_B,int cpuTaskLen)
{
    for(long i=0;i<cpuTaskLen*CPU_UNIT_TASK;++i)
    {
        cpu_B[i] += 9875.4321/cpu_A[i];
        cpu_B[i] += 9875.4321/cpu_A[i];
        cpu_B[i] += 9875.4321/cpu_A[i];
        cpu_B[i] += 9875.4321/cpu_A[i];
        cpu_B[i] += 98888/cpu_A[i];
        cpu_B[i] += 9875.4321/cpu_A[i];
        cpu_B[i] += 9875.4321/cpu_A[i];
        cpu_B[i] += 9875.4321/cpu_A[i];
        cpu_B[i] += 9875.4321/cpu_A[i];
        cpu_B[i] += 98888/cpu_A[i];
        cpu_B[i] += 9875.4321/cpu_A[i];
        cpu_B[i] += 9875.4321/cpu_A[i];
        cpu_B[i] += 9875.4321/cpu_A[i];
        cpu_B[i] += 9875.4321/cpu_A[i];
        cpu_B[i] += 9875.4321;
        cpu_B[i] += 9875.4321;
        cpu_B[i] += 98888/cpu_A[i];
    }
}
void *thread_func(void *pd)
{
    // set scheduling policy
    struct sched_param schedParam;
    schedParam.sched_priority = 99;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &schedParam);

    cpu_set_t cpuSet;    // create a cpu set
    CPU_ZERO(&cpuSet);   // initialize
    CPU_SET(4, &cpuSet); // add core 4 to cpu set
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuSet);

    struct timeval startTime;
    struct timeval endTime;
    long duration; // unit: microsecond

    // prepare data
    float *cpu_A = (float *)malloc((int)(CPU_UNIT_TASK * sizeof(float))); // cpu A
    float *cpu_B = (float *)malloc((int)(CPU_UNIT_TASK * sizeof(float))); // cpu B

    printf("Generating data\n");
    for (int i = 0; i < CPU_UNIT_TASK; ++i)
    {
        cpu_A[i] = 1075.329;
        cpu_B[i] = 1234.456;
    }


    while(1)
    {
        gettimeofday(&startTime, NULL);

        cpuTaskFunc(cpu_A,cpu_B,1);

        gettimeofday(&endTime, NULL);

        duration = endTime.tv_sec * 1000000 + endTime.tv_usec -
                    (startTime.tv_sec * 1000000 + startTime.tv_usec);
                    printf("duration is %ld\n",duration);
    }

    return NULL;
}

int main(int argc, char **argv)
{
    pthread_data_t pthreadDataMain[8];
    pthread_t pthreads[8];

    pthread_create(&pthreads[0], NULL, thread_func,
                    (void *)&pthreadDataMain[0]);
    pthread_join(pthreads[0], NULL);

    return 0;
}