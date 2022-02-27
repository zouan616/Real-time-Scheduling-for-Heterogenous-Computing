#include "schdeuling_experiment.h"

void pthreadDataGenerator(pthread_data_t *pd)
{
    int taskLenSum = 0;
    pd->cpuTaskLen = (int *)malloc(sizeof(int) * CPU_TASK_NUM);
    pd->gpuTaskLen = (int *)malloc(sizeof(int) * GPU_TASK_NUM);

    // Randomly assign the length of each cpu and gpu task
    for (int i = 0; i < CPU_TASK_NUM; ++i)
    {
        taskLenSum += (pd->cpuTaskLen[i] = rand() % 100 + 1);
        srand((unsigned)taskLenSum);
        printf("pd->cpuTaskLen[%d] = %d\n",i,pd->cpuTaskLen[i]);
    }
    for (int i = 0; i < GPU_TASK_NUM; ++i)
    {
        taskLenSum += (pd->gpuTaskLen[i] = rand() % 100 + 1);
        srand((unsigned)taskLenSum);
        printf("pd->gpuTaskLen[%d] = %d\n",i,pd->gpuTaskLen[i]);
    }
    // Determine ddl according to the total length of tasks
    pd->ddl = (int)(taskLenSum / UTIL_RATE);
    printf("pd->ddl = %d\n",pd->ddl);
}

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

__global__ void gpuTaskFunc(float* d_A, float* d_B, int gpuTaskLen)
{
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    long stride = blockDim.x * gridDim.x;
    while(i<gpuTaskLen*GPU_UNIT_TASK)
    {
        d_B[i] += 9875.4321/d_A[i];
        d_B[i] += 9875.4321/d_A[i];
        d_B[i] += 9875.4321/d_A[i];
        d_B[i] += 9875.4321/d_A[i];
        d_B[i] += 98888/d_A[i];
        d_B[i] += 9875.4321/d_A[i];
        d_B[i] += 9875.4321/d_A[i];
        d_B[i] += 9875.4321/d_A[i];
        d_B[i] += 9875.4321/d_A[i];
        d_B[i] += 98888/d_A[i];
        d_B[i] += 9875.4321/d_A[i];
        d_B[i] += 9875.4321/d_A[i];
        d_B[i] += 9875.4321/d_A[i];
        d_B[i] += 9875.4321/d_A[i];
        d_B[i] += 9875.4321;
        d_B[i] += 9875.4321;
        d_B[i] += 98888/d_A[i];
        i += stride;
    }
}

void *thread_func(void *pd)
{

    pthread_data_t *pthreadData = (pthread_data_t *)pd;
    int cpuTaskLen = *pthreadData->cpuTaskLen;
    int gpuTaskLen = *pthreadData->gpuTaskLen;
    int ddl = pthreadData->ddl;


    // set scheduling policy
    struct sched_param schedParam;
    schedParam.sched_priority = 99;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &schedParam);

    cpu_set_t cpuSet;    // create a cpu set
    CPU_ZERO(&cpuSet);   // initialize
    CPU_SET(4, &cpuSet); // add core 4 to cpu set
    CPU_SET(5, &cpuSet); // add core 5 to cpu set
    // now the thread will be assigned to either core
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuSet);

    struct timeval startTime;
    struct timeval endTime;
    long duration; // unit: microsecond

    // prepare data
    float *cpu_A = (float *)malloc(cpuTaskLen * CPU_UNIT_TASK * sizeof(float)); // cpu A
    float *cpu_B = (float *)malloc(cpuTaskLen * CPU_UNIT_TASK * sizeof(float)); // cpu B

    for (int i = 0; i < cpuTaskLen * CPU_UNIT_TASK; ++i)
    {
        cpu_A[i] = 1075.329;
        cpu_B[i] = 1234.456;
    }

    float *h_A = (float *)malloc(gpuTaskLen * GPU_UNIT_TASK * sizeof(float)); // host A
    float *h_B = (float *)malloc(gpuTaskLen * GPU_UNIT_TASK * sizeof(float)); // host B

    for (int i = 0; i < gpuTaskLen * GPU_UNIT_TASK; ++i)
    {
        h_A[i] = 1075.329;
        h_B[i] = 1234.456;
    }

    float *d_A; //device A
    cudaMalloc((void**)&d_A,gpuTaskLen * GPU_UNIT_TASK * sizeof(float));
    cudaMemcpy(d_A, h_A, gpuTaskLen * GPU_UNIT_TASK * sizeof(float), cudaMemcpyHostToDevice);

    float *d_B; //device B
    cudaMalloc((void**)&d_B,gpuTaskLen * GPU_UNIT_TASK * sizeof(float));
    cudaMemcpy(d_B, h_B, gpuTaskLen * GPU_UNIT_TASK * sizeof(float), cudaMemcpyHostToDevice);


    cudaStream_t strm;
    debugCall(cudaStreamCreate(&strm));


    while (1)
    {
        gettimeofday(&startTime, NULL);

        // perform cpu & gpu tasks
        for (int i = 0; i < CPU_TASK_NUM; ++i)
        {
            cpuTaskFunc(cpu_A,cpu_B,cpuTaskLen);

            debugCall(cudaStreamSynchronize(strm));
            gpuTaskFunc<<<2, 1024, 0, strm>>>(d_A,d_B,gpuTaskLen);
            debugCall(cudaStreamSynchronize(strm));

        }

        gettimeofday(&endTime, NULL);

        duration = endTime.tv_sec * 1000000 + endTime.tv_usec -
                    (startTime.tv_sec * 1000000 + startTime.tv_usec);

        printf("sleep time / ddl = %d/%d = %f\n",ddl * 1000 - duration,ddl*1000,((float)((ddl * 1000 - duration)/(float)(ddl*1000))));
        if (duration > ddl * 1000)
        {
            printf("Thread time exceeded, abort!\n");
            exit(0);
        }

        usleep(ddl * 1000 - duration);
    }

    free(cpu_A);
    free(cpu_B);
    return NULL;
}

int main(int argc, char **argv)
{
    srand( (unsigned)time(NULL) );
    pthread_data_t pthreadDataMain[8];
    printf("Generating pthreadData\n");
    for (int i = 0; i < 8; ++i)
    {
        pthreadDataGenerator(&pthreadDataMain[i]);
    }
    printf("End of Generating pthreadData\n\n");
    pthread_t pthreads[8];

    for(int i = 0;i<8;++i)
    pthread_create(&pthreads[i], NULL, thread_func,
                    (void *)&pthreadDataMain[i]);
    for(int i = 0;i<8;++i)
    pthread_join(pthreads[i], NULL);

    return 0;
}
