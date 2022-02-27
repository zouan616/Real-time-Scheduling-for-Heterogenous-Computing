#include "schdeuling_experiment.h"

__global__ void gpuTaskFunc(float* d_A, float* d_B, int gpuTaskLen) // TODO
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

    // set scheduling policy
    struct sched_param schedParam;
    schedParam.sched_priority = 99;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &schedParam);

    cpu_set_t cpuSet;    // create a cpu set
    CPU_ZERO(&cpuSet);   // initialize
    CPU_SET(4, &cpuSet); // add core 4 to cpu set
    // now the thread will be assigned to either core
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuSet);

    struct timeval startTime;
    struct timeval endTime;
    long duration; // unit: microsecond

    // prepare data

    float *h_A = (float *)malloc(GPU_UNIT_TASK * sizeof(float)); // host A
    float *h_B = (float *)malloc(GPU_UNIT_TASK * sizeof(float)); // host B

    for (int i = 0; i < GPU_UNIT_TASK; ++i)
    {
        h_A[i] = 1075.329;
        h_B[i] = 1234.456;
    }

    float *d_A; //device A
    cudaMalloc((void**)&d_A, GPU_UNIT_TASK * sizeof(float));
    cudaMemcpy(d_A, h_A, GPU_UNIT_TASK * sizeof(float), cudaMemcpyHostToDevice);

    float *d_B; //device B
    cudaMalloc((void**)&d_B, GPU_UNIT_TASK * sizeof(float));
    cudaMemcpy(d_B, h_B, GPU_UNIT_TASK * sizeof(float), cudaMemcpyHostToDevice);


    cudaStream_t strm;
    debugCall(cudaStreamCreate(&strm));
    {
        while(1)
        {
            gettimeofday(&startTime, NULL);

            debugCall(cudaStreamSynchronize(strm));
            gpuTaskFunc<<<2, 1024, 0, strm>>>(d_A,d_B,1);
            debugCall(cudaStreamSynchronize(strm));
            gettimeofday(&endTime, NULL);
            duration = endTime.tv_sec * 1000000 + endTime.tv_usec -
                        (startTime.tv_sec * 1000000 + startTime.tv_usec);
                        printf("duration is %ld\n",duration);

        }


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