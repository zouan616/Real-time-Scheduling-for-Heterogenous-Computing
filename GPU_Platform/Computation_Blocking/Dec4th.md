# How different CUDA programs block each other (Dec. 4th)

## Current Progress

1. From the example

``` C++
__global__ void add(int n, float *x, float *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x * 2;
  for (int i = index; i < n; i += stride)
    y[i] += x[i];
}
...
add<<<2, 1024>>>(N, x, y);
```
The $i$ th kernel calculates all array entries with index $1024k+i$.

2. Run ``hello2.cu``, things output normally. However, run ``lagger.cu``, on another terminal, then run ``hello2.cu``, the program exits with no output. We can see that blocking GPU memory can cause programs to abort.
```
./hello2
line 21: out of memory
line 23: out of memory
```

3. Run ``hello.cu`` and ``bye.cu`` together, the programs output alternatively. Part of the output is:

```
// line 1 to line 3809 is output of hello.cu
line 3809: Hello, this is indBye this is index 256 at block 15, N is 0
// line 3809 to line 4122 is output of bye.cu
line 4122: Bye this is index 89 at block 12, N iHello, this is index 224 at block 15, N is 0
```

Turns out that programs running together can abort each other. There is no clear sign of the pattern how they abort.

## Problems

1. The Visual Profiler ``nvvp`` can open, but some profiling information are missing. We wish to have some materials to learn ``nvvp``
2. Hopefully we can use ``ssh`` to connect to the remote host.


## Programs used

- hello.cu
```C++
#include <stdio.h>
#include <math.h>
#include <cuda_profiler_api.h>


__global__ void hello(unsigned long long n, int *x)
{
    int index = threadIdx.x;
    int stride = blockDim.x * blockIdx.x;

    for (int i = 0;i<1024;i++)
    {
        unsigned long long N = i * (1 << 10);
        printf("Hello, this is index %d at block %d, N is %d\n",index,blockIdx.x,N);
    }
}

int main(void)
{
    cudaProfilerStart();
    unsigned long long N = 1 << 20;
    size_t size = N * sizeof(int);
    int *h_A = (int *)malloc(size);

    int *d_A;

    printf("line 21: %s\n",cudaGetErrorString(cudaMalloc(&d_A, size)));

    printf("line 23: %s\n",cudaGetErrorString(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice)));

    hello<<<512, 1024>>>(N, d_A);

    cudaFree(d_A);
    cudaProfilerStop();
    return 0;
}
```

- bye.cu

```C++
#include <stdio.h>
#include <math.h>
#include <cuda_profiler_api.h>


__global__ void hello(unsigned long long n, int *x)
{
    int index = threadIdx.x;
    int stride = blockDim.x * blockIdx.x;

    for (int i = 0;i<1024;i++)
    {
        unsigned long long N = i * (1 << 10);
        printf("Bye this is index %d at block %d, N is %d\n",index,blockIdx.x,N);
    }
}

int main(void)
{
    cudaProfilerStart();
    unsigned long long N = 1 << 20;
    size_t size = N * sizeof(int);
    int *h_A = (int *)malloc(size);

    int *d_A;

    printf("line 21: %s\n",cudaGetErrorString(cudaMalloc(&d_A, size)));

    printf("line 23: %s\n",cudaGetErrorString(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice)));

    hello<<<512, 1024>>>(N, d_A);

    cudaFree(d_A);
    cudaProfilerStop();
    return 0;
}
```

- hello2.cu

```C++
#include <stdio.h>
#include <math.h>
#include <cuda_profiler_api.h>


__global__ void hello(unsigned long long n, int *x)
{
    int index = threadIdx.x;
    int stride = blockDim.x * blockIdx.x;
    printf("Hello, this is index %d at block %d\n",index,blockIdx.x);
}

int main(void)
{
    cudaProfilerStart();
    unsigned long long N = 1 << 20;
    size_t size = N * sizeof(int);
    int *h_A = (int *)malloc(size);

    int *d_A;

    printf("line 21: %s\n",cudaGetErrorString(cudaMalloc(&d_A, size)));

    printf("line 23: %s\n",cudaGetErrorString(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice)));

    hello<<<1, 1024>>>(N, d_A);

    cudaFree(d_A);
    cudaProfilerStop();
    return 0;
}
```