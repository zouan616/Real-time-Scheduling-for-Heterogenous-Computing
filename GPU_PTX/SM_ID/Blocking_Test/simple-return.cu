#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdint.h>
#include <iostream>
#include <stdlib.h>      
#include <time.h>  
#include <string.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <float.h>
#include <sys/time.h>
#include <fstream>
#include <algorithm>
#include <iostream>
#define MAXN 524288


static __device__ __inline__ uint32_t __mysmid(){    
  uint32_t smid;    
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));    
  return smid;}


//global function
__global__ void large_computation(float* x, float* y, float* z, int SM_num_start, int SM_num_end){

int SM_num;
SM_num = __mysmid();

if((SM_num_start <= SM_num)&&(SM_num <= SM_num_end))
{    
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    // Make sure we do not go out of bounds
    if (id < MAXN)
        for(int j = 0; j < 100000; j++)
        z[id] = x[id] + y[id] + j;
    }

}

//global function
__global__ void simple_return(float* x, float* y, float* z, int SM_num_start, int SM_num_end){

int SM_num;
SM_num = __mysmid();

if((SM_num_start <= SM_num)&&(SM_num <= SM_num_end))
{    
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    // Make sure we do not go out of bounds
    if (id < MAXN)
        for(int j = 0; j < 10; j++)
        z[id] = x[id] + y[id] + j;
    }

}



int main( int argc, char ** argv ) {
float h_A[MAXN], h_B[MAXN], h_C[MAXN];
float * d_A, * d_B, * d_C;
int i;
for ( i = 0 ; i < MAXN; i ++ ) {
h_A[i] = i;
h_B[i] = 1 ;
h_C[i] = 0 ;
}

int size = MAXN * sizeof ( float );
cudaMalloc( ( void ** ) & d_A, size );
cudaMalloc( ( void ** ) & d_B, size );
cudaMalloc( ( void ** ) & d_C, size );

cudaMemcpy( d_A, h_A, size, cudaMemcpyHostToDevice );
cudaMemcpy( d_B, h_B, size, cudaMemcpyHostToDevice );

// Number of threads in each thread block
int blockSize = 1024;
 
// Number of thread blocks in grid
int gridSize = (int)ceil((float)MAXN/blockSize);
 
// Execute the kernel
cudaStream_t stream1, stream2;
cudaStreamCreate ( &stream1) ;
cudaStreamCreate ( &stream2) ;

large_computation <<< 524288/1024, 1024, 0, stream1 >>> ( d_A, d_B, d_C, 1, 10);

simple_return <<< 524288/1024, 1024, 0, stream2 >>> ( d_A, d_B, d_C, -1, -1);



cudaMemcpy( h_C, d_C, size, cudaMemcpyDeviceToHost );

for ( i = 0 ; i < MAXN; i ++ ) {
//std::cout << h_A[i] << " " ;
//std::cout << h_B[i] << " " ;
//std::cout << h_C[i] << " " ;
}

//std::cout << std::endl;

return 0 ;
} 




