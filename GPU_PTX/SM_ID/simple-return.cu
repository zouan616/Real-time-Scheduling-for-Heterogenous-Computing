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
#define MAXN 128


static __device__ __inline__ uint32_t __mysmid(){    
  uint32_t smid;    
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));    
  return smid;}




//global function
__global__ void computation(float* x, float* y, float* z, int SM_num_start, int SM_num_end){

int SM_num;
SM_num = __mysmid();
printf("SM num: %d \n", SM_num);

int i = threadIdx.x;

if((SM_num_start <= SM_num)&&(SM_num <= SM_num_end))
{    
z[i] = x[i] + y[i];
//printf("SM num: %d \n", SM_num);
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

computation <<< 1 , 128 >>> ( d_A, d_B, d_C, 0, 0);

cudaMemcpy( h_C, d_C, size, cudaMemcpyDeviceToHost );

for ( i = 0 ; i < MAXN; i ++ ) {
//std::cout << h_A[i] << " " ;
//std::cout << h_B[i] << " " ;
std::cout << h_C[i] << " " ;
}

std::cout << std::endl;

return 0 ;
} 




