#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "simple-return.fatbin.c"
extern void __device_stub__Z11computationPfS_S_ii(float *, float *, float *, int, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z11computationPfS_S_ii(float *__par0, float *__par1, float *__par2, int __par3, int __par4){__cudaLaunchPrologue(5);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 24UL);__cudaSetupArgSimple(__par4, 28UL);__cudaLaunch(((char *)((void ( *)(float *, float *, float *, int, int))computation)));}
# 31 "simple-return.cu"
void computation( float *__cuda_0,float *__cuda_1,float *__cuda_2,int __cuda_3,int __cuda_4)
# 31 "simple-return.cu"
{__device_stub__Z11computationPfS_S_ii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4);
# 45 "simple-return.cu"
}
# 1 "simple-return.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T0) {  __nv_dummy_param_ref(__T0); __nv_save_fatbinhandle_for_managed_rt(__T0); __cudaRegisterEntry(__T0, ((void ( *)(float *, float *, float *, int, int))computation), _Z11computationPfS_S_ii, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
