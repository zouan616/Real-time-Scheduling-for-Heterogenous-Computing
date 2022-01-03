一个 stream 上的计算按 kernel task 提交的顺序串行执行

<details>
<summary>debugCall</summary>

```cpp
#define debugCall(F)                                                           \
  if ((F) != cudaSuccess) {                                                    \
    printf("Error at line %d: %s\n", __LINE__,                                 \
           cudaGetErrorString(cudaGetLastError()));                            \
    exit(-1);                                                                  \
  }
```
</details>

<details>
<summary>printDeviceProperties</summary>

```cpp
void printDeviceProp() {

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  std::cout << "deviceCount: " << deviceCount << "\n\n";
  if (deviceCount == 0) {
    std::cout << "Error: no devices supporting CUDA.\n";
    return;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  std::cout << "name: " << prop.name << "\n";
  std::cout << "totalGlobalMem: " << prop.totalGlobalMem << "\n";
  std::cout << "regsPerBlock: " << prop.regsPerBlock << "\n";
  std::cout << "warpSize: " << prop.warpSize << "\n";
  std::cout << "memPitch: " << prop.memPitch << "\n\n";

  std::cout << "一个线程块中可使用的最大共享内存\n";
  std::cout << "sharedMemPerBlock: " << prop.sharedMemPerBlock << "\n\n";

  std::cout << "一个线程块中可包含的最大线程数量\n";
  std::cout << "maxThreadsPerBlock: " << prop.maxThreadsPerBlock << "\n\n";

  std::cout << "多维线程块数组中每一维可包含的最大线程数量\n";
  std::cout << "maxThreadsDim[0]: " << prop.maxThreadsDim[0] << "\n";
  std::cout << "maxThreadsDim[1]: " << prop.maxThreadsDim[1] << "\n";
  std::cout << "maxThreadsDim[2]: " << prop.maxThreadsDim[2] << "\n\n";

  std::cout << "一个线程格中每一维可包含的最大线程块数量\n";
  std::cout << "maxGridSize[0]: " << prop.maxGridSize[0] << "\n";
  std::cout << "maxGridSize[1]: " << prop.maxGridSize[1] << "\n";
  std::cout << "maxGridSize[2]: " << prop.maxGridSize[2] << "\n\n";

  std::cout << "clockRate: " << prop.clockRate << "\n";
  std::cout << "totalConstMem: " << prop.totalConstMem << "\n";
  std::cout << "textureAlignment: " << prop.textureAlignment << "\n\n";

  std::cout << "计算能力：" << prop.major << "." << prop.minor << "\n\n";

  std::cout << "minor: " << prop.minor << "\n";
  std::cout << "texturePitchAlignment: " << prop.texturePitchAlignment << "\n";
  std::cout << "deviceOverlap: " << prop.deviceOverlap << "\n";
  std::cout << "multiProcessorCount: " << prop.multiProcessorCount << "\n";
  std::cout << "kernelExecTimeoutEnabled: " << prop.kernelExecTimeoutEnabled
            << "\n";
  std::cout << "integrated: " << prop.integrated << "\n";
  std::cout << "canMapHostMemory: " << prop.canMapHostMemory << "\n";
  std::cout << "computeMode: " << prop.computeMode << "\n";
  std::cout << "maxTexture1D: " << prop.maxTexture1D << "\n";
  std::cout << "maxTexture1DMipmap: " << prop.maxTexture1DMipmap << "\n";
  std::cout << "maxTexture1DLinear: " << prop.maxTexture1DLinear << "\n";
  std::cout << "maxTexture2D: " << prop.maxTexture2D << "\n";
  std::cout << "maxTexture2DMipmap: " << prop.maxTexture2DMipmap << "\n";
  std::cout << "maxTexture2DLinear: " << prop.maxTexture2DLinear << "\n";
  std::cout << "maxTexture2DGather: " << prop.maxTexture2DGather << "\n";
  std::cout << "maxTexture3D: " << prop.maxTexture3D << "\n";
  std::cout << "maxTexture3DAlt: " << prop.maxTexture3DAlt << "\n";
  std::cout << "maxTextureCubemap: " << prop.maxTextureCubemap << "\n";
  std::cout << "maxTexture1DLayered: " << prop.maxTexture1DLayered << "\n";
  std::cout << "maxTexture2DLayered: " << prop.maxTexture2DLayered << "\n";
  std::cout << "maxTextureCubemapLayered: " << prop.maxTextureCubemapLayered
            << "\n";
  std::cout << "maxSurface1D: " << prop.maxSurface1D << "\n";
  std::cout << "maxSurface2D: " << prop.maxSurface2D << "\n";
  std::cout << "maxSurface3D: " << prop.maxSurface3D << "\n";
  std::cout << "maxSurface1DLayered: " << prop.maxSurface1DLayered << "\n";
  std::cout << "maxSurface2DLayered: " << prop.maxSurface2DLayered << "\n";
  std::cout << "maxSurfaceCubemap: " << prop.maxSurfaceCubemap << "\n";
  std::cout << "maxSurfaceCubemapLayered: " << prop.maxSurfaceCubemapLayered
            << "\n";
  std::cout << "surfaceAlignment: " << prop.surfaceAlignment << "\n";
  std::cout << "concurrentKernels: " << prop.concurrentKernels << "\n";
  std::cout << "ECCEnabled: " << prop.ECCEnabled << "\n";
  std::cout << "pciBusID: " << prop.pciBusID << "\n";
  std::cout << "pciDeviceID: " << prop.pciDeviceID << "\n";
  std::cout << "pciDomainID: " << prop.pciDomainID << "\n";
  std::cout << "tccDriver: " << prop.tccDriver << "\n";
  std::cout << "asyncEngineCount: " << prop.asyncEngineCount << "\n";
  std::cout << "unifiedAddressing: " << prop.unifiedAddressing << "\n";
  std::cout << "memoryClockRate: " << prop.memoryClockRate << "\n";
  std::cout << "memoryBusWidth: " << prop.memoryBusWidth << "\n";
  std::cout << "l2CacheSize: " << prop.l2CacheSize << "\n";
  std::cout << "maxThreadsPerMultiProcessor: "
            << prop.maxThreadsPerMultiProcessor << "\n";
  std::cout << "streamPrioritiesSupported: " << prop.streamPrioritiesSupported
            << "\n";
  std::cout << "globalL1CacheSupported: " << prop.globalL1CacheSupported
            << "\n";
  std::cout << "localL1CacheSupported: " << prop.localL1CacheSupported << "\n";
  std::cout << "sharedMemPerMultiprocessor: " << prop.sharedMemPerMultiprocessor
            << "\n";
  std::cout << "regsPerMultiprocessor: " << prop.regsPerMultiprocessor << "\n";
  std::cout << "isMultiGpuBoard: " << prop.isMultiGpuBoard << "\n";
  std::cout << "multiGpuBoardGroupID: " << prop.multiGpuBoardGroupID << "\n";
  std::cout << "singleToDoublePrecisionPerfRatio: "
            << prop.singleToDoublePrecisionPerfRatio << "\n";
  std::cout << "pageableMemoryAccess: " << prop.pageableMemoryAccess << "\n";
  std::cout << "concurrentManagedAccess: " << prop.concurrentManagedAccess
            << "\n";
}
```
</details>
