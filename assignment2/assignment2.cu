#include <iostream> 
#include <vector>
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n",cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void sum_GPU(float * device_dataIn1, float * device_dataIn2, float * device_dataIn3)
{
   int threadId = threadIdx.x + blockIdx.x * blockDim.x;
   device_dataIn1[threadId] = device_dataIn1[threadId] + device_dataIn2[threadId] + device_dataIn3[threadId];
}

void sum_CPU(float* host_dataCPU, float* host_dataIn1, float* host_dataIn2, float*  host_dataIn3, int DIM_X)
{
    for ( int x = 0; x < DIM_X; ++x )
    {
        host_dataCPU[x] = host_dataIn1[x] + host_dataIn2[x] + host_dataIn3[x];
    }
}

int main(int argc, char* argv[]) 
{

  // Dimensions of the 3D data
  int DIM_X = 4194304;

  // Definition of host and device pointers
  float * host_dataIn1, * host_dataIn2, * host_dataIn3, * host_dataCPU, * host_dataGPU;
  float * device_dataIn1, * device_dataIn2, * device_dataIn3;

  // Memory allocation for host data
  host_dataIn1 = new float [DIM_X];
  host_dataIn2 = new float [DIM_X];
  host_dataIn3 = new float [DIM_X];
  host_dataGPU = new float [DIM_X];
  host_dataCPU = new float [DIM_X];


  // Calculation of required spaces 
  const int DATA_BYTES  = (DIM_X) * sizeof(float);
  // Initialize host_dataIn with random numbers
  for ( int x = 0; x < DIM_X; ++x)
  {
      host_dataIn1 [x] = (rand() / (float)RAND_MAX * 100) + 1;
      host_dataIn2 [x] = (rand() / (float)RAND_MAX * 100) + 1;
      host_dataIn3 [x] = (rand() / (float)RAND_MAX * 100) + 1;
  }

  clock_t cpu_start = clock();
  sum_CPU(host_dataCPU, host_dataIn1, host_dataIn2, host_dataIn3, DIM_X);
  clock_t cpu_end = clock();

  // Allocate DEVICE arrays on GPU for dataIn
  gpuErrchk( cudaMalloc( ( void**) &device_dataIn1,  DATA_BYTES));
  gpuErrchk( cudaMalloc( ( void**) &device_dataIn2,  DATA_BYTES));
  gpuErrchk( cudaMalloc( ( void**) &device_dataIn3,  DATA_BYTES));

  clock_t htod_start, htod_end, dtoh_start, dtoh_end;
  clock_t gpu_start, gpu_end;

  htod_start = clock();
  gpuErrchk( cudaMemcpy( device_dataIn1, host_dataIn1, DATA_BYTES, cudaMemcpyHostToDevice));
  gpuErrchk( cudaMemcpy( device_dataIn2, host_dataIn2, DATA_BYTES, cudaMemcpyHostToDevice));
  gpuErrchk( cudaMemcpy( device_dataIn3, host_dataIn3, DATA_BYTES, cudaMemcpyHostToDevice));
  htod_end = clock();

  int blockX = 64;
  dim3 blockSize(blockX);

  int gridX = (DIM_X) / blockX;
  dim3 gridSize(gridX);

  gpu_start = clock();
  ///////////////////////////////////////////////////////////////////////////////
  // Kernel Call
  ///////////////////////////////////////////////////////////////////////////////
  sum_GPU<<<gridSize,blockSize>>>(device_dataIn1, device_dataIn2, device_dataIn3);
  ///////////////////////////////////////////////////////////////////////////////
  gpu_end = clock();

  dtoh_start = clock();
  gpuErrchk( cudaMemcpy(host_dataGPU, device_dataIn1, DATA_BYTES, cudaMemcpyDeviceToHost));
  dtoh_end = clock();

  printf("Sum array CPU execution time : %4.6f \n",
		  (double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));

  printf("Sum array GPU execution time : %4.6f \n",
                  (double)((double)(gpu_end - gpu_start) / CLOCKS_PER_SEC));

  printf("htod mem transfer time : %4.6f \n",
                  (double)((double)(htod_end - htod_start) / CLOCKS_PER_SEC));

  printf("dtoh mem transfer time : %4.6f \n",
                  (double)((double)(dtoh_end - dtoh_start) / CLOCKS_PER_SEC));

  printf("Sum array GPU total execution time : %4.6f \n",
                  (double)((double)(dtoh_end - htod_start) / CLOCKS_PER_SEC));


  cudaFree( device_dataIn1 );
  cudaFree( device_dataIn2 );
  cudaFree( device_dataIn3 );

  delete[] host_dataIn1;
  delete[] host_dataIn2;
  delete[] host_dataIn3;
  delete[] host_dataGPU;
  delete[] host_dataCPU;

  return 0;
}
