#include <iostream> 
#include <vector>
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, 
                                                              bool abort=true){
   if (code != cudaSuccess){
      fprintf(stderr,"GPUassert: %s %s %d\n", 
                                         cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void filter(float * device_dataIn)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;

    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                   + (threadIdx.z * (blockDim.x * blockDim.y))
                   + (threadIdx.y * blockDim.x) + threadIdx.x;

    printf("threadId %d has %f as data \n",threadId,device_dataIn[threadId]);

}

int main(int argc, char* argv[]) {

  // Dimensions of the 3D data
  int DIM_X = 4;
  int DIM_Y = 4;
  int DIM_Z = 4;

  // Definition of host and device pointers
  float * host_dataIn;
  float * device_dataIn;

  // Memory allocation for host data
  host_dataIn = new float [DIM_X * DIM_Y * DIM_Z];
  
  // Calculation of required spaces 
  const int DATA_BYTES      = (DIM_X * DIM_Y * DIM_Z) * sizeof(float);

  int i = 0;
  // Initialize host_dataIn with random numbers
  for( int x = 0; x < DIM_X; ++x )
  {
      for(int y = 0; y < DIM_Y; ++y )
      {
          for(int z = 0; z < DIM_Z; ++z )
	  { 
              host_dataIn [z + y * DIM_Z + x * DIM_Z * DIM_Y] = i++; 
                                          // (rand() / (float)RAND_MAX * 100) + 1;
          }
      }
  }

  // Allocate DEVICE arrays on GPU for dataIn
  gpuErrchk( cudaMalloc( ( void**) &device_dataIn,  DATA_BYTES));
  gpuErrchk( cudaMemcpy( device_dataIn, host_dataIn, DATA_BYTES, cudaMemcpyHostToDevice));
  int blockX = 2;
  int blockY = 2;
  int blockZ = 2;
  dim3 blockSize(blockX,blockY,blockZ);

  int gridX = (DIM_X) / blockX;
  int gridY = (DIM_Y) / blockY; 
  int gridZ = (DIM_Z) / blockZ;
  dim3 gridSize(gridX,gridY,gridZ);


///////////////////////////////////////////////////////////////////////////////
// Kernel Call
///////////////////////////////////////////////////////////////////////////////
  filter<<<gridSize,blockSize>>>(device_dataIn);
///////////////////////////////////////////////////////////////////////////////



  cudaFree( device_dataIn );
  delete[] host_dataIn;
  return 0;
}
