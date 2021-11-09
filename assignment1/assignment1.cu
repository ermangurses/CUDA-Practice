#include <iostream> 
#include <vector>
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>


__global__ void print_cuda_info()
{
    printf("Hello CUDA world from thread x: %d  y: %d  z: %d at block  x: %d  y: %d  z: %d and grid dim  x: %d,  y: %d,  z: %d\n",
		    threadIdx.x,threadIdx.y,threadIdx.z,blockIdx.x,blockIdx.y,blockIdx.z,gridDim.x, gridDim.y, gridDim.z);
}

int main()
{
    dim3 block(2,2,2);
    dim3 grid(2,2,2);

    print_cuda_info<<<grid,block>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}

