#include <iostream> 
#include <vector>
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>


__global__ void hello_cuda()
{
    printf("Hello CUDA world from thread %d%d%d at block %d%d%d\n",threadIdx.x,threadIdx.y,threadIdx.z,blockIdx.x,blockIdx.y,blockIdx.z);
}

int main()
{
    dim3 block(2,3,4);
    dim3 grid(4,3,2);

    hello_cuda<<<grid,block>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}

