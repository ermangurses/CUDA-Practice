#include <iostream> 
#include <vector>
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>


__global__ void hello_cuda()
{
    printf("Hello CUDA world from thread %d at block %d\n",threadIdx.x,blockIdx.x);
    
}

int main()
{
    hello_cuda<<<4,4>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}

