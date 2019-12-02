#include <stdio.h>

__global__ void helloFromGPU(void)
{
	printf("Hello World from GPU!\n");
	printf("threadIdx:(%d %d %d)  blockIdx:(%d %d %d)\n",threadIdx.x,threadIdx.y,threadIdx.z,blockIdx.x,blockIdx.y,blockIdx.z);
}

int main(void)
{
	helloFromGPU<<<2,10>>>();
	//cudaDeviceReset();
	cudaDeviceSynchronize();
	return 0;
}
