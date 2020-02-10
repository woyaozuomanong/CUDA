#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#define CHECK(call)       \
{\
	const cudaError_t error=call;\
	if(error!=cudaSuccess)\
	{\
		printf("error:%s:%d, ",__FILE__,__LINE__);\
		printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
		exit(1);\
	}\
}\

double cpuSecond()
{
	_LARGE_INTEGER time_start;    /*开始时间*/
	double dqFreq;                /*计时器频率*/
	LARGE_INTEGER f;            /*计时器频率*/
	QueryPerformanceFrequency(&f);
	dqFreq = (double)f.QuadPart;
	QueryPerformanceCounter(&time_start);
	return time_start.QuadPart / dqFreq ;//单位为秒，精度为1000 000/（cpu主频）微秒
}


void checkResult(float *hostRef, float *gpuRef, const int N)
{
	double epsilon = 1.0e-8;
	bool match = 1;
	for (int i = 0; i < N; i++)
	{
		if (abs(hostRef[i] - gpuRef[i]) > epsilon)
		{
			match = 0;
			printf("Arrays do not match!\n");
			printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
			break;
		}
	}
	if (match) printf("Arrays match.\n\n");
}

void initialData(float *ip, int size)
{
	//generate different seed for random number
	time_t t;
	srand((unsigned)time(&t));

	for (int i = 0; i < size; i++)
	{
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}
}

void transposeOnHost(float *out, float *in, const int nx, const int ny)
{
	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
			out[ix*ny + iy] = in[iy*nx + ix];
	}
}

__global__ void warmup(float *out, float *in, const int nx, const int ny)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny)
		out[iy*nx + ix] = in[iy*nx + ix];
}

__global__ void copybyRow(float *out, float *in, const int nx, const int ny)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny)
		out[iy*nx + ix] = in[iy*nx + ix];
}

__global__ void copybyCol(float *out, float *in, const int nx, const int ny)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny)
		out[ix*ny + iy] = in[ix*ny + iy];
}

__global__ void transposeNaiveRow(float *out, float *in, const int nx, const int ny)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny)
		out[ix*ny + iy] = in[iy*nx + ix];
}

__global__ void transposeNaiveCol(float *out, float *in, const int nx, const int ny)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix < nx && iy < ny)
		out[iy*nx + ix] = in[ix*ny + iy];
}

__global__ void transposeUnroll4Row(float *out, float *in, const int nx, const int ny)
{
	int ix = blockIdx.x*blockDim.x * 4 + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;

	int ti = iy * nx + ix;
	int to = ix * ny + iy;

	if (ix + 3 * blockDim.x < nx && iy < ny)
		out[to] = in[ti];
	out[to + ny * blockDim.x] = in[ti + blockDim.x];
	out[to + ny * 2 * blockDim.x] = in[ti + 2 * blockDim.x];
	out[to + ny * 3 * blockDim.x] = in[ti + 3 * blockDim.x];
}

__global__ void transposeUnroll4Col(float *out, float *in, const int nx, const int ny)
{
	int ix = blockIdx.x*blockDim.x * 4 + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;

	int ti = iy * nx + ix;
	int to = ix * ny + iy;

	if (ix + 3 * blockDim.x < nx && iy < ny)
		out[ti] = in[to];
	out[ti + blockDim.x] = in[to + ny * blockDim.x];
	out[ti + 2 * blockDim.x] = in[to + ny * 2 * blockDim.x];
	out[ti + 3 * blockDim.x] = in[to + ny * 3 * blockDim.x];
}



int main(int argc, char** argv)
{
	printf("%s Starting transpose at ", argv[0]);

	//set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("device %d:%s", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	//set up array size 
	int nx = 1 << 11;
	int ny = 1 << 11;

	//select a kernel and block size
	int iKernel = 0;
	int blockx = 16;
	int blocky = 16;
	if (argc > 1) iKernel = atoi(argv[1]);
	if (argc > 2) blockx = atoi(argv[2]);
	if (argc > 3) blocky = atoi(argv[3]);
	if (argc > 4) nx = atoi(argv[4]);
	if (argc > 5) ny = atoi(argv[5]);
	printf(" with array size nx %d * ny %d with kernel %d\n", nx, ny, iKernel);

	//execution configuration
	dim3 block(blockx, blocky);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	//malloc host memory;
	size_t nBytes = nx * ny * sizeof(float);

	float *h_A; float *hostRef; float *gpuRef;
	h_A = (float*)malloc(nBytes);
	hostRef = (float*)malloc(nBytes);
	gpuRef = (float*)malloc(nBytes);

	double iStart, iElaps;

	//initialize data at host side
	iStart = cpuSecond();
	initialData(h_A, nx*ny);
	iElaps = cpuSecond() - iStart;
	printf("Initialize data at host side, time elapsed:%f sec\n", iElaps);

	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	//transpose at host side
	iStart = cpuSecond();
	transposeOnHost(hostRef, h_A, nx, ny);
	iElaps = cpuSecond() - iStart;
	printf("Transpose array at host side, time elapsed:%f sec\n", iElaps);



	//malloc device global memory;
	float *d_A, *d_C;
	cudaMalloc((float**)&d_A, nBytes);
	cudaMalloc((float**)&d_C, nBytes);

	//transfer data from host to device;
	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

	//warmup to avoid startup overhead
	iStart = cpuSecond();
	warmup << <grid, block >> > (d_C, d_A, nx, ny);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	printf("warmup time elapsed:%f sec\n", iElaps);

	//kernel pointer and descriptor
	void(*kernel) (float*, float*, int, int);
	char* kernelname;

	//set up kernel
	switch (iKernel)
	{
	case 0:
		kernel = &copybyRow;
		kernelname = "CopyRow     ";
		break;
	case 1:
		kernel = &copybyCol;
		kernelname = "CopyCol     ";
		break;
	case 2:
		kernel = &transposeNaiveRow;
		kernelname = "TransposeNaiveRow";
		break;
	case 3:
		kernel = &transposeNaiveCol;
		kernelname = "transposeNaiveCol";
		break;
	case 4:
		kernel = &transposeUnroll4Row;
		kernelname = "Unroll4Row";
		grid.x = (nx + block.x * 4 - 1) / (block.x * 4);
		break;
	case 5:
		kernel = &transposeUnroll4Col;
		kernelname = "Unroll4Col";
		grid.x = (nx + block.x * 4 - 1) / (block.x * 4);
		break;
	default:
		break;
	}

	//run kernel
	iStart = cpuSecond();
	kernel << < grid, block >> > (d_C, d_A, nx, ny);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	printf("Execution configuration <<<%d,%d>>> Time elapsed %f sec\n", grid.x, block.x, iElaps);

	//calculate effective bandwidth
	float ibnd = 2 * nBytes / 1e9 / iElaps;
	printf("%s elapsed %f sec <<<grid (%d,%d) block (%d,%d)>>> effective bandwidth %f GB\n", kernelname, iElaps, grid.x, grid.y, block.x, block.y, ibnd);


	//check device results;
	if (iKernel > 1)
	{
		cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
		checkResult(hostRef, gpuRef, nx*ny);
	}
	//free device global memory;
	cudaFree(d_A);
	cudaFree(d_C);

	//free host memory
	free(h_A);
	free(hostRef);
	free(gpuRef);

	//reset device
	cudaDeviceReset();
	return EXIT_SUCCESS;
}