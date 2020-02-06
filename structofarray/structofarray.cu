#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#define LEN 1<<20

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

struct innerArray
{
float x[LEN];
float y[LEN];
};

typedef struct innerArray iA;

double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ((double)tp.tv_sec+(double)tp.tv_usec*1e-6);
}


void checkInnerArray(iA *hostRef,iA  *gpuRef,const int N)
{
	double epsilon=1.0e-8;
	bool match=1;
	for(int i=0;i<N;i++)
	{
		if(abs(hostRef->x[i]-gpuRef->x[i])>epsilon||abs(hostRef->y[i]-gpuRef->y[i])>epsilon)
		{
			match=0;
			printf("Arrays do not match!\n");
			break;
		}
	}
	if(match) printf("Arrays match.\n\n");
}

void initialInnerArray(iA *ip,int size)
{
	//generate different seed for random number
	time_t t;
	srand((unsigned) time(&t));

	for(int i=0;i<size;i++)
	{
		ip->x[i]=(float)(rand() & 0xFF)/10.0f;
		ip->y[i]=(float)(rand() & 0xFF)/10.0f;
	}
}


void testInnerArrayHost(iA *A,iA *B,const int N)
{
	for(int idx=0;idx<N;idx++)
	{
	B->x[idx]=A->x[idx]+10.f;
	B->y[idx]=A->y[idx]+20.f;
	}
}

__global__ void  warmup(iA *A,iA *B,const int N)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<N)
	{
	B->x[i]=A->x[i]+10.f;
	B->y[i]=A->y[i]+20.f;
	}
}

__global__ void  testInnerArray(iA *A,iA *B,const int N)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<N)
	{
	B->x[i]=A->x[i]+10.f;
	B->y[i]=A->y[i]+20.f;
	}
}

int main(int argc,char** argv)
{
	printf("%s Starting...\n",argv[0]);

	//set up device
	int dev=0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp,dev));
	printf("%s test struct of array at ",argv[0]);
	printf("device %d:%s\n",dev,deviceProp.name);
	CHECK(cudaSetDevice(dev));

	//set up data size of vectors
	int nElem=LEN;
	size_t nBytes=sizeof(iA);

	iA *h_A; iA *hostRef;iA *gpuRef;
	h_A      =(iA*)malloc(nBytes);
	hostRef  =(iA*)malloc(nBytes);
	gpuRef   =(iA*)malloc(nBytes);

	double iStart,iElaps;

	//initialize data at host side
	iStart=cpuSecond();
	initialInnerArray(h_A,nElem);
	iElaps=cpuSecond()-iStart;
	printf("Initialize data at host side, time elapsed:%f sec\n",iElaps);

	memset(hostRef,0,nBytes);
	memset(gpuRef, 0,nBytes);


	//malloc device global memory;
	iA *d_A,*d_B;
	cudaMalloc((iA**)&d_A,nBytes);
	cudaMalloc((iA**)&d_B,nBytes);

	//transfer data from host to device;
	cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice);

	//setup offset for summary
	int blocksize=128;
	if(argc>1) blocksize=atoi(argv[1]);

	dim3 block(blocksize,1);
	dim3 grid ((nElem+block.x-1)/block.x,1);

	//kernel 1:warmup
	iStart=cpuSecond();
	warmup<<< grid,block >>>(d_A,d_B,nElem);
	cudaDeviceSynchronize();
	iElaps=cpuSecond()-iStart;
	printf("Warmup execution configuration <<<%d,%d>>> Time elapsed %f sec\n",grid.x,block.x,iElaps);

	//kernel 2:
	iStart=cpuSecond();
        testInnerArray<<< grid,block >>>(d_A,d_B,nElem);
	cudaDeviceSynchronize();
	iElaps=cpuSecond()-iStart;
	printf("TestInnerStructHost Execution configuration <<<%d,%d>>> Time elapsed %f sec\n",grid.x,block.x,iElaps);

	//copy kernel result back to host side
	cudaMemcpy(gpuRef,d_B,nBytes,cudaMemcpyDeviceToHost);

	//test inner struct of array on host
	iStart=cpuSecond();
	testInnerArrayHost(h_A,hostRef,nElem);
	iElaps=cpuSecond()-iStart;
	printf("Execution on Host Time elapsed %f sec\n",iElaps);
	//check device results;
	checkInnerArray(hostRef,gpuRef,nElem);
	
	//free device global memory;
	cudaFree(d_A);
	cudaFree(d_B);

	//free host memory
	free(h_A);
	free(hostRef);
	free(gpuRef);
	return(0);
}
