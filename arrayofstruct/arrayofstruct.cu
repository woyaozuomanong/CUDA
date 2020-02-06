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

struct innerStruct
{
float x;
float y;
};

typedef struct innerStruct iS;

double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ((double)tp.tv_sec+(double)tp.tv_usec*1e-6);
}


void checkInnerStruct(iS *hostRef,iS  *gpuRef,const int N)
{
	double epsilon=1.0e-8;
	bool match=1;
	for(int i=0;i<N;i++)
	{
		if(abs(hostRef[i].x-gpuRef[i].x)>epsilon||abs(hostRef[i].y-gpuRef[i].y)>epsilon)
		{
			match=0;
			printf("Arrays do not match!\n");
			break;
		}
	}
	if(match) printf("Arrays match.\n\n");
}

void initialInnerStruct(iS *ip,int size)
{
	//generate different seed for random number
	time_t t;
	srand((unsigned) time(&t));

	for(int i=0;i<size;i++)
	{
		ip[i].x=(float)(rand() & 0xFF)/10.0f;
		ip[i].y=(float)(rand() & 0xFF)/10.0f;
	}
}


void testInnerStructHost(iS *A,iS *B,const int N)
{
	for(int idx=0;idx<N;idx++)
	{
	B[idx].x=A[idx].x+10.f;
	B[idx].y=A[idx].y+20.f;
	}
}

__global__ void  warmup(iS *A,iS *B,const int N)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<N)
	{
	B[i].x=A[i].x+10.f;
	B[i].y=A[i].y+20.f;
	}
}

__global__ void  testInnerStruct(iS *A,iS *B,const int N)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<N)
	{
	B[i].x=A[i].x+10.f;
	B[i].y=A[i].y+20.f;
	}
}

int main(int argc,char** argv)
{
	printf("%s Starting...\n",argv[0]);

	//set up device
	int dev=0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp,dev));
	printf("%s test array of struct at ",argv[0]);
	printf("device %d:%s\n",dev,deviceProp.name);
	CHECK(cudaSetDevice(dev));

	//set up data size of vectors
	int nElem=LEN;
	size_t nBytes=nElem*sizeof(iS);

	iS *h_A; iS *hostRef;iS *gpuRef;
	h_A      =(iS*)malloc(nBytes);
	hostRef  =(iS*)malloc(nBytes);
	gpuRef   =(iS*)malloc(nBytes);

	double iStart,iElaps;

	//initialize data at host side
	iStart=cpuSecond();
	initialInnerStruct(h_A,nElem);
	iElaps=cpuSecond()-iStart;
	printf("Initialize data at host side, time elapsed:%f sec\n",iElaps);

	memset(hostRef,0,nBytes);
	memset(gpuRef, 0,nBytes);


	//malloc device global memory;
	iS *d_A,*d_B;
	cudaMalloc((iS**)&d_A,nBytes);
	cudaMalloc((iS**)&d_B,nBytes);

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
        testInnerStruct<<< grid,block >>>(d_A,d_B,nElem);
	cudaDeviceSynchronize();
	iElaps=cpuSecond()-iStart;
	printf("TestInnerStructHost Execution configuration <<<%d,%d>>> Time elapsed %f sec\n",grid.x,block.x,iElaps);

	//copy kernel result back to host side
	cudaMemcpy(gpuRef,d_B,nBytes,cudaMemcpyDeviceToHost);

	//test inner struct of array on host
	iStart=cpuSecond();
	testInnerStructHost(h_A,hostRef,nElem);
	iElaps=cpuSecond()-iStart;
	printf("Execution on Host Time elapsed %f sec\n",iElaps);
	//check device results;
	checkInnerStruct(hostRef,gpuRef,nElem);
	
	//free device global memory;
	cudaFree(d_A);
	cudaFree(d_B);

	//free host memory
	free(h_A);
	free(hostRef);
	free(gpuRef);
	return(0);
}
