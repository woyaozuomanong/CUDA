#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

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
	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ((double)tp.tv_sec+(double)tp.tv_usec*1e-6);
}


void checkResult(float *hostRef,float *gpuRef,const int N)
{
	double epsilon=1.0e-8;
	bool match=1;
	for(int i=0;i<N;i++)
	{
		if(abs(hostRef[i]-gpuRef[i])>epsilon)
		{
			match=0;
			printf("Arrays do not match!\n");
			printf("host %5.2f gpu %5.2f at current %d\n",hostRef[i],gpuRef[i],i);
			break;
		}
	}
	if(match) printf("Arrays match.\n\n");
}

void initialData(float *ip,int size)
{
	//generate different seed for random number
	time_t t;
	srand((unsigned) time(&t));

	for(int i=0;i<size;i++)
	{
		ip[i]=(float)(rand() & 0xFF)/10.0f;
	}
}

void sumArraysOnHost(float *A,float *B,float *C,const int N,int offset)
{
	for(int idx=offset,k=0;idx<N;idx++,k++)
		C[idx]=A[k]+B[k];
}

__global__ void warmup(float *A,float *B,float *C,const int N,int offset)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int k=i+offset;
	if(k<N)C[i]=A[k]+B[k];
}

__global__ void writeSegmentoffset(float *A,float *B,float *C,const int N,int offset)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int k=i+offset;
	if(k<N)C[i]=A[k]+B[k];
}

int main(int argc,char** argv)
{
	printf("%s Starting reduction at ",argv[0]);

	//set up device
	int dev=0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp,dev));
	printf("device %d:%s ",dev,deviceProp.name);
	CHECK(cudaSetDevice(dev));

	//set up data size of vectors
	int nElem=1<<24;
	printf("with array size %d\n",nElem);
	size_t nBytes=nElem*sizeof(float);

	//set up offset for summary
	int blocksize=512;
	int offset=0;
	if(argc>1) offset  =atoi(argv[1]);
	if(argc>2) blocksize  =atoi(argv[2]);

	//execution configuration
	dim3 block(blocksize,1);
	dim3 grid((nElem+block.x-1)/block.x,1);


	//malloc host memory;
	float *h_A;float *h_B; float *hostRef;float *gpuRef;
	h_A      =(float*)malloc(nBytes);
	h_B      =(float*)malloc(nBytes);
	hostRef  =(float*)malloc(nBytes);
	gpuRef   =(float*)malloc(nBytes);

	double iStart,iElaps;

	//initialize data at host side
	iStart=cpuSecond();
	initialData(h_A,nElem);
	initialData(h_B,nElem);
	iElaps=cpuSecond()-iStart;
	printf("Initialize data at host side, time elapsed:%f sec\n",iElaps);

	memset(hostRef,0,nBytes);
	memset(gpuRef, 0,nBytes);


	//malloc device global memory;
	float *d_A,*d_B,*d_C;
	cudaMalloc((float**)&d_A,nBytes);
	cudaMalloc((float**)&d_B,nBytes);
	cudaMalloc((float**)&d_C,nBytes);

	//transfer data from host to device;
	cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,h_B,nBytes,cudaMemcpyHostToDevice);

        //warmup
	iStart=cpuSecond();
        warmup<<< grid,block >>>(d_A,d_B,d_C,nElem,offset);
	cudaDeviceSynchronize();
	iElaps=cpuSecond()-iStart;
	printf("warmup<<< %4d,%4d >>> offset %4d elapsed %f sec\n",grid.x,block.x,offset,iElaps);

        //kernel 1
	iStart=cpuSecond();
        writeSegmentoffset<<< grid,block >>>(d_A,d_B,d_C,nElem,offset);
	cudaDeviceSynchronize();
	iElaps=cpuSecond()-iStart;
	printf("writeOffset <<< %4d,%4d >>> offset %4d elapsed %f sec\n",grid.x,block.x,offset,iElaps);

	//copy kernel result back to host side
	cudaMemcpy(gpuRef,d_C,nBytes,cudaMemcpyDeviceToHost);

	// summary at host side
	iStart=cpuSecond();
	sumArraysOnHost(h_A,h_B,hostRef,nElem,offset);
	iElaps=cpuSecond()-iStart;
	printf("Execution on Host Time elapsed %f sec\n",iElaps);
	//check device results;
	checkResult(hostRef,gpuRef,nElem);
	
	//free device global memory;
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	//free host memory
	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);
        
	//reset device
	cudaDeviceReset();
	return EXIT_SUCCESS;
}
