#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

inline double seconds()
{
        struct timeval tp;
	struct timezone tzp;
        int i = gettimeofday(&tp, &tzp);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
	//return ((double)tp.tv_sec*1e6 + (double)tp.tv_usec );
}


__global__ void mathkernel1(float *c)
{
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	float a,b;
	a=b=0.0f;

	if(tid%2==0)
	{
		a=100.0;
	}
	else
	{
		b=200.0;
	}
	c[tid]=a+b;
//        printf("c[%d]= %f\n",tid,c[tid]);
}

__global__ void mathkernel2(float *c)
{
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	float a,b;
	a=b=0.0f;

	if((tid/warpSize)%2==0)
	{
		a=100.0;
	}
	else
	{
		b=200.0;
	}
	c[tid]=a+b;
        //printf("c[%d]= %f\n",tid,c[tid]);
}
__global__ void warmingup(float *c)
{
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	float a,b;
	a=b=0.0f;

	if((tid/warpSize)%2==0)
	{
		a=100.0;
	}
	else
	{
		b=200.0;
	}
	c[tid]=a+b;
}




int main(int argc, char **argv)
{
	//set up device
	int dev=0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp,dev);
	printf("%s using Device %d: %s\n",argv[0],dev,deviceProp.name);

	//set up data size
	int size=64;
	int blocksize=64;
	if(argc>1)blocksize=atoi(argv[1]);
	if(argc>2)size=atoi(argv[2]);
	printf("Data size %d\n ",size);

	//set up execution configuration
	dim3 block(blocksize,1);
	dim3 grid((size+block.x-1)/block.x,1);
	printf("Execution Configure (block %d grid %d)\n",block.x,grid.x);

	//allocate gpu memory
	float *d_C;
        size_t nBytes=size*sizeof(float);
	cudaMalloc((float **)&d_C,nBytes);

	//run a warmup kernel to remove overhead
	double iStart,iElaps;
	cudaDeviceSynchronize();
	iStart=seconds();
	warmingup<<<grid,block>>> (d_C);
	cudaDeviceSynchronize();
	iElaps=seconds()-iStart;
	printf("warmup    <<<%4d %4d>>> elapsed %f sec \n",grid.x,block.x,iElaps);
	
	//run kernel 1
	iStart=seconds();
	mathkernel1<<<grid,block>>> (d_C);
	cudaDeviceSynchronize();
	iElaps=seconds()-iStart;
	printf("mathkernel1<<<%4d %4d>>> elapsed %f sec \n",grid.x,block.x,iElaps);
	//run kernel 2
	iStart=seconds();
	mathkernel2<<<grid,block>>> (d_C);
	cudaDeviceSynchronize();
	iElaps=seconds()-iStart;
	printf("mathkernel2<<<%4d %4d>>> elapsed %f sec \n",grid.x,block.x,iElaps);
	//run kernel 3
	iStart=seconds();
	//mathkernel3<<<grid,block>>> (d_C);
	cudaDeviceSynchronize();
	iElaps=seconds()-iStart;
	printf("mathkernel3<<<%4d %4d>>> elapsed %f sec \n",grid.x,block.x,iElaps);
	//run kernel 4
	iStart=seconds();
	//mathkernel4<<<grid,block>>> (d_C);
	cudaDeviceSynchronize();
	iElaps=seconds()-iStart;
	printf("mathkernel4<<<%4d %4d>>> elapsed %f sec \n",grid.x,block.x,iElaps);

	//free gpu memory and reset device
	cudaFree(d_C);
	cudaDeviceReset();
	return EXIT_SUCCESS;
}
