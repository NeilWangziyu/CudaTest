
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <windows.h>


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

#define CHECK(call)\
{				\
	const cidaError_t error = call;	\
	if (error != cudaSuccess)\
	{\
		printf("Erroe: %s: %d,  ", __FILE__, __LINE__);\
		printf("code: %d, reason: %s\n, ", error, cudaGetErrorString);\
		exit(1);\
	}\
}\


__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
	float *ia = A;
	float *ib = B;
	float *ic = C;
	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			ic[ix] = ia[ix] + ib[ix];
		}
		ia += nx;
		ib += nx;
		ic += nx;
	}
	return;
}

__global__ void SumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * nx + ix;
	if (ix < nx && iy < ny)
	{
		MatC[idx] = MatA[idx] + MatB[idx];

	}
	
}
void initialInx(int *ip, int size)
{
	for (int i = 0; i < size; i++)
	{
		ip[i] = i;
	}
}

void printmatrix(int *C, const int nx, const int ny)
{
	int *ic = C;
	printf("Matrix: (%d.%d)\n", nx, ny);
	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			printf("%3d", ic[ix]);
		}
		ic += nx;
		printf("\n");
	}
}

void printmatrix(float *C, const int nx, const int ny)
{
	float *ic = C;
	printf("Matrix: (%d.%d)\n", nx, ny);
	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			printf("%3f\t", ic[ix]);
		}
		ic += nx;
		printf("\n");
	}
}

void initialData(float *ip, int size)
{
	time_t t;
	srand((unsigned int)time(&t));
	for (int i = 0; i < size; i++)
	{
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}
}

void CheckResult(float *hostref, float *gpuRef, const int N)
{
	double epsilon = 1.0E-8;
	bool match = 1;
	for (int i = 0; i < N; i++)
	{
		if (abs(hostref[i] - gpuRef[i]) > epsilon)
		{
			match = 0;
			printf("Array do not match!\n");
			printf("host %5.2f gpu %5.2f at current %d\n", hostref[i], gpuRef[i], i);
			break;
		}
	}
	if (match) printf("Array match.\n\n");
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	printf("Finished Default \n ----------------------- \n");

	int dev = 0;
	cudaDeviceProp deviceProb;
	//CHECK(cudaGetDeviceProperties(&deviceProb, dev));
	//printf("Using Device: %d: %s \n", dev, deviceProb.name);
	printf("Using Device: %d\n", dev);

	int nx = 1<<12;
	int ny = 1<<12;
	int nxy = nx * ny;
	int nBytes = nxy * sizeof(float);
	printf("matrix size: nx: %d, ny: %d\n", nx, ny);


	//malloc
	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuRef = (float *)malloc(nBytes);
	

	//init
	initialData(h_A, nxy);
	initialData(h_B, nxy);

	//printmatrix(h_A, nx, ny);
	//printmatrix(h_B, nx, ny);


	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	DWORD time_kernel = GetTickCount(); //获取毫秒级数目


	sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
	
	std::cout << "Totally host function cost:" << GetTickCount() - time_kernel << " mSecond" << std::endl;


	float *d_MatA, *d_MatB, *d_MatC;
	cudaMalloc((void**)&d_MatA, nBytes);
	cudaMalloc((void**)&d_MatB, nBytes);
	cudaMalloc((void**)&d_MatC, nBytes);


	cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

	int dimx = 32;
	int dimy = 32;
	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);




	DWORD time_kernel_gpu = GetTickCount(); //获取毫秒级数目

	SumMatrixOnGPU2D <<<grid, block >>> (d_MatA, d_MatB, d_MatC, nx, ny);

	std::cout << "Totally GPU function cost:" << GetTickCount() - time_kernel_gpu << "mSecond" << std::endl;



	cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

	printf("host result:\n");
	//printmatrix(hostRef, nx, ny);

	printf("gpu result:\n");

	//printmatrix(gpuRef, nx, ny);

	CheckResult(hostRef, gpuRef, nxy);


	cudaFree(d_MatA);
	cudaFree(d_MatB);
	cudaFree(d_MatC);



	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
