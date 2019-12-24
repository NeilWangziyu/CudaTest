#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <Windows.h>




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

void SumArrayOnHost(float *A, float *B, float *C, const int N)
{
	for (int i = 0; i < N; i++)
	{
		C[i] = A[i] + B[i];
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

void initialData(float *ip, int size)
{
	time_t t;
	srand((unsigned int)time(&t));
	for (int i = 0; i < size; i++)
	{
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}
}

__global__ void SumArrayOnGpu(float *A, float *B, float *C)
{
	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}

int main(int argc, char **argv)
{
	printf("Starting...");

	SYSTEMTIME st;
	GetSystemTime(&st);
	// then convert st to your precision needs

	printf("starting time is : %d\n", st);
	int dev = 0;
	cudaSetDevice(dev);


	int nElem = 1<<24;
	//printf("Vector size:32");

	size_t nByte = nElem * sizeof(float);

	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (float *)malloc(nByte);
	h_B = (float *)malloc(nByte);
	gpuRef = (float *)malloc(nByte);
	hostRef = (float *)malloc(nByte);

	initialData(h_A, nElem);
	initialData(h_B, nElem);

	memset(hostRef, 0, nByte);
	memset(gpuRef, 0, nByte);



	//malloc
	float *d_A, *d_B, *d_C;
	cudaMalloc((float**)&d_A, nByte);
	cudaMalloc((float**)&d_B, nByte);
	cudaMalloc((float**)&d_C, nByte);

	//transfer
	cudaMemcpy(d_A, h_A, nByte, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nByte, cudaMemcpyHostToDevice);


	//inovke kernal at host side
	int iLen = 1024;
	dim3 block(iLen);
	dim3 grid ((nElem + block.x - 1)/ block.x);

	SYSTEMTIME st_gpu, end_gpu;
	int total_gpu;
	GetSystemTime(&st_gpu);
	SumArrayOnGpu <<<grid, block >>> (d_A, d_B, d_C);
	cudaDeviceSynchronize();
	GetSystemTime(&end_gpu);
	total_gpu = end_gpu.wMilliseconds - st_gpu.wMilliseconds;
	printf("the start time use of GPU cal is %d,\nthe end time of GPU is %d,\n the total use time is %d\n", st_gpu.wMilliseconds, end_gpu.wMilliseconds, total_gpu);


	printf("EXECUTATE CONFIGUREATION <<<%d, %d>>>\n", grid.x, block.x);

	//copy kernel result back to host side
	cudaMemcpy(gpuRef, d_C, nByte, cudaMemcpyDeviceToHost);

	//add 

	SYSTEMTIME st_host, end_host;
	int total_host;
	GetSystemTime(&st_host);
	SumArrayOnHost(h_A, h_B, hostRef, nElem);
	GetSystemTime(&end_host);
	total_host = end_host.wMilliseconds - st_host.wMilliseconds;
	printf("the start time use of CPU cal is %d,\nthe end time of CPU is %d,\n the total use time is %d\n", st_host.wMilliseconds, end_host.wMilliseconds,total_host);

	//check
	CheckResult(hostRef, gpuRef, nElem);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_B);



	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);

	return 0;
}