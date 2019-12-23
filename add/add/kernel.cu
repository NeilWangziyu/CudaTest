// https://devblogs.nvidia.com/even-easier-introduction-cuda/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <math.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__
void add(int n, float * x, float *y)
{
	for (int i = 0; i < n; i++)
	{
		y[i] = x[i] + y[i];
	}
}

__global__
void add2(int n, float *x, float *y)
{
	int index = threadIdx.x;
	int stride = blockDim.x;
	for (int i = 0; i < n; i+=stride)
	{
		y[i] = x[i] + y[i];
	}
}

__global__
void add3(int n, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i+=stride)
	{
		y[i] = x[i] + y[i];
	}
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

	printf("below is our codes:\n");

	int N = 1 << 20;
	float *x, *y;

	cudaMallocManaged(&x, N * sizeof(float));
	cudaMallocManaged(&y, N * sizeof(float));

	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	add <<<1, 1 >>> (N, x, y);

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	/*for (auto i : *y)
	{
		std::cout << i << " ";
	}
	std::cout << std::endl;*/

	std::cout <<"Result is" <<  *y << std::endl;

	// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
		maxError = fmax(maxError, fabs(y[i] - 3.0f));
	std::cout << "Max error: " << maxError << std::endl;

;
	// Free memory
	cudaFree(x);
	cudaFree(y);

	//---------------------------

	int N2 = 1 << 20;
	float *x2, *y2;

	cudaMallocManaged(&x2, N2 * sizeof(float));
	cudaMallocManaged(&y2, N2 * sizeof(float));

	// initialize x and y arrays on the host
	for (int i = 0; i < N2; i++) {
		x2[i] = 1.0f;
		y2[i] = 2.0f;
	}

	add2 <<<1, 256 >>> (N2, x2, y2);
	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	std::cout << "Result2 is " << *y2 << std::endl;
	// Free memory
	cudaFree(x2);
	cudaFree(y2);

	//---------------------------

	int N3 = 1 << 20;
	float *x3, *y3;

	cudaMallocManaged(&x3, N3 * sizeof(float));
	cudaMallocManaged(&y3, N3 * sizeof(float));

	// initialize x and y arrays on the host
	for (int i = 0; i < N3; i++) {
		x3[i] = 4.0f;
		y3[i] = 2.0f;
	}

	add2 <<<1, 256 >>> (N3, x3, y3);
	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	std::cout << "Result3 is " << *y3 << std::endl;
	// Free memory
	cudaFree(x3);
	cudaFree(y3);


	int N4 = 1 << 20;
	float *x4, *y4;


	int blocksize = 256;
	int numofblock = (N4 + blocksize - 1) / blocksize;

	cudaMallocManaged(&x4, N4 * sizeof(float));
	cudaMallocManaged(&y4, N4 * sizeof(float));

	// initialize x and y arrays on the host
	for (int i = 0; i < N4; i++) {
		x4[i] = 4.0f;
		y4[i] = 9.0f;
	}

	add3 <<<numofblock, blocksize >>> (N4, x4, y4);
	cudaDeviceSynchronize();
	std::cout << "Result4 is " << *y4 << std::endl;
	// Free memory
	cudaFree(x4);
	cudaFree(y4);


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
