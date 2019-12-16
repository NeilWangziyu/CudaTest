
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <conio.h>
#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

typedef unsigned int u32;
#define NUM_ELEM 32

__host__ void cpu_sort(u32 * const data, const u32 num_elements)
{
	static u32 cpu_tmp_0[NUM_ELEM];
	static u32 cpu_tmp_1[NUM_ELEM];

	for (u32 bit = 0; bit < 32; bit++)
	{
		u32 base_cnt_0 = 0;
		u32 base_cnt_1 = 0;

		for (u32 i = 0; i < num_elements; i++)
		{
			const u32 d = data[i];
			const u32 bit_mask = (1 << bit);

			if ((d & bit_mask) > 0)
			{
				cpu_tmp_1[base_cnt_1] = d;
				base_cnt_1++;
			}
			else
			{
				cpu_tmp_0[base_cnt_0] = d;
				base_cnt_0++;
			}

		}

		//copy data back to source - first the zero first
		for (u32 i = 0; i < base_cnt_0; i++)
		{
			data[i] = cpu_tmp_0[i];
		}

		//copy data back to source - then the one list
		for (u32 i = 0; i < base_cnt_1; i++)
		{
			data[base_cnt_0 + i] = cpu_tmp_1[i];
		}

	}

}

__device__ void radix_sort(u32 * const sort_tmp,
	const u32 num_lists,
	const u32 num_elements,
	const u32 tid,
	u32 * const sort_tmp_0,
	u32 * const sort_tmp_1)
{
	//sort into num_list, lists
	// Apply radix sort on 32 bits of data
	for (u32 bit = 0; bit < 32; bit++)
	{
		u32 base_cnt_0 = 0;
		u32 base_cnt_1 = 0;

		for (u32 i = 0; i < num_elements; i+=num_lists)
		{
			const u32 elem = sort_tmp[i + tid];
			const u32 bit_mask = (1 << bit);

			if ((elem & bit_mask) > 0)
			{
				sort_tmp_1[base_cnt_1 + tid] = elem;
				base_cnt_1 += num_lists;
			}
			else
			{
				sort_tmp_0[base_cnt_0 + tid] = elem;
				base_cnt_0 += num_lists;
			}
		}

		//copy data back to source - first the zero list
		for (u32 i = 0; i < base_cnt_0; i+=num_lists)
		{
			sort_tmp[i + tid] = sort_tmp_0[i + tid];
		}

		for (u32 i = 0; i < base_cnt_1; i += num_lists)
		{
			sort_tmp[base_cnt_0 + i + tid] = sort_tmp_1[i + tid];
		}

	}

	__syncthreads();
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

	printf("CPU version of bin sort.\n");

	u32 input[arraySize] = { 11, 2, 9, 4, 5 };

	printf("Before sort: {%d,%d,%d,%d,%d}\n",
		input[0], input[1], input[2], input[3], input[4]);

	cpu_sort(input, arraySize);

	printf("After sort: {%d,%d,%d,%d,%d}\n",
		input[0], input[1], input[2], input[3], input[4]);


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
