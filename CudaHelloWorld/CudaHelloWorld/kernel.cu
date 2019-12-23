#include <stdio.h>

__global__ void helloCuda(void)
{
	printf("hello from GPU\n");
}

int main(void)
{
	printf("hello from CPU\n");

	helloCuda <<< 1, 10 >>> ();
	cudaDeviceReset();
	return 0;
}