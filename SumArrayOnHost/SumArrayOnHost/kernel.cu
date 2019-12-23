#include <stdlib.h>
#include <string.h>
#include <time.h>

void SumArrayOnHost(float *A, float *B, float *C, const int N)
{
	for (int i = 0; i < N; i++)
	{
		C[i] = A[i] + B[i];
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

int main(int argc, char **argv)
{
	int nElem = 1024;
	size_t nByte = nElem * sizeof(float);

	float *h_A, *h_B, *h_C;
	h_A = (float *)malloc(nByte);
	h_B = (float *)malloc(nByte);
	h_C = (float *)malloc(nByte);

	initialData(h_A, nElem);
	initialData(h_B, nElem);

	SumArrayOnHost(h_A, h_B, h_C, nElem);

	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}