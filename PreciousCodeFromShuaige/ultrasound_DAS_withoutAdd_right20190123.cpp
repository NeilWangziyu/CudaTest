#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include<cmath>
#include "mclmcr.h"
#include<complex>
#include <cuComplex.h>
#include<stdlib.h>
#include <stdio.h>
#include <iostream>
#include <windows.h>
#include"matrix.h"

//#include<complex.h>

using namespace std;
//cudaError_t theta_calculate_WithCuda(int a, int b);

//定义常量内存
__constant__ double x_begin = (-3.0 / 1000);
__constant__ double z_begin = 27.5 / 1000;
__constant__ double d_x = 0.05 / 1000;
__constant__ double d_z = 0.05 / 1000;
__constant__ double pitch = (3.08e-4);
__constant__ int N_elements = 128;
__constant__ int c = 1540;
__constant__ double theta_threshold = 2.8;
__constant__ int emit_pitch = 1;
__constant__ double t1 = (3.895e-5);
__constant__ int fs = 40000000;
__constant__ int zerolength = 1000;

__global__ void theta_calculate_WithCuda(cuDoubleComplex *d_init_M1, cuDoubleComplex *beamDASDAS, cuDoubleComplex * x_pointer)
{



	int idx = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * blockDim.y + threadIdx.y;
	int b = (idx / 121) + 1;
	int a = (idx - (b - 1) * 121) + 1;
	int i;
	int k;
	int theta_start;
	int theta_end;
	int M_elements;
	int emit_aperture;
	int emit_start;

	double xp, zp;
	double d_t_delay[128];
	double d_theta_ctrb[128];
	double xr[128];

	//printf("a=%d,b=%d\n", a, b);
	//printf("d_init_M1 %d is %e,%e\n", idx, cuCreal(d_init_M1[0]), cuCimag(d_init_M1[0]));

	double a_begin = -double((N_elements - 1)) / 2 * pitch;
	if (idx < 501 * 121)
	{
		xp = x_begin + d_x * (a - 1);
		zp = z_begin + d_z * (b - 1);

		/*cudaMalloc(&d_t_delay, 128 * sizeof(double));
		cudaMalloc(&d_theta_ctrb, 128 * sizeof(double));
		cudaMalloc(&xr, 128 * sizeof(double));*/

		//printf("idx=%d \n", idx);


		for (i = 0; i < (N_elements); i++)
		{
			xr[i] = a_begin + pitch * i;
			d_t_delay[i] = sqrt(pow((xp - xr[i]), 2) + pow(zp, 2)) / c;
			d_theta_ctrb[i] = zp / (abs(xp - xr[i]));

		}
		//printf("Till for 1\n");

		k = 1;
		while ((k <= N_elements) && (d_theta_ctrb[k - 1] < theta_threshold))
		{
			k = k + 1;
		}//求出theta_start的值

		//printf("Till while 1\n");
		if (k == N_elements + 1)
			NULL;
		//开始求theta_end的值
		else
		{
			theta_start = k;
			//	cout << "k or theta_start is" << k << endl;

			while ((k <= N_elements) && (d_theta_ctrb[k - 1] >= theta_threshold))
			{
				k = k + 1;
			}
			if (k == N_elements + 1)
				theta_end = k - 1;
			else
			{
				theta_end = k;
			}

		}
		//printf("theta_start=%d,theta_end=%d\n",theta_start,theta_end);
		/*cudaFree(d_t_delay);
		cudaFree(d_theta_ctrb);*/
		int Trans_elements;
		M_elements = theta_end - theta_start + 1;
		emit_aperture = M_elements;
		emit_start = theta_start;

		//double index_finished = 0;
		int index_1;
		int row_v2;
		int column_v2;
		int temp;
		int temp_1;
		int temp_2;

		//printf("Before 分配x_pointer内存\n");
		//cuDoubleComplex  x_pointer[128 * 128];//为x数组分配动态内存，记得用完delete，x二维数组，是经过正取延时，并且取正确孔径的，回波矩阵。维度为M_elements*M_elements
		double  delay_pointer[128];//为delay数组分配动态内存，记得用完delete
		int delay_int[128];//为取整后的delay值分配内存。
		//printf("After 分配x_pointer内存\n");
						   //*************接下来计算正确延时和回波矩阵x【M_elements*M_elements】*******************************************//
		for (Trans_elements = emit_start; Trans_elements <= (emit_start + emit_pitch * (emit_aperture - 1)); Trans_elements++)
		{
			for (i = theta_start; i <= theta_end; i++)
			{
				temp = i - theta_start;
				delay_pointer[temp] = d_t_delay[Trans_elements - 1] + d_t_delay[i - 1];

				delay_pointer[temp] = delay_pointer[temp] - t1;
				delay_pointer[temp] = ((delay_pointer[temp] * fs) + 25.5) + zerolength;
				delay_int[temp] = (round)(delay_pointer[temp]);
				//	int abc = delay_int[temp];

			}

			//printf("delay_int %d is %d\n", idx,delay_int[0]);//delay_int正确
			for (i = theta_start; i <= theta_end; i++)
			{
				index_1 = (i - theta_start);
				row_v2 = delay_int[index_1];//此处计算有误！
				column_v2 = (Trans_elements - 1)*N_elements + i - 1;

				temp_1 = ((Trans_elements - emit_start) / emit_pitch)*M_elements + i - theta_start;
				//x_pointer[temp_1] = init_M1[row_v2*M + (Trans_elements - 1)*N_elements + i-1];//M为v2矩阵的行数
				temp_2 = column_v2 * 3534 + row_v2 - 1;
				x_pointer[temp_1 + idx * 128 * 128] = d_init_M1[temp_2];//每个线程的x_pointer都分配了128*128的内存，所以下一个线程在全部121*128*128的内存里，其地址要加上idx*128*128

													  //((Trans_elements - 1)*N_elements + i - 1)*M+row_v2
			}


		}

		//printf("x_pointer[0] of thread %d is %e,%e\n", idx, cuCreal(x_pointer[0+ idx * 128 * 128]), cuCimag(x_pointer[0+ idx * 128 * 128]));
		//*************计算延时和回波矩阵完毕*********************************************************************//
		//*************接下来计算本像素点的beamDASDAS*******************************************//
		for (i = 0; i < M_elements; i++)
		{
			int j;
			for (j = 0; j < M_elements; j++)
			{
				cuDoubleComplex temp_1 = make_cuDoubleComplex(0, 0);
				temp_1 = cuCadd(x_pointer[i*M_elements + j + (idx * 128 * 128)], beamDASDAS[(b - 1) * 121 + (a - 1)]);//注意此处x_pointer的地址也加上了idx*128*128
				beamDASDAS[(b - 1) * 121 + (a - 1)] = temp_1;
			}
		}
		beamDASDAS[(b - 1) * 121 + (a - 1)] = make_cuDoubleComplex(cuCreal(beamDASDAS[(b - 1) * 121 + (a - 1)]) / ((double)(M_elements*M_elements)), cuCimag(beamDASDAS[(b - 1) * 121 + (a - 1)]) / ((double)(M_elements*M_elements)));
		//printf("beam %d is %e,%e\n", ((b - 1) * 121 + (a - 1)), cuCreal(beamDASDAS[(b - 1) * 121 + (a - 1)]), cuCimag(beamDASDAS[(b - 1) * 121 + (a - 1)]));
	}
}

//核函数定义结束！

/*********************************************核函数定义结束***************************************************************************/
#include"mat.h"
static MATFile *pMF = NULL;
static mxArray *pA = NULL;

static MATFile *pMF_out = NULL;
static mxArray *pA_out = NULL;
static int TOTAL_ELEMENTS = (3534 * 16384);
int i;



int main()
{


	/*******************************************数据预处理部分---将v2的complex数据读进来**************************************/
	double *init_imaginary;
	double *init_real;
	cuDoubleComplex * init_M1 = new cuDoubleComplex[3534 * 16384];
	//complex<double> *init_M1;

	pMF = matOpen("D:\\WenshuaiZhao\\ProjectFiles\\VisualStudioFiles\\Ultrasound_GPU\\GPU_Data\\v2.mat", "r");//打开MAT文件，返回文件指针

	pA = matGetVariable(pMF, "v2");//获取v2.mat文件里的变量
	init_real = (double*)mxGetPr(pA);
	init_imaginary = (double*)mxGetPi(pA);


	//init_M1 = (complex<double>*)mxGetData(pA);
	//cuDoubleComplex temp;
	for (i = 0; i < TOTAL_ELEMENTS; i++)
	{
		//complex<double> temp(init_real[i], init_imaginary[i]);
		init_M1[i] = make_cuDoubleComplex(init_real[i], init_imaginary[i]);
	}//将从v2读出的实部和虚部重新组合在一起，放入init_M1.



	cuDoubleComplex * d_init_M1;
	cudaMalloc(&d_init_M1, 3534 * 16384 * sizeof(cuDoubleComplex));

	for (i = 0; i < 10; i++)
	{
		cout << "init_M1_"
			<< i
			<< "="
			<< cuCreal(init_M1[i]) << "," << cuCimag(init_M1[i]) << endl;
	}//验证init_M1的数据同v2的原始数据是一致的，复数。

	//for (i = 0; i < 10; i++)
	//{
	//	printf("init_M1 %d is %e,%e\n", i,cuCreal(init_M1[i]), cuCimag(init_M1[i]));
	//}//验证init_M1的数据同v2的原始数据是一致的，复数。



	cudaMemcpy(d_init_M1, init_M1, 3534 * 16384 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	printf("已经将init_M1考培到GPU全局内存\n");
	delete[] init_M1;

	/*******************************************数据预处理部分结束*********************************************************/

	//size_t size = 501 * 121 * sizeof(int);
	dim3 threadsPerBlock(1, 4);
	dim3 numBlocks(1, 1210);
	//由于线程内的内存不足以存放x_pointer的数据，因此将其放在全局内存
	cuDoubleComplex * x_pointer;
	cudaMalloc(&x_pointer, 128 * 128 * (4840) * sizeof(cuDoubleComplex));//121是线程总数
	//分配host beamDASDAS内存；
	cuDoubleComplex *h_beamDASDAS = new cuDoubleComplex[501 * 121];

	memset(h_beamDASDAS, 0, sizeof(h_beamDASDAS));

	//分配device的beamDASDAS内存；
	cuDoubleComplex *beamDASDAS;
	cudaMalloc(&beamDASDAS, 501 * 121 * sizeof(cuDoubleComplex));

	/* int *d_theta_start;
	int *d_theta_end;


	int *h_theta_start = new int[60621];
	int *h_theta_end = new int[60621];

	printf("h_theta_start[0]=%d\n", h_theta_start[0]);

	cudaMalloc(&d_theta_start, size);
	cudaMalloc(&d_theta_end, size);
	cudaError_t error = cudaGetLastError();
	printf("内存分配完毕...\n", cudaGetErrorString(error));
	printf("开始调用核函数...\n", cudaGetErrorString(error)); */

	DWORD time_kernel = GetTickCount(); //获取毫秒级数目

	theta_calculate_WithCuda << <numBlocks, threadsPerBlock >> > (d_init_M1, beamDASDAS, x_pointer);
	//theta_calculate_WithCuda << <(1,1), (1,1) >> >(d_theta_start, d_theta_end);
	cudaDeviceSynchronize();

	cout << "核函数执行共用了：" << GetTickCount() - time_kernel << "毫秒" << endl;

	cudaError_t error = cudaGetLastError();
	printf("CUDA error: %s\n", cudaGetErrorString(error));

	printf("核函数调用完毕...\n", cudaGetErrorString(error));

	cudaMemcpy(h_beamDASDAS, beamDASDAS, 60621 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_theta_end, d_theta_end, 60621 * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaFree(d_theta_start);
	cudaFree(beamDASDAS);
	for (int i = 0; i < 968; i++)
	{
		printf("h_beamDASDAS[%d] is %e,%e\n", i, cuCreal(h_beamDASDAS[i]), cuCimag(h_beamDASDAS[i]));
	}

	cudaDeviceReset();
	delete[] h_beamDASDAS;
	getchar();

	/* delete[] h_theta_start;
	delete[] h_theta_end; */

}






