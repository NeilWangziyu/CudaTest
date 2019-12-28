
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

__device__ void swap_rows(cuDoubleComplex(*Rxx1)[52], int r1, int r2) {
	cuDoubleComplex temp[1][52];
	int i, j;
	for (i = 0; i < 52; i++)
	{
		temp[0][i] = Rxx1[r1][i];
		Rxx1[r1][i] = Rxx1[r2][i];
		Rxx1[r2][i] = temp[0][i];
	}

	/*temp[1] = Rxx1[r1];
	Rxx1[r1] = Rxx1[r2];
	Rxx1[r2] = temp[1];*/
}

__device__ void scale_row(cuDoubleComplex(*Rxx1)[52], int r, cuDoubleComplex scalar, int L1) {
	int i;
	for (i = 0; i < L1; i++) {
		Rxx1[r][i] = cuCmul(Rxx1[r][i], scalar);
	}
}

__device__ void shear_row(cuDoubleComplex(*Rxx1)[52], int r1, int r2, cuDoubleComplex scalar, int L1) {
	int i;
	for (i = 0; i < L1; i++) {
		Rxx1[r1][i] = cuCadd(Rxx1[r1][i], cuCmul(Rxx1[r2][i], scalar));
	}
}

__device__ void  matrix_inv(cuDoubleComplex(*Rxx1)[52], int L1) {
	cuDoubleComplex Rxx1_out[52][52] = { 0.0 };
	int i, j, r;
	cuDoubleComplex scalar;
	cuDoubleComplex shear_needed;
	for (i = 0; i < L1; i++)
	{
		Rxx1_out[i][i] = make_cuDoubleComplex( 1.0,0.0);
	}
	for (i = 0; i < L1; i++)
	{
		if ((cuCreal(Rxx1[i][i]) == 0.0) && (cuCimag(Rxx1[i][i]) == 0.0)) {
			for (r = i + 1; r < L1; r++)
			{
				if ((cuCreal(Rxx1[i][i]) != 0.0) || (cuCimag(Rxx1[i][i]) != 0.0))
				{
					break;
				}
			}
			swap_rows(Rxx1, i, r);
			swap_rows(Rxx1_out, i, r);
		}
		scalar = cuCdiv(make_cuDoubleComplex(1.0, 0.0), Rxx1[i][i]);
		scale_row(Rxx1, i, scalar, L1);
		scale_row(Rxx1_out, i, scalar, L1);
		for (j = 0; j < L1; j++)
		{
			if (i == j) { continue; }
			shear_needed = make_cuDoubleComplex(-cuCreal(Rxx1[j][i]), -cuCimag(Rxx1[j][i]));
			shear_row(Rxx1, j, i, shear_needed, L1);
			shear_row(Rxx1_out, j, i, shear_needed, L1);
		}
	}
	Rxx1 = Rxx1_out;

}

__device__ void output_beam(int M_elements, int emit_aperture, int L1, int L2, cuDoubleComplex *beamDASDAS, cuDoubleComplex *x_pointer, cuDoubleComplex *wmv1, cuDoubleComplex *wmv2, int idx, int a, int b) {
	int l1, l2;
	int i, j;
	cuDoubleComplex xL[52][52] = { 0.0 };
	for (l1 = 0; l1 < M_elements - L1; l1++)
	{
		for (l2 = 0; l2 < emit_aperture - L2; l2++)
		{
			for (i = 0; i < L2; i++)
			{
				for (j = 0; j < L1; j++)
				{
					xL[i][j] = cuCadd(xL[i][j], x_pointer[idx * 128 * 128 + 128 * (i + l2) + (l1 + j)]);
				}
			}
		}
	}
	cuDoubleComplex temp[52] = { 0.0 };
	for (l1 = 0; l1 < L1; l1++)
	{
		for (l2 = 0; l2 < L2; l2++)
		{
			temp[l1] = cuCadd(temp[l1], cuCmul(wmv2[52 * idx + l2], xL[l2][l1]));
		}
	}
	for (int i = 0; i < L1; i++)
	{
		beamDASDAS[(b - 1) * 121 + (a - 1)] = cuCadd(beamDASDAS[(b - 1) * 121 + (a - 1)], cuCmul(temp[i], wmv1[i]));
	}

}

__device__ void weigt_cal_MV(int M_elements, int emit_aperture, cuDoubleComplex *x_pointer, cuDoubleComplex *wmv1, cuDoubleComplex *wmv2, int idx, int L1, int L2)
{
	int i, j, k, l;
	
	printf("Have entered into weight_cal_MV function!\n");
	cuDoubleComplex Rxx1[52][52] = { 0.0 };
	printf("Have finished the definiton of Rxx1[52][52]!\n");
	for (l = 0; l < M_elements - L1; l++)
	{
		printf("Have entered into Rxx1 calculation function!\n");
		for (i = 0; i < L1; i++)
		{
			for (j = 0; j < L1; j++)
			{
				for (k = 0; k < M_elements; k++)
				{
					Rxx1[i][j] = cuCadd(Rxx1[i][j],cuCmul(x_pointer[idx * 128 * 128 + (k) * 128 + i + l], cuConj(x_pointer[idx * 128 * 128 + k * 128 + j + l])));
					
				}
			}
		}
	}

	printf("Have calculated the Rxx1!\n");

	cuDoubleComplex trace = make_cuDoubleComplex(0.0,0.0);
	for (i = 0; i < L1; i++)
	{
		trace = cuCadd(Rxx1[i][i], trace);
	}
	for (i = 0; i < L1; i++)
	{
		Rxx1[i][i] = cuCadd(Rxx1[i][i], cuCmul(trace, (make_cuDoubleComplex((double)(0.01 / L1), 0.0))));
	}

	printf("Have add the trace with Rxx1!\n");

	matrix_inv(Rxx1, L1);

	printf("Have finished the inversion of Rxx1!\n");

	cuDoubleComplex right_value = make_cuDoubleComplex(0.0, 0.0);
	for (int i = 0; i < L1; i++)
	{
		for (int j = 0; j < L1; j++)
		{
			right_value = cuCadd(right_value, Rxx1[i][j]);
		}
	}
	right_value = cuCdiv(make_cuDoubleComplex(1.0, 0.0), right_value);
	for (i = 0; i < L1; i++)
	{
		for (j = 0; j < L1; j++)
		{
			wmv1[i + idx * 52] = cuCadd(Rxx1[i][j], wmv1[i + idx * 52]);
		}
		wmv1[i + idx * 52] = cuCmul(wmv1[i + idx * 52], right_value);
	}

	printf("Have got the final values of wmv1!\n");
	//To now, we have calculated the weight of wmv1, and similarly we should calculate the weights fo wmv2

	
	cuDoubleComplex Rxx2[52][52] = { 0.0 };

	for (l = 0; l < M_elements - L2; l++)
	{
		for (i = 0; i < L2; i++)
		{
			for (j = 0; j < L2; j++)
			{
				for (k = 0; k < emit_aperture; k++)
				{
					Rxx2[i][j] = Rxx2[i][j] = cuCmul(x_pointer[idx * 128 * 128 + k + (i + l) * 128], cuConj(x_pointer[idx * 128 * 128 + k + (j + l) * 128]));
				}
			}
		}
	}
	trace = make_cuDoubleComplex(0.0, 0.0);
	for (i = 0; i < L2; i++)
	{
		trace = cuCadd(Rxx2[i][i], trace);
	}
	for (i = 0; i < L1; i++)
	{
		Rxx2[i][i] = cuCadd(Rxx2[i][i], cuCmul(trace, (make_cuDoubleComplex((double)(0.01 / L2), 0.0))));
	}

	printf("Have got Rxx2!\n");

	matrix_inv(Rxx2, L2);

	printf("Have finished the inversion of Rxx2!\n");

	right_value = make_cuDoubleComplex(0.0, 0.0);
	for (int i = 0; i < L2; i++)
	{
		for (int j = 0; j < L2; j++)
		{
			right_value = cuCadd(right_value, Rxx2[i][j]);
		}
	}
	right_value = cuCdiv(make_cuDoubleComplex(1.0, 0.0), right_value);
	for (i = 0; i < L2; i++)
	{
		for (j = 0; j < L2; j++)
		{
			wmv2[i + idx * 52] = cuCadd(Rxx2[i][j], wmv2[i + idx * 52]);
		}
		wmv2[i + idx * 52] = cuCmul(wmv2[i + idx * 52], right_value);
	}

	printf("Have got the final wmv2!\n");

}

__global__ void theta_calculate_WithCuda(cuDoubleComplex *d_init_M1, cuDoubleComplex *beamDASDAS, cuDoubleComplex *x_pointer, cuDoubleComplex *wmv1, cuDoubleComplex *wmv2)
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

		double subarray_length = 0.4;
		int L1 = rint(M_elements*subarray_length);
		int L2 = rint(emit_aperture*subarray_length);
		printf("Begin to use weight_cal_MV function!\n");

		weigt_cal_MV(M_elements, emit_aperture, x_pointer, wmv1, wmv2, idx,L1,L2);

		printf("Have finished calculating the weights!\n");

		for (i = 0; i < 52; i++)
		{
			printf("wmv1[%d] is %f,%f\n", i, cuCreal(wmv1[idx * 52 + i]), cuCimag(wmv1[idx * 52 + i]));
		}
		//output_beam(beamDASDAS,x_pointer,wmv1,wmv2,idx);
		output_beam(M_elements, emit_aperture, L1, L2, beamDASDAS, x_pointer, wmv1, wmv2, idx, a, b);

		//printf("x_pointer[0] of thread %d is %e,%e\n", idx, cuCreal(x_pointer[0+ idx * 128 * 128]), cuCimag(x_pointer[0+ idx * 128 * 128]));
		//*************计算延时和回波矩阵完毕*********************************************************************//
		//*************接下来计算本像素点的beamDASDAS*******************************************//
		// for (i = 0; i < M_elements; i++)
		// {
		// 	int j;
		// 	for (j = 0; j < M_elements; j++)
		// 	{
		// 		cuDoubleComplex temp_1 = make_cuDoubleComplex(0, 0);
		// 		temp_1 = cuCadd(x_pointer[i*M_elements + j + (idx * 128 * 128)], beamDASDAS[(b - 1) * 121 + (a - 1)]);//注意此处x_pointer的地址也加上了idx*128*128
		// 		beamDASDAS[(b - 1) * 121 + (a - 1)] = temp_1;
		// 	}
		// }
		// beamDASDAS[(b - 1) * 121 + (a - 1)] = make_cuDoubleComplex(cuCreal(beamDASDAS[(b - 1) * 121 + (a - 1)]) / ((double)(M_elements*M_elements)), cuCimag(beamDASDAS[(b - 1) * 121 + (a - 1)]) / ((double)(M_elements*M_elements)));
		//printf("beam %d is %e,%e\n", ((b - 1) * 121 + (a - 1)), cuCreal(beamDASDAS[(b - 1) * 121 + (a - 1)]), cuCimag(beamDASDAS[(b - 1) * 121 + (a - 1)]));
	}
}

//核函数定义结束！

/*********************************************核函数定义结束***************************************************************************/
/*********Begin for the MV algorithms**********************/







#include "mat.h"

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
	dim3 threadsPerBlock(1, 1);
	dim3 numBlocks(1, 1);
	//由于线程内的内存不足以存放x_pointer的数据，因此将其放在全局内存
	cuDoubleComplex * x_pointer;
	cudaMalloc(&x_pointer, 128 * 128 * (1 * 1) * sizeof(cuDoubleComplex));//121是线程总数
	//分配host beamDASDAS内存；
	cuDoubleComplex *h_beamDASDAS = new cuDoubleComplex[1 * 1];

	memset(h_beamDASDAS, 0, sizeof(h_beamDASDAS));

	//Begin to distribute the weight memory
	cuDoubleComplex *wmv1;
	cuDoubleComplex *wmv2;
	cudaMalloc(&wmv1, (1*1) * 52 * sizeof(cuDoubleComplex));
	cudaMemset(wmv1, 0, sizeof(wmv1));
	cudaMalloc(&wmv2, (1*1) * 52 * sizeof(cuDoubleComplex));
	cudaMemset(wmv2, 0, sizeof(wmv2));


	//分配device的beamDASDAS内存；
	cuDoubleComplex *beamDASDAS;
	cudaMalloc(&beamDASDAS, 1 * 1 * sizeof(cuDoubleComplex));

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

	theta_calculate_WithCuda << <numBlocks, threadsPerBlock >> > (d_init_M1, beamDASDAS, x_pointer,wmv1,wmv2);
	//theta_calculate_WithCuda << <(1,1), (1,1) >> >(d_theta_start, d_theta_end);
	cudaDeviceSynchronize();

	cout << "核函数执行共用了：" << GetTickCount() - time_kernel << "毫秒" << endl;

	cudaError_t error = cudaGetLastError();
	printf("CUDA error: %s\n", cudaGetErrorString(error));

	printf("核函数调用完毕...\n", cudaGetErrorString(error));

	cudaMemcpy(h_beamDASDAS, beamDASDAS, 121 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_theta_end, d_theta_end, 60621 * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaFree(d_theta_start);
	cudaFree(wmv1);
	cudaFree(wmv2);
	cudaFree(beamDASDAS);
	
	for (int i = 0; i < 1; i++)
	{
		printf("h_beamDASDAS[%d] is %e,%e\n", i, cuCreal(h_beamDASDAS[i]), cuCimag(h_beamDASDAS[i]));
	}

	cudaDeviceReset();
	delete[] h_beamDASDAS;
	getchar();

	/* delete[] h_theta_start;
	delete[] h_theta_end; */

}
