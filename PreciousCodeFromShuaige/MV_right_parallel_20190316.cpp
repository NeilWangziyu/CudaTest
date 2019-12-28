
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
#include <cublasXt.h>
#include <cublas_v2.h>

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




__device__ __host__ void  matrix_inv_lu(cuDoubleComplex *u, int n, cuDoubleComplex x[]) {
	int i, r, k;
	for (r = 0; r <= n - 1; r++)
	{
		for (i = r; i <= n; i++)
			for (k = 0; k <= r - 1; k++)
				//*(u + r * (n + 1) + i) -= *(u + r * (n + 1) + k)**(u + k * (n + 1) + i);
				*(u + r * (n + 1) + i) = cuCsub(*(u + r * (n + 1) + i), cuCmul(*(u + r * (n + 1) + k), *(u + k * (n + 1) + i)));
		for (i = r + 1; i <= n - 1; i++)
		{
			for (k = 0; k <= r - 1; k++)
				//*(u + i * (n + 1) + r) -= *(u + i * (n + 1) + k)*(*(u + k * (n + 1) + r));
				*(u + i * (n + 1) + r) = cuCsub(*(u + i * (n + 1) + r), cuCmul(*(u + i * (n + 1) + k), *(u + k * (n + 1) + r)));
			//*(u + i * (n + 1) + r) /= *(u + r * (n + 1) + r);
			*(u + i * (n + 1) + r) = cuCdiv(*(u + i * (n + 1) + r), *(u + r * (n + 1) + r));
		}
	}

	for (i = n - 1; i >= 0; i--)
	{
		for (r = n - 1; r >= i + 1; r--)
			//*(u + i * (n + 1) + n) -= *(u + i * (n + 1) + r)*x[r];
			*(u + i * (n + 1) + n) = cuCsub(*(u + i * (n + 1) + n), cuCmul(*(u + i * (n + 1) + r), x[r]));
		//x[i] = *(u + i * (n + 1) + n) / (*(u + i * (n + 1) + i));
		x[i] = cuCdiv(*(u + i * (n + 1) + n), *(u + i * (n + 1) + i));
	}

}

__device__ void output_beam(int M_elements, int emit_aperture, int L1, int L2, cuDoubleComplex *beamDASDAS, cuDoubleComplex *x_pointer, cuDoubleComplex *wmv1, cuDoubleComplex *wmv2, int idx, int a, int b, cuDoubleComplex *xL, cuDoubleComplex *temp) {
	int l1, l2;
	int i, j;
	// cuDoubleComplex xL[52][52];
	//printf("Have entered into the output_beam function!\n");
	for (l1 = 0; l1 < M_elements - L1 + 1; l1++)//Rows
	{
		//printf("Have enter into loop1...%d!\n",l1);
		for (l2 = 0; l2 < emit_aperture - L2 + 1; l2++)//columns
		{
			for (i = 0; i < L2; i++)//Rows
			{
				for (j = 0; j < L1; j++)//Columns
				{
					//xL[i*L2+j] = cuCadd(xL[i*L2 + j], x_pointer[idx * 128 * 128 + M_elements * (i + l2) + (l1 + j)]);
					xL[i*L2 + j] = cuCadd(xL[i*L2 + j], x_pointer[M_elements * (i + l2) + (l1 + j)]);
				}
			}
		}
	}
	/*printf("Have calculated the xL values!\n");

	for (i = 0; i < L2*3; i++)
	{
		printf("xL[%d] is %e,%e\n",i,cuCreal(xL[i]),cuCimag(xL[i]));
	}*/


	//printf("Have calculate the xL values!\n");
	//__shared__ cuDoubleComplex temp[52];
	for (l1 = 0; l1 < L1; l1++)
	{
		for (l2 = 0; l2 < L2; l2++)
		{
			//temp[l1] = cuCadd(temp[l1], cuCmul(cuConj(wmv2[52 * idx + l2]), xL[l2*L2+l1]));
			//temp[l1] = cuCadd(temp[l1], cuCmul((wmv2[52 * idx + l2]), xL[l2*L2 + l1]));
			temp[l1] = cuCadd(temp[l1], cuCmul((wmv2[l2]), xL[l2*L2 + l1]));
		}
	}

	/*for (i = 0; i < L2 ; i++)
	{
		printf("temp[%d] is %e,%e\n", i, cuCreal(temp[i]), cuCimag(temp[i]));
	}*/


	for (int i = 0; i < L1; i++)
	{
		beamDASDAS[(b - 1) * 121 + (a - 1)] = cuCadd(beamDASDAS[(b - 1) * 121 + (a - 1)], cuCmul(temp[i], wmv1[i]));
	}

}



__device__ __host__ void weigt_cal_MV(int M_elements, int emit_aperture, cuDoubleComplex *x_pointer, cuDoubleComplex *wmv1, cuDoubleComplex *wmv2, int idx, int L1, int L2, cuDoubleComplex *Rxx)
{
	int i, j, k, l;

	//printf("Have entered into weight_cal_MV function!\n");
	// cuDoubleComplex Rxx[52 * 53];
	/*__shared__ cuDoubleComplex Rxx[52][52];
	Rxx[0][0] = make_cuDoubleComplex(1.0, 1.0);
	Rxx[0][1] = make_cuDoubleComplex(1.0, 2.0);*/
	//printf("Have finished the definiton of Rxx1[52][52]!\n");

	//printf("Have entered into the weight_cal_MV function 1...!\n");

	for (l = 0; l < M_elements - L1 + 1; l++)
	{
		//printf("Have entered into Rxx1 calculation function 2....%d!\n",l);
		for (i = 0; i < L1; i++)
		{
			for (j = 0; j < L1; j++)
			{
				for (k = 0; k < M_elements; k++)
				{
					//Rxx[i*L1 + j+i] = cuCadd(Rxx[i*L1 + j+i], cuCmul(x_pointer[idx * 128 * 128 + (k)* M_elements + i + l], cuConj(x_pointer[idx * 128 * 128 + k * M_elements + j + l])));
					Rxx[i*L1 + j + i] = cuCadd(Rxx[i*L1 + j + i], cuCmul(x_pointer[(k)* M_elements + i + l], cuConj(x_pointer[k * M_elements + j + l])));
				}
				//printf("Rxx1[%d][%d]=%e,%e\n", i, j, cuCreal(Rxx1[i][j]), cuCimag(Rxx1[i][j]));
				
			}
			//printf("Have entered into Rxx1 calculation function 3....,%d!\n",i);
		}


	}

	//printf("Have finished calculating the Rxx1!\n");



	cuDoubleComplex trace = make_cuDoubleComplex(0.0, 0.0);
	for (i = 0; i < L1; i++)
	{
		trace = cuCadd(Rxx[i*L1 + i+i], trace);
	}
	//printf("trace is %e,%e\n",cuCreal(trace),cuCimag(trace));
	for (i = 0; i < L1; i++)
	{
		Rxx[i*L1 + i+i] = cuCadd(Rxx[i*L1 + i+i], cuCmul(trace, (make_cuDoubleComplex((double)(0.01 / L1), 0.0))));
	}

	//printf("Have finished trace loaded!\n");

	

	//printf("Have add the trace with Rxx1!\n");

	//Begin to construct the new linear equation matrix
	/*__shared__ cuDoubleComplex Rxx_new[52 * 53];
	for (i = 0; i < L1; i++)
	{
		for (j = 0; j < L1; j++)
		{
			Rxx_new[i*L1 + j + i] = Rxx[i*L1 + j];

		}
		Rxx_new[(i + 1)*L1 + i] = make_cuDoubleComplex(1.0, 0.0);
	}*/

	for (i = 0; i < L1; i++)
	{
		/*for (j = 0; j < L1; j++)
		{
			Rxx[i*L1 + j + i] = Rxx[i*L1 + j];

		}*/
		Rxx[(i + 1)*L1 + i] = make_cuDoubleComplex(1.0, 0.0);
	}
	//printf("Have finished FINAL Rxx !\n");

	/*for (i = 0; i < 2 * L1 + 3; i++)
	{
		printf("Rxx1[%d]is %e,%e\n", i, cuCreal(Rxx[i]), cuCimag(Rxx[i]));
	}*/



	//Have finished constructing the new matrix 
	//Begin to employ the new LU method.

	//matrix_inv_lu(Rxx_new, L1, wmv1);

	matrix_inv_lu(Rxx, L1, wmv1);
	

	//printf("Have finished Rxx inverse!\n");

	cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
	for (i = 0; i < L1; i++)
	{
		sum = cuCadd(sum, wmv1[i]);
	}

	for (i = 0; i < L1; i++)
	{
		wmv1[i] = cuCdiv(wmv1[i], sum);
		wmv1[i] = make_cuDoubleComplex(cuCreal(wmv1[i]), 0.0);
	}

	/*for (i = 0; i < L1; i++)
	{
		printf("wmv1[%d] is %e,%e\n", i, cuCreal(wmv1[i]), cuCimag(wmv1[i]));
	}
*/

	/*for (i = 0; i < L1; i++)
	{

		printf("wmv1[%d] is %e,%e\n", i, cuCreal(wmv1[i]), cuCimag(wmv1[i]));

	}*/

	//The above has been checked right!


	//printf("Have finished wmv1 !\n");

	//cuDoubleComplex Rxx2[52 * 52] = { 0.0 };
	//Rxx[52 * 53] = { 0.0 };
	memset(Rxx,0.0,52*53*sizeof(cuDoubleComplex));
	//printf("Have entered into Rxx2!\n");
	for (l = 0; l < emit_aperture - L2 + 1; l++)
	{
		for (i = 0; i < L2; i++)
		{
			for (j = 0; j < L2; j++)
			{
				for (k = 0; k < emit_aperture; k++)
				{
					//Rxx[i*L2 + j+i] = cuCadd(Rxx[i*L2 + j+i], cuCmul(x_pointer[idx * 128 * 128 + k + (i + l) * emit_aperture], cuConj(x_pointer[idx * 128 * 128 + k + (j + l) * emit_aperture])));
					Rxx[i*L2 + j + i] = cuCadd(Rxx[i*L2 + j + i], cuCmul(x_pointer[k + (i + l) * emit_aperture], cuConj(x_pointer[k + (j + l) * emit_aperture])));
				}
			}
		}
	}

	//printf("Have finished Rxx2 calculation!\n");

	trace = make_cuDoubleComplex(0.0, 0.0);
	for (i = 0; i < L2; i++)
	{
		trace = cuCadd(Rxx[i*L2 + i+i], trace);
	}
	for (i = 0; i < L2; i++)
	{
		Rxx[i*L2 + i+i] = cuCadd(Rxx[i*L2 + i+i], cuCmul(trace, (make_cuDoubleComplex((double)(0.01 / L2), 0.0))));
	}

	//printf("Have got Rxx2 after the trace operation!\n");

	//cuDoubleComplex Rxx2_new[52 * 53] = { 0.0 };
	//Rxx_new[52 * 53] = { 0.0 };
	for (i = 0; i < L2; i++)
	{
		/*for (j = 0; j < L2; j++)
		{
			Rxx_new[i*L2 + j + i] = Rxx[i*L2 + j];

		}*/
		Rxx[(i + 1)*L2 + i] = make_cuDoubleComplex(1.0, 0.0);
	}
	//Have finished constructing the new matrix 
	//Begin to employ the new LU method.
	matrix_inv_lu(Rxx, L2, wmv2);
	cuDoubleComplex sum_2 = make_cuDoubleComplex(0.0, 0.0);
	for (i = 0; i < L2; i++)
	{
		sum_2 = cuCadd(sum_2, wmv2[i]);
	}

	for (i = 0; i < L2; i++)
	{
		wmv2[i] = cuCdiv(wmv2[i], sum);
		wmv2[i] = make_cuDoubleComplex(cuCreal(wmv2[i]), 0.0);
	}

	/*for (i = 0; i < L2; i++)
	{
		printf("wmv2[%d] is %e,%e\n", i, cuCreal(wmv2[i]), cuCimag(wmv2[i]));
	}
*/

}


__global__ void theta_calculate_WithCuda(cuDoubleComplex *d_init_M1, cuDoubleComplex *beamDASDAS, cuDoubleComplex *x_pointer, cuDoubleComplex *wmv1, cuDoubleComplex *wmv2, cuDoubleComplex *Rxx, cuDoubleComplex *xL, cuDoubleComplex *temp)
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
		int temp_Or;
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
				temp_Or = i - theta_start;
				delay_pointer[temp_Or] = d_t_delay[Trans_elements - 1] + d_t_delay[i - 1];

				delay_pointer[temp_Or] = delay_pointer[temp_Or] - t1;
				delay_pointer[temp_Or] = ((delay_pointer[temp_Or] * fs) + 25.5) + zerolength;
				delay_int[temp_Or] = (round)(delay_pointer[temp_Or]);
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

		//printf("Have finished calculating the x_pointer!\n");

		double subarray_length = 0.4;
		int L1 = rint(M_elements*subarray_length);
		int L2 = rint(emit_aperture*subarray_length);
		//printf("Begin to use weight_cal_MV function!\n");

		weigt_cal_MV(M_elements, emit_aperture, x_pointer+idx*128*128, wmv1+idx*52, wmv2+idx*52, idx,L1,L2,Rxx+idx*52*53);

		//printf("Have finished calculating the weights!\n");

		
		output_beam(M_elements, emit_aperture, L1, L2, beamDASDAS, x_pointer+idx*128*128, wmv1+idx*52, wmv2+idx*52, idx, a, b,xL+idx*52*52,temp+idx*52);
		//printf("beamDASDAS[%d] is %e,%e", ((b - 1) * 121 + (a - 1)), cuCreal(beamDASDAS[(b - 1) * 121 + (a - 1)]), cuCimag(beamDASDAS[(b - 1) * 121 + (a - 1)]));
		
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


	cudaMemcpy(d_init_M1, init_M1, 3534 * 16384 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	printf("已经将init_M1考培到GPU全局内存\n");
	delete[] init_M1;

	/*******************************************数据预处理部分结束*********************************************************/

	//size_t size = 501 * 121 * sizeof(int);
	dim3 threadsPerBlock(1, 10);
	dim3 numBlocks(1, 501);
	int threads = 10;
	int blocks = 501;
	//由于线程内的内存不足以存放x_pointer的数据，因此将其放在全局内存
	cuDoubleComplex * x_pointer;
	cudaMalloc(&x_pointer, 128 * 128 * (threads*blocks) * sizeof(cuDoubleComplex));//121是线程总数
	//分配host beamDASDAS内存；
	cuDoubleComplex *h_beamDASDAS = new cuDoubleComplex[121*501];

	memset(h_beamDASDAS, 0, sizeof(cuDoubleComplex)*121*501);

	//Begin to distribute the weight memory
	

	/*int *M_elements;
	cudaMalloc(&M_elements, (threads*blocks) * sizeof(int));*/


	//分配device的beamDASDAS内存；
	cuDoubleComplex *beamDASDAS;
	cudaMalloc(&beamDASDAS, (threads*blocks) * sizeof(cuDoubleComplex));

	cuDoubleComplex *wmv1;
	cuDoubleComplex *wmv2;
	cudaMalloc(&wmv1, (threads*blocks) * 52 * sizeof(cuDoubleComplex));
	cudaMemset(wmv1, 0.0, sizeof(cuDoubleComplex) * 52 * (threads*blocks));
	cudaMalloc(&wmv2, (threads*blocks) * 52 * sizeof(cuDoubleComplex));
	cudaMemset(wmv2, 0.0, sizeof(cuDoubleComplex) * 52 * (threads*blocks));

	cuDoubleComplex *Rxx;
	cuDoubleComplex *xL;
	cuDoubleComplex *temp;
	cudaMalloc(&Rxx, (threads*blocks) * 52 *53* sizeof(cuDoubleComplex));
	cudaMemset(Rxx, 0.0, sizeof(cuDoubleComplex) * 52*53* (threads*blocks));

	cudaMalloc(&xL, (threads*blocks) * 52 * 52 * sizeof(cuDoubleComplex));
	cudaMemset(xL, 0.0, sizeof(cuDoubleComplex) * 52 * 52 * (threads*blocks));

	cudaMalloc(&temp, (threads*blocks) * 52 * sizeof(cuDoubleComplex));
	cudaMemset(temp, 0.0, sizeof(cuDoubleComplex) * 52 * (threads*blocks));


	DWORD time_kernel = GetTickCount(); //获取毫秒级数目

	theta_calculate_WithCuda <<<numBlocks, threadsPerBlock >>> (d_init_M1, beamDASDAS, x_pointer,wmv1,wmv2,Rxx,xL,temp);
	//Above has got the x_pointer and M_elements, they are all in device memory;
	//Then we should calculate the wmv and beamDASDAS by cublasxt;

	cudaDeviceSynchronize();

	

	/*double subarray_length = 0.4;
	int L1 = rint(M_elements*subarray_length);
	int L2 = rint(emit_aperture*subarray_length);
	for (int t = 0; t < (threads*blocks); t++)
	{
		for(int l = 0; l < M_elements - L1 + 1; l++)

	}*/




	cout << "核函数执行共用了：" << GetTickCount() - time_kernel << "毫秒" << endl;

	cudaError_t error = cudaGetLastError();
	printf("CUDA error: %s\n", cudaGetErrorString(error));

	printf("核函数调用完毕...\n", cudaGetErrorString(error));

	cudaMemcpy(h_beamDASDAS, beamDASDAS, (threads*blocks) * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_theta_end, d_theta_end, 60621 * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaFree(d_theta_start);
	cudaFree(wmv1);
	cudaFree(wmv2);
	cudaFree(beamDASDAS);
	
	for (int i = 0; i < (501); i++)
	{
		printf("h_beamDASDAS[%d] is %e,%e\n", i, cuCreal(h_beamDASDAS[i]), cuCimag(h_beamDASDAS[i]));
	}

	cudaDeviceReset();
	delete[] h_beamDASDAS;
	getchar();

	/* delete[] h_theta_start;
	delete[] h_theta_end; */

}
