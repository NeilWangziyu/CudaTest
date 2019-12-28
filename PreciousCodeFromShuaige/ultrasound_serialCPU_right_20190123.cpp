//#include "stdafx.h" 
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"mat.h"
#include <stdio.h>
#include<stdlib.h>
#include<cmath>
#include "mclmcr.h"
#include"matrix.h"
//#include <opencv2/core/core.hpp>
//#include<opencv2/highgui/highgui.hpp>

//#include<complex.h>
#include<complex>

using namespace std;
//using namespace cv;

static int TOTAL_ELEMENTS = (3534 * 16384);

static MATFile *pMF = NULL;
static mxArray *pA = NULL;

static MATFile *pMF_out = NULL;
static mxArray *pA_out = NULL;

static MATFile *beamF_out = NULL;
static mxArray *beamM_out = NULL;

#define N_elements 128

int M_elements;
int emit_aperture;
int emit_pitch;
int emit_start;
int Trans_elements;
int zerolength = 1000;
int fs = 40000000;



//double A[3534][16384];

double x_begin = (-3.0 / 1000);
double z_begin = 27.5 / 1000;
double d_x = 0.05 / 1000;
double d_z = 0.05 / 1000;
double pitch = (3.08e-4);
double t1 = (3.895e-5);

double a_begin = -double((N_elements - 1)) / 2 * pitch;

int c = 1540;

int xsize = 501;
int ysize = 121;

double *init_imaginary;
double *init_real;
unsigned char Result[501][121];

int main()
{
	int i;
	double index_finished = 0;
	int index_1;
	int row_v2;
	int column_v2;
	int temp;
	int temp_1;
	int temp_2;
	/*******************************************数据预处理部分---将v2的complex数据读进来**************************************/
	complex<double> * init_M1 = new complex<double>[3534 * 16384];
	//complex<double> *init_M1;

	pMF = matOpen("D:\\WenshuaiZhao\\ProjectFiles\\VisualStudioFiles\\Ultrasound_GPU\\GPU_Data\\v2.mat", "r");//打开MAT文件，返回文件指针

	pA = matGetVariable(pMF, "v2");//获取v2.mat文件里的变量
	init_real = (double*)mxGetPr(pA);
	init_imaginary = (double*)mxGetPi(pA);


	//init_M1 = (complex<double>*)mxGetData(pA);

	for (i = 0; i < TOTAL_ELEMENTS; i++)
	{
		complex<double> temp(init_real[i], init_imaginary[i]);
		init_M1[i] = temp;
	}//将从v2读出的实部和虚部重新组合在一起，放入init_M1.

	for (i = 0; i < 10; i++)
	{
		cout << "init_M1"
			<< i
			<< "="
			<< init_M1[i] << endl;
	}//验证init_M1的数据同v2的原始数据是一致的，复数。

	 /*******************************************数据预处理部分结束*********************************************************/
	int M = mxGetM(pA);
	int N = mxGetN(pA);

	int a, b;
	double xp, zp;
	double xr[N_elements];
	double t_delay[N_elements];

	double theta_ctrb[N_elements];
	double theta_threshold;
	int theta_start;
	int theta_end;

	int k;
	complex<double> *beamDASDAS = new complex<double>[xsize*ysize];

	memset(beamDASDAS, 0, sizeof(beamDASDAS));

	int verify[501 * 121];

	//***************************************主循环开始*************************************//

	for (a = 1; a < (ysize + 1); a++)
	{
		for (b = 1; b < (xsize + 1); b++)
		{
			theta_threshold = 2.8;
			theta_start = 0;
			theta_end = 0;
			xp = x_begin + d_x*(a - 1);
			zp = z_begin + d_z*(b - 1);

			/********************************计算子孔径的大小，以及，对应像素点，各个阵元的延时***************************/
			for (i = 0; i < (N_elements); i++)
			{
				xr[i] = a_begin + pitch*i;
				t_delay[i] = sqrt(pow((xp - xr[i]), 2) + pow(zp, 2)) / c;
				theta_ctrb[i] = zp / (abs(xp - xr[i]));

			}


			/*for (i = 0; i < (N_elements); i++)
			{
			cout << "theta_ctrb "<<i<<"is " << theta_ctrb[i] << endl;

			}*/

			k = 1;
			while ((k <= N_elements) && (theta_ctrb[k - 1] < theta_threshold))
			{
				k = k + 1;
			}//求出theta_start的值




			if (k == N_elements + 1)
				NULL;
			//开始求theta_end的值
			else
			{
				theta_start = k;
				//	cout << "k or theta_start is" << k << endl;

				while ((k <= N_elements) && (theta_ctrb[k - 1] >= theta_threshold))
				{
					k = k + 1;
				}
				if (k == N_elements + 1)
					theta_end = k - 1;
				else
				{
					theta_end = k;
				}
				//	cout << "k or theta_end is" << k << endl;
			}

			//求出theta_end的值
			//验证theta_start的值
			/*if ((a < 2) && (b < 9))
			cout << "theta_start is:" << theta_start << endl;*/

			//验证计算出的theta_start和theta_end正确与否
			/*cout << "theta_start=" << theta_start << endl;
			cout << "theta_end=" << theta_end << endl;*/
			//verify[(a - 1)*ysize + b - 1] = theta_start;

			/*******************************计算出了theta_start和theta_end******************************/

			/*******************************为计算正确延时和回波矩阵做准备******************************/
			M_elements = theta_end - theta_start + 1;
			emit_aperture = M_elements;
			emit_pitch = 1;
			emit_start = theta_start;

			complex<double> * x_pointer = new complex<double>[emit_aperture*M_elements];//为x数组分配动态内存，记得用完delete，x二维数组，是经过正取延时，并且取正确孔径的，回波矩阵。维度为M_elements*M_elements
			double * delay_pointer = new double[M_elements * 1];//为delay数组分配动态内存，记得用完delete
			int *delay_int = new int[M_elements * 1];//为取整后的delay值分配内存。

													 //为刚分配内存的三个数组初始化
			memset(delay_pointer, 0, sizeof(delay_pointer));
			memset(delay_int, 0, sizeof(delay_pointer));
			memset(x_pointer, 0, sizeof(x_pointer));


			//*************接下来计算正确延时和回波矩阵x【M_elements*M_elements】*******************************************//
			for (Trans_elements = emit_start; Trans_elements <= (emit_start + emit_pitch*(emit_aperture - 1)); Trans_elements++)
			{
				for (i = theta_start; i <= theta_end; i++)
				{
					temp = i - theta_start;
					delay_pointer[temp] = t_delay[Trans_elements - 1] + t_delay[i - 1];

					delay_pointer[temp] = delay_pointer[temp] - t1;
					delay_pointer[temp] = ((delay_pointer[temp] * fs) + 25.5) + zerolength;
					delay_int[temp] = (round)(delay_pointer[temp]);
					//	int abc = delay_int[temp];

				}

				for (i = theta_start; i <= theta_end; i++)
				{
					index_1 = (i - theta_start);
					row_v2 = delay_int[index_1];//此处计算有误！
					column_v2 = (Trans_elements - 1)*N_elements + i - 1;

					temp_1 = ((Trans_elements - emit_start) / emit_pitch)*M_elements + i - theta_start;
					//x_pointer[temp_1] = init_M1[row_v2*M + (Trans_elements - 1)*N_elements + i-1];//M为v2矩阵的行数
					temp_2 = column_v2*M + row_v2 - 1;
					x_pointer[temp_1] = init_M1[temp_2];//M为v2矩阵的行数

														//((Trans_elements - 1)*N_elements + i - 1)*M+row_v2
				}


			}
			//*************计算延时和回波矩阵完毕*********************************************************************//


			//*************接下来计算本像素点的beamDASDAS*******************************************//
			for (i = 0; i < M_elements; i++)
			{
				int j;
				for (j = 0; j < M_elements; j++)
				{
					complex<double> temp_1 = 0;
					temp_1 = x_pointer[i*M_elements + j] + beamDASDAS[(b - 1)*ysize + (a - 1)];
					beamDASDAS[(b - 1)*ysize + (a - 1)] = temp_1;
				}
			}
			beamDASDAS[(b - 1)*ysize + (a - 1)] = beamDASDAS[(b - 1)*ysize + (a - 1)] / ((double)(M_elements*M_elements));

			//*****************计算出了beamDASDAS*************************************************//

			//删除所用的动态内存
			delete[]x_pointer;
			delete[]delay_pointer;
			delete[]delay_int;


		}
		index_finished = 100.0 * a / ysize;
		printf("%.6lf\n", index_finished);

	}

	//************************************************主循环结束*********************************************************//

	/***********************打印结果**********************************************/
	//printf("The beamDASDAS matrix 501*121 is: \n");


	for (i = 0; i < 8; i++)
	{
		cout << "beamDASDAS[" << i << "] is: ";
		cout << beamDASDAS[i];
		/*cout << "verify is";
		cout << verify[i];*/
		cout << endl;
	}
	//*******************将beamDASDAS的结果保存到MAT文件，便于MATLAB进行图像生成***************************************//

	//pMF_out = matOpen("E:\\Project_files\\visual studio projects\\Ultrasound_GPU\\GPU_Data\\out_beamDASDAS.mat", "w");
	//cout << "已完成打开文件" << endl;
	////mwArray matrixComplex(row, column, mxDOUBLE_CLASS, mxCOMPLEX);//定义数组，行，列，double类型复数矩阵
	////pA = mxCreateStructMatrix(row, column, complex<double>);.
	//pA_out = mxCreateDoubleMatrix(xsize, ysize, mxCOMPLEX);//pA_out是mxArray类型的指针
	//cout << "已完成创建矩阵" << endl;

	////mxSetData(pA_out, out_beam);

	///*在早前版本的matlab自带的版本中使用的是mxSetData函数进行数值输入，
	//  但是在比较新的版本中，这个函数使用会出现错误，笔者也不知道为什么。
	//  所以在不确定的情况下，你可以使用memcpy；*/

	//memcpy((void*)mxGetData((pA_out)), beamDASDAS, sizeof(complex<double>) * xsize*ysize);
	////memcpy((complex<double>*)mxGetData((pA_out)), beamDASDAS, sizeof(complex<double>) * xsize*ysize);
	////使用mxGetData函数获取数据阵列中的数据;返回时需要使用强制类型转换。

	///*for (i = 0; i < 10; i++)
	//{
	//	cout << "pA_out is:"
	//		<< mxGetData(pA_out) << endl;
	//}*/

	//cout << "已完成copy数据" << endl;
	//matPutVariable(pMF_out, "beamDASDAS_GPU", pA_out);//如果matlab正在读该变量，则运行至此时会报错，只需在matlab里clear一下就OK了！
	//cout << "已完成放入变量" << endl;
	////matClose(pMF_out);

	/***************************************已得到正确beamDASDAS矩阵,接下来进行对数压缩************************/
	////先求beamDASDAS的绝对值
	//double *beam2 = new double[xsize*ysize];
	//double *beamshow = new double[xsize*ysize];
	//double peakall=0;
	//double peakall_2 = 0;
	//for (i = 0; i < xsize*ysize; i++)
	//{
	//	beam2[i] = abs(beamDASDAS[i]);
	//	if (beam2[i] >= peakall)
	//		peakall = beam2[i];
	//}

	//for (i = 0; i < xsize*ysize; i++)
	//{
	//	beamshow[i] = 20 * log10(beam2[i] / peakall);
	//	if (beamshow[i] < -60)
	//		beamshow[i] = -60;
	//}
	//for (i = 0; i < xsize*ysize; i++)
	//{
	//	beamshow[i] = 60 + beamshow[i];
	//}
	//for (i = 0; i < xsize*ysize; i++)
	//{
	//	if(peakall_2<=beamshow[i])
	//		peakall_2=beamshow[i];
	//}
	//
	//for (i = 0; i < xsize*ysize; i++)
	//{
	//	beamshow[i] = 255*(beamshow[i] / peakall_2);
	//	
	//}

	//for (i = 0; i < 8; i++)
	//{
	//	cout << "beamshow[" << i << "] is: ";
	//	cout << beamshow[i];
	//	/*cout << "verify is";
	//	cout << verify[i];*/
	//	cout << endl;
	//}
	//for (i = 0; i < xsize; i++)
	//{
	//	for (int j = 0; j < ysize; j++)
	//	{
	//		Result[i][j] = unsigned char((floor(beamshow[i * 121 + j])));
	//	}
	//}

	//Mat MM=Mat(501, 121, CV_8UC1);
	//memcpy(MM.data, Result, sizeof(unsigned char)*xsize*ysize);
	////Mat MM= Mat(501, 121, CV_32SC3, Result).clone();
	////namedWindow("Ultrasound_Image");
	//imshow("Ultrasound_Image", MM);
	////imshow("Ultrasound_Image", Result);

	delete[]beamDASDAS;
	delete[]init_M1;

	//printf("One Element is %e\n",init_M1[M*3]);
	//printf("One Element is %lf\n", init_M1[M * 3]);
	//printf("The no. of rows of Matrix M1 is %d\n", M);
	//printf("The no. of column of Matrix M1 is %d\n", N);

	getchar();
	getchar();
	return 0;

}


