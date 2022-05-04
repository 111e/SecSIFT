#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "tool.h"
#include <opencv2/opencv.hpp>
using namespace std;
//using namespace cv;

void secMul1_1(int ii,double* ptext1, double* ptext2, double* a1, double* a2, double* a3,double& a_1, double& b_1, double& c_1, double& x_1, double& y_1) {
	a2[ii] = x_1 - a_1;
	a3[ii] = y_1 - b_1;
	ptext1[ii] = a2[ii];
	ptext2[ii] = a3[ii]; 
}
void secMul1_2(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1) {
	a1[ii] = ptext1[ii] + a2[ii];//e
	a2[ii] = ptext2[ii] + a3[ii];//f
	a3[ii] = c_1+ b_1* a1[ii]+ a_1* a2[ii]+ a1[ii]* a2[ii];
	
}

void secMul2_1(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3, double& a_2, double& b_2, double& c_2, double& x_2, double& y_2) {
	b2[ii] = x_2 - a_2;//e_2
	b3[ii] = y_2 - b_2;//f_2
	
	b1[ii] = b2[ii] + ptext1[ii];//e
	ptext1[ii] = b2[ii];
	b2[ii] = b3[ii] + ptext2[ii];//f
	ptext2[ii] = b3[ii];
	
	b3[ii] = c_2 + b_2 * b1[ii] + a_2 * b2[ii];
	
}

void secCmp1_1(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1, double& x_1, double& y_1, double& t_1) {
	a1[ii] = x_1 - y_1;
	secMul1_1(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, t_1, a1[ii]);
}
void secCmp2_1(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3, double& a_2, double& b_2, double& c_2, double& x_2, double& y_2, double& t_2) {
	b1[ii] = x_2 - y_2;
	secMul2_1(ii, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, t_2, b1[ii]);
}
void secCmp1_2(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1) {
	secMul1_2(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
	ptext1[ii] = a3[ii];
	
}
void secCmp2_2(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3) {
	b1[ii] = sgn(ptext1[ii] + b3[ii]);//sgn_xy
	ptext1[ii] = b3[ii];
	
}
void secCmp1_3(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3) {
	a1[ii]= sgn(ptext1[ii] + a3[ii]);//sgn_xy
}
void secAbsCmp1_1(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1, double& x_1) {
	secMul1_1(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, x_1, x_1);
}
void secAbsCmp2_1(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3, double& a_2, double& b_2, double& c_2, double& x_2) {
	secMul2_1(ii, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, x_2, x_2);
}
void secAbsCmp1_2(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1, double& t_1, double& tresh) {
	secMul1_2(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);//x_x_1
	double thr = tresh * tresh;
	secCmp1_1(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, a3[ii], thr, t_1);
}
void secAbsCmp2_2(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3, double& a_2, double& b_2, double& c_2, double& t_2) {
	double thr = 0;
	secCmp2_1(ii, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, b3[ii], thr, t_2);
}
void secAbsCmp1_3(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1) {
	secCmp1_2(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
}
void secAbsCmp2_3(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3) {
	secCmp2_2(ii, ptext1, ptext2, b1, b2, b3);
}
void secAbsCmp1_4(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3) {
	secCmp1_3(ii, ptext1, ptext2, a1, a2, a3);
}
void secMatMul1_1(int ii, double* ptext1, double* ptext2, double* A1, double* A2, double* A3, Mat& A_1, Mat& B_1, Mat& C_1, Mat& X_1, Mat& Y_1, int Xr, int Xc) {
	int i, j;
	Mat E_1 = X_1 - A_1;
	Mat F_1 = Y_1 - B_1;
	for (i = 0; i < Xr; i++) {
		for (j = 0; j < Xc; j++) {
			*(A2 + Xr * Xc * ii + i * Xr + j) = E_1.at<double>(i, j);
			*(A3 + Xr * Xc * ii + i * Xr + j) = F_1.at<double>(i, j);
			*(ptext1 + Xr * Xc * ii + i * Xr + j) = *(A2 + Xr * Xc * ii + i * Xr + j);
			*(ptext2 + Xr * Xc * ii + i * Xr + j) = *(A3 + Xr * Xc * ii + i * Xr + j);
			//cout << "ii" << ii  << " F_1:" << *(A3 + Xr * Xc * ii + i * Xr + j) << endl;
		}
	}
}

void secMatMul2_1(int ii, double* ptext1, double* ptext2, double* B1, double* B2, double* B3, Mat& A_2, Mat& B_2, Mat& C_2, Mat& X_2, Mat& Y_2, int Xr, int Xc) {
	Mat XY_2;
	Mat E(Xr, Xc, CV_64FC1);
	Mat F(Xr, Xc, CV_64FC1);
	int i, j;
	Mat E_2 = X_2 - A_2;
	Mat F_2 = Y_2 - B_2;
	for (i = 0; i < Xr; i++) {
		for (j = 0; j < Xc; j++) {
			*(B2 + Xr * Xc * ii + i * Xr + j) = E_2.at<double>(i, j);
			*(B3 + Xr * Xc * ii + i * Xr + j) = F_2.at<double>(i, j);
			*(B1 + Xr * Xc * ii + i * Xr + j) = E_2.at<double>(i, j) + *(ptext1 + Xr * Xc * ii + i * Xr + j);
			E.at<double>(i, j) = *(B1 + Xr * Xc * ii + i * Xr + j);
			//cout << "ii" << ii << " E:" << E.at<double>(i, j) << endl;
			*(ptext1 + Xr * Xc * ii + i * Xr + j) = E_2.at<double>(i, j);
			*(B2 + Xr * Xc * ii + i * Xr + j) = F_2.at<double>(i, j) + *(ptext2 + Xr * Xc * ii + i * Xr + j);
			F.at<double>(i, j) = *(B2 + Xr * Xc * ii + i * Xr + j);
			//cout << "ii" << ii << " F:" << F.at<double>(i, j) << endl;
			*(ptext2 + Xr * Xc * ii + i * Xr + j) = F_2.at<double>(i, j);
			//cout << "ii" << Xr * Xc * ii + i * Xr + j << " F:" << *(B2 + Xr * Xc * ii + i * Xr + j) << endl;
			
		}
	}
	XY_2 = E * B_2 + A_2 * F + C_2;
	for (i = 0; i < Xr; i++) {
		for (j = 0; j < Xc; j++) {
			*(B3 + Xr * Xc * ii + i * Xr + j) = XY_2.at<double>(i, j);
			//cout << "ii" << ii  << " XY_2:" << *(B3 + Xr * Xc * ii + i * Xr + j) << endl;
		}
	}
}

void secMatMul1_2(int ii, double* ptext1, double* ptext2, double* A1, double* A2, double* A3, Mat& A_1, Mat& B_1, Mat& C_1, int Xr, int Xc) {
	Mat XY_1;
	Mat E(Xr, Xc, CV_64FC1);
	Mat F(Xr, Xc, CV_64FC1);
	int i, j;
	for (i = 0; i < Xr; i++) {
		for (j = 0; j < Xc; j++) {
			//cout << "ii" << Xr * Xc * ii + i * Xr + j << " F_2:" << *(ptext2 + Xr * Xc * ii + i * Xr + j) << endl;
			E.at<double>(i, j) = *(A2 + Xr * Xc * ii + i * Xr + j) + *(ptext1 + Xr * Xc * ii + i * Xr + j);
			F.at<double>(i, j) = *(A3 + Xr * Xc * ii + i * Xr + j) + *(ptext2 + Xr * Xc * ii + i * Xr + j);
			//cout << "ii" << ii << " F:" << F.at<double>(i, j) << endl;
		}
	}
	XY_1 = E * B_1 + A_1 * F + E * F + C_1;
	for (i = 0; i < Xr; i++) {
		for (j = 0; j < Xc; j++) {
			*(A3 + Xr * Xc * ii + i * Xr + j) = XY_1.at<double>(i, j);
			//cout << "ii" << ii << " XY_1:" << *(A3 + Xr * Xc * ii + i * Xr + j) << endl;
		}
	}
}

void secMatInv1_1(int ii, double* ptext1, double* ptext2, double* A1, double* A2, double* A3, Mat& A_1, Mat& B_1, Mat& C_1, Mat& Z_1, Mat& X_1, int Xr, int Xc) {
	secMatMul1_1(ii, ptext1, ptext2, A1, A2, A3, A_1, B_1, C_1, Z_1, X_1, Xr, Xc);
	
}
void secMatInv2_1(int ii, double* ptext1, double* ptext2, double* B1, double* B2, double* B3, Mat& A_2, Mat& B_2, Mat& C_2, Mat& Z_2, Mat& X_2, int Xr, int Xc) {
	secMatMul2_1(ii, ptext1, ptext2, B1, B2, B3, A_2, B_2, C_2, Z_2, X_2, Xr, Xc);//ZX_2
	
}
void secMatInv1_2(int ii, double* ptext1, double* ptext2, double* A1, double* A2, double* A3, Mat& A_1, Mat& B_1, Mat& C_1, int Xr, int Xc) {
	secMatMul1_2(ii, ptext1, ptext2, A1, A2, A3, A_1, B_1, C_1, Xr, Xc);
	for (int i = 0; i < Xr; i++) {
		for (int j = 0; j < Xc; j++) {
			*(ptext1 + Xr * Xc * ii + i * Xr + j) = *(A3 + Xr * Xc * ii + i * Xr + j);
			//cout << "*(A3 + Xr * Xc * ii + i * Xr + j)------:" << *(A3 + Xr * Xc * ii + i * Xr + j) << endl;
		}
	}
}
void secMatInv2_2(int ii, double* ptext1, double* ptext2, double* B1, double* B2, double* B3, Mat& A_2, Mat& B_2, Mat& C_2, Mat& Z_2, int Xr, int Xc) {
	int n = 0;
	Mat ZX(Xr, Xc, CV_64FC1);
	Mat nX_2(Xr, Xc, CV_64FC1);
	for (int i = 0; i < Xr; i++) {
		for (int j = 0; j < Xc; j++) {
			*(B1 + Xr * Xc * ii + i * Xr + j) = *(B3 + Xr * Xc * ii + i * Xr + j) + *(ptext1 + Xr * Xc * ii + i * Xr + j);
			ZX.at<double>(i, j) = *(B1 + Xr * Xc * ii + i * Xr + j);
			*(ptext1 + Xr * Xc * ii + i * Xr + j) = *(B3 + Xr * Xc * ii + i * Xr + j);
		}
	}
	if (fabs(determinant(ZX)) < 1e-20) {
		for (int i = 0; i < Xr; i++) {
			for (int j = 0; j < Xc; j++) {
				nX_2.at<double>(i, j) = 0;
				*(B2 + Xr * Xc * ii + i * Xr + j) = nX_2.at<double>(i, j);
				//cout << "ii" << ii << " nX_2:" << *(B2 + Xr * Xc * ii + i * Xr + j) << endl;
			}
		}
	}
	else {
		invert(ZX, nX_2);
		nX_2 = nX_2 * Z_2;
		//cout << "nX_2" << nX_2 << endl;
		for (int i = 0; i < Xr; i++) {
			for (int j = 0; j < Xc; j++) {
				*(B2 + Xr * Xc * ii + i * Xr + j) = nX_2.at<double>(i, j);
				//cout << "ii" << ii << " nX_2:" << *(B2 + Xr * Xc * ii + i * Xr + j) << endl;
			}
		}
	}
}
void secMatInv1_3(int ii, double* ptext1, double* ptext2, double* A1, double* A2, double* A3, Mat& A_1, Mat& B_1, Mat& C_1, Mat& Z_1, int Xr, int Xc) {
	Mat ZX(Xr, Xc, CV_64FC1);
	Mat nX_1(Xr, Xc, CV_64FC1);
	for (int i = 0; i < Xr; i++) {
		for (int j = 0; j < Xc; j++) {
			ZX.at<double>(i, j) = *(A3 + Xr * Xc * ii + i * Xr + j) + *(ptext1 + Xr * Xc * ii + i * Xr + j);
			*(A1 + Xr * Xc * ii + i * Xr + j) = ZX.at<double>(i, j);
		}
	}
	if (fabs(determinant(ZX)) < 1e-20) {
		for (int i = 0; i < Xr; i++) {
			for (int j = 0; j < Xc; j++) {
				nX_1.at<double>(i, j) = 0;
				*(A2 + Xr * Xc * ii + i * Xr + j) = nX_1.at<double>(i, j);
				//cout << "ii" << ii << " nX_1:" << *(A2 + Xr * Xc * ii + i * Xr + j) << endl;
			}
		}
	}
	else {
		invert(ZX, nX_1);
		nX_1 = nX_1 * Z_1;
		for (int i = 0; i < Xr; i++) {
			for (int j = 0; j < Xc; j++) {
				*(A2 + Xr * Xc * ii + i * Xr + j) = nX_1.at<double>(i, j);
				//cout << "ii" << ii << " nX_1:" << *(A2 + Xr * Xc * ii + i * Xr + j) << endl;
			}
		}
	}
}

void secAddRes1_1(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1, double& x_1, double& cc_1) {
	
	
	secMul1_1(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, x_1, cc_1);
	
}
void secAddRes2_1(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3, double& a_2, double& b_2, double& c_2, double& x_2, double& cc_2) {
	secMul2_1(ii, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, x_2, cc_2);
	
}
void secAddRes1_2(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1, double& r_1) {
	secMul1_2(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
	ptext1[ii] = a3[ii];
	a1[ii] = 1 / r_1;
}
void secAddRes2_2(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3, double& a_2, double& b_2, double& c_2, double& r_2) {
	b1[ii] = (ptext1[ii] + b3[ii]) / r_2;
	
}
void secPRE1_1(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1, double& x_1, double& cc_1) {
	secAddRes1_1(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, x_1, cc_1);
}
void secPRE2_1(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3, double& a_2, double& b_2, double& c_2, double& x_2, double& cc_2) {
	secAddRes2_1(ii, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, x_2, cc_2);
	
}
void secPRE1_2(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1, double& r_1, double& power) {
	
	secAddRes1_2(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, r_1);
	
	a2[ii] = pow(fabs(a1[ii]), power);
	
	a3[ii] = a2[ii] / r_1;
	//cout << "a3[ii]:" << a3[ii] << endl;
}
void secPRE2_2(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3, double& a_2, double& b_2, double& c_2, double& r_2, double& power) {
	secAddRes2_2(ii, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, r_2);
	b2[ii]= pow(fabs(b1[ii]), power);
	b3[ii] = b2[ii] / r_2;
	//cout << "b3[ii]:" << b3[ii] << endl;
	ptext1[ii] = b3[ii];
}
void secPRE1_3(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& cc_1) {
	a1[ii] = a3[ii] * ptext1[ii] * cc_1;
	//cout<< a3[ii] * ptext1[ii] <<endl;
	ptext1[ii] = a3[ii];
}
void secPRE2_3(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3, double& cc_2) {
	b1[ii] = ptext1[ii] * b3[ii] * cc_2;
	//cout <<"---"<< b3[ii] * ptext1[ii] << endl;
}
void secDiv1_1(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1, double& x_1, double& t_1) {
	secMul1_1(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, t_1, x_1);
}
void secDiv2_1(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3, double& a_2, double& b_2, double& c_2, double* tx_2, double& x_2, double& t_2) {
	secMul2_1(ii, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, t_2, x_2);
	tx_2[ii] = b3[ii];
}
void secDiv1_2(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1, double* tx_1, double& y_1, double& t_1) {
	secMul1_2(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
	tx_1[ii] = a3[ii];
	secMul1_1(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, t_1, y_1);
}
void secDiv2_2(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3, double& a_2, double& b_2, double& c_2, double& y_2, double& t_2) {
	secMul2_1(ii, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, t_2, y_2);
}
void secDiv1_3(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1) {
	secMul1_2(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
	ptext1[ii] = a3[ii];
}
void secDiv2_3(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3, double* tx_2) {
	b1[ii] = tx_2[ii] / (ptext1[ii] + b3[ii]);
	ptext1[ii] = b3[ii];
}
void secDiv1_4(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double* tx_1) {
	a1[ii] = tx_1[ii] / (a3[ii] + ptext1[ii]);
}
