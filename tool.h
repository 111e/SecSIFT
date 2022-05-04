#pragma once
#ifndef TOOL_H
#define TOOL_H
#include <opencv2/opencv.hpp>
//#include <string.h>
//#include <unistd.h>
//#include <errno.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include <sys/shm.h>
//#include "iostream"

using namespace cv;

class CSEM
{
private:
    union semun  // 用于信号灯操作的共同体。
    {
        int val;
        struct semid_ds* buf;
        unsigned short* arry;
    };

    int  sem_id;  // 信号灯描述符。
public:
    bool init(key_t key); // 如果信号灯已存在，获取信号灯；如果信号灯不存在，则创建信号灯并初始化。
    bool wait();          // 等待信号灯挂出。
    bool post();          // 挂出信号灯。
    bool destroy();       // 销毁信号灯。
};


void generate_random(double* a, double* a_1, double* a_2, double* b, double* b_1, double* b_2, double* c, double* c_1, double* c_2, double* t, double* t_1, double* t_2);
int sgn(double xy);
void MatToArray(Mat& img, double* array);
Mat ArrayToMat(double* array, int row, int col);




void secMul1_1(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1, double& x_1, double& y_1);
void secMul1_2(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1);
void secMul2_1(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3, double& a_2, double& b_2, double& c_2, double& x_2, double& y_2);
void secCmp1_1(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1, double& x_1, double& y_1, double& t_1);
void secCmp1_2(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1);
void secCmp1_3(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3);
void secCmp2_1(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3, double& a_2, double& b_2, double& c_2, double& x_2, double& y_2, double& t_2);
void secCmp2_2(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3);
void secAbsCmp1_1(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1, double& x_1);
void secAbsCmp1_2(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1, double& t_1, double& tresh);
void secAbsCmp1_3(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1);
void secAbsCmp1_4(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3);
void secAbsCmp2_1(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3, double& a_2, double& b_2, double& c_2, double& x_2);
void secAbsCmp2_2(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3, double& a_2, double& b_2, double& c_2, double& t_2);
void secAbsCmp2_3(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3);
void secMatMul1_1(int ii, double* ptext1, double* ptext2, double* A1, double* A2, double* A3, Mat& A_1, Mat& B_1, Mat& C_1, Mat& X_1, Mat& Y_1, int Xr, int Xc);
void secMatMul2_1(int ii, double* ptext1, double* ptext2, double* B1, double* B2, double* B3, Mat& A_2, Mat& B_2, Mat& C_2, Mat& X_2, Mat& Y_2, int Xr, int Xc);
void secMatMul1_2(int ii, double* ptext1, double* ptext2, double* A1, double* A2, double* A3, Mat& A_1, Mat& B_1, Mat& C_1, int Xr, int Xc);
void secMatInv1_1(int ii, double* ptext1, double* ptext2, double* A1, double* A2, double* A3, Mat& A_1, Mat& B_1, Mat& C_1, Mat& Z_1, Mat& X_1, int Xr, int Xc);
void secMatInv1_2(int ii, double* ptext1, double* ptext2, double* A1, double* A2, double* A3, Mat& A_1, Mat& B_1, Mat& C_1, int Xr, int Xc);
void secMatInv1_3(int ii, double* ptext1, double* ptext2, double* A1, double* A2, double* A3, Mat& A_1, Mat& B_1, Mat& C_1, Mat& Z_1, int Xr, int Xc);
void secMatInv2_1(int ii, double* ptext1, double* ptext2, double* B1, double* B2, double* B3, Mat& A_2, Mat& B_2, Mat& C_2, Mat& Z_2, Mat& X_2, int Xr, int Xc);
void secMatInv2_2(int ii, double* ptext1, double* ptext2, double* B1, double* B2, double* B3, Mat& A_2, Mat& B_2, Mat& C_2, Mat& Z_2, int Xr, int Xc);
void secAddRes1_1(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1, double& x_1, double& cc_1);
void secAddRes2_1(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3, double& a_2, double& b_2, double& c_2, double& x_2, double& cc_2);
void secAddRes1_2(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1, double& r_1);
void secAddRes2_2(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3, double& a_2, double& b_2, double& c_2, double& r_2);
void secPRE1_1(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1, double& x_1, double& cc_1);
void secPRE1_2(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1, double& r_1, double& power);
void secPRE1_3(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& cc_1);
void secPRE2_1(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3, double& a_2, double& b_2, double& c_2, double& x_2, double& cc_2);
void secPRE2_2(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3, double& a_2, double& b_2, double& c_2, double& r_2, double& power);
void secPRE2_3(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3, double& cc_2);
void secDiv1_1(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1, double& x_1, double& t_1);
void secDiv1_2(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1, double* tx_1, double& y_1, double& t_1);
void secDiv1_3(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double& a_1, double& b_1, double& c_1);
void secDiv1_4(int ii, double* ptext1, double* ptext2, double* a1, double* a2, double* a3, double* tx_1);
void secDiv2_1(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3, double& a_2, double& b_2, double& c_2, double* tx_2, double& x_2, double& t_2);
void secDiv2_2(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3, double& a_2, double& b_2, double& c_2, double& y_2, double& t_2);
void secDiv2_3(int ii, double* ptext1, double* ptext2, double* b1, double* b2, double* b3, double* tx_2);
#endif