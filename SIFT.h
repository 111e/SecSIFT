#pragma once
#ifndef SIFT_H
#define SIFT_H

#include "tool.h"
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"



using namespace cv;
using namespace std;


typedef double pixel_t;                             //【1】像素类型
// assumed gaussian blur for input image
#define INIT_SIGMA 0.5                               //【2】初始sigma
#define SIGMA 1.6
#define INTERVALS 3                                  //【3】高斯金字塔中每组图像中有三层/张图片

#define RATIO 10                                     //【4】半径r
#define MAX_INTERPOLATION_STEPS 5                    //【5】最大空间间隔
#define DXTHRESHOLD 0.04                             //【6】|D(x^)| < 0.04   

#define ORI_HIST_BINS 36                             //【7】bings=36

// determines gaussian sigma for orientation assignment
#define ORI_SIGMA_TIMES 1.5
// determines the radius of the region used in orientation assignment
#define ORI_WINDOW_RADIUS 3.0 * ORI_SIGMA_TIMES
#define ORI_SMOOTH_TIMES 2
// orientation magnitude relative to max that results in new feature
#define ORI_PEAK_RATIO 0.8
//描述符维数
#define FEATURE_ELEMENT_LENGTH 128
// default number of bins per histogram in descriptor array
#define DESCR_HIST_BINS 8
// width of border in which to ignore keypoints
#define IMG_BORDER 5
#define DESCR_WINDOW_WIDTH 4
#define DESCR_SCALE_ADJUST 3
// threshold on magnitude of elements of descriptor vector
#define DESCR_MAG_THR 0.2
// factor used to convert floating-point descriptor to unsigned char
#define INT_DESCR_FCTR 512.0

//边缘效应 r
#define EdgeThreshold 10

#define SIFT_FIXPT_SCALE 1
/*********************************************************************************************
*模块说明：
*        关键点/特征点的结构体声明
*注意点1：
*        在高斯金字塔构建的过程中，一幅图像可以产生好几组(octave)图像，一组图像包含几层(inteval)
*        图像
*********************************************************************************************/
struct Keypoint
{
	int    octave;                                        //【1】关键点所在组
	int    interval;                                      //【2】关键点所在层
	double offset_interval;                               //【3】调整后的层的增量

	int    x;                                             //【4】x,y坐标,根据octave和interval可取的层内图像
	int    y;
	double scale;                                         //【5】空间尺度坐标scale = sigma0*pow(2.0, o+s/S)

	double dx;                                            //【6】特征点坐标，该坐标被缩放成原图像大小
	double dy;

	double offset_x;
	double offset_y;

	//============================================================
	//1---高斯金字塔组内各层尺度坐标，不同组的相同层的sigma值相同
	//2---关键点所在组的组内尺度
	//============================================================
	double octave_scale;                                  //【1】offset_i;
	double ori;                                           //【2】方向
	int    descr_length;
	double descriptor[FEATURE_ELEMENT_LENGTH];            //【3】特征点描述符           
	double val;                                           //【4】极值
};

void CreateInitSmoothGray(const Mat& src, Mat& dst, double sigma = SIGMA);
void GaussianPyramid(const Mat& src, vector<Mat>& gauss_pyr, int octaves, int intervals = INTERVALS, double sigma = SIGMA);
void DogPyramid(const vector<Mat>& gauss_pyr, vector<Mat>& dog_pyr, int octaves, int intervals = INTERVALS);
void DetectionLocalExtrema1(CSEM sem, double* ptext1, double* ptext2, const vector<Mat>& dog_pyr1, vector<Keypoint>& extrema1, int octaves, double a_1, double b_1, double c_1, double t_1, Mat& A_1, Mat& B_1, Mat& C_1, Mat& Z_1, int intervals = INTERVALS);
void DetectionLocalExtrema2(CSEM sem, double* ptext1, double* ptext2, const vector<Mat>& dog_pyr2, vector<Keypoint>& extrema2, int octaves, double a_2, double b_2, double c_2, double t_2, Mat& A_2, Mat& B_2, Mat& C_2, Mat& Z_2, int intervals = INTERVALS);
void OrientationAssignment1(CSEM sem, double* ptext1, double* ptext2, vector<Keypoint>& extrema1, vector<Keypoint>& features1, const vector<Mat>& gauss_pyr1, double a_1, double b_1, double c_1, double t_1, double cc_1, double r_1);
void OrientationAssignment2(CSEM sem, double* ptext1, double* ptext2, vector<Keypoint>& extrema2, vector<Keypoint>& features2, const vector<Mat>& gauss_pyr2, double a_2, double b_2, double c_2, double t_2, double cc_2, double r_2);
void DescriptorRepresentation1(CSEM sem, double* ptext1, double* ptext2, vector<Keypoint>& features1, const vector<Mat>& gauss_pyr1, double a_1, double b_1, double c_1, double t_1, double cc_1, double r_1, int bins = DESCR_HIST_BINS, int width = DESCR_WINDOW_WIDTH);
void DescriptorRepresentation2(CSEM sem, double* ptext1, double* ptext2, vector<Keypoint>& features2, const vector<Mat>& gauss_pyr2, double a_2, double b_2, double c_2, double t_2, double cc_2, double r_2, int bins = DESCR_HIST_BINS, int width = DESCR_WINDOW_WIDTH);
void Sift1(CSEM sem, double* ptext1, double* ptext2, const Mat& src1, vector<Keypoint>& features1, double* a_1, double* b_1, double* c_1, double* t_1, double cc_1, double r_1, Mat& A_1, Mat& B_1, Mat& C_1, Mat& Z_1, double sigma = SIGMA, int intervals = INTERVALS);
void Sift2(CSEM sem, double* ptext1, double* ptext2, const Mat& src2, vector<Keypoint>& features2, double* a_2, double* b_2, double* c_2, double* t_2, double cc_2, double r_2, Mat& A_2, Mat& B_2, Mat& C_2, Mat& Z_2, double sigma = SIGMA, int intervals = INTERVALS);
void CalculateScale(vector<Keypoint>& features, double sigma, int intervals);
void HalfFeatures(vector<Keypoint>& features);
void write_features(const vector<Keypoint>& features, const char* file);
void InterpHistEntry(int ii, double* hist, double xbin, double ybin, double obin, double mag, int bins, int d);
#endif
