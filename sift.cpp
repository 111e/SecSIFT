#include"SIFT.h"
#include "tool.h"
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
using namespace std;
using namespace cv;

#define HIat(i, j) (*(H_inve+(i)*3 + (j)))
#define Hat(i, j) (*(H+(i)*3 + (j)))
#define DAt(x, y) (*(data+(y)*step+(x)))


void Sift1(CSEM sem, double* ptext1, double* ptext2, const Mat& src1, vector<Keypoint>& features1, double* a_1, double* b_1, double* c_1, double* t_1, double cc_1, double r_1, Mat& A_1, Mat& B_1, Mat& C_1, Mat& Z_1, double sigma, int intervals) {
	time_t now_time_1 = time(NULL);

	std::cout << "[Step_one]Create -1 octave gaussian pyramid image" << std::endl;

	std::cout << "sigma_1      = " << sigma << std::endl;
	std::cout << "intervals_1     = " << intervals << std::endl;
	cv::Mat init_gray1;

	CreateInitSmoothGray(src1, init_gray1, sigma);

	
	int octaves = log((double)min(init_gray1.rows, init_gray1.cols)) / log(2.0) - 2;             //计算高斯金字塔的层数
	std::cout << "[1]The height and width of init_gray_img = " << init_gray1.rows << "*" << init_gray1.cols << std::endl;
	std::cout << "[2]The octaves of the gauss pyramid      = " << octaves << std::endl;
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("1A  sem.wait ok\n");
	std::cout << "[Step_two]Building gaussian pyramid ..." << std::endl;
	std::vector<Mat> gauss_pyr1;
	GaussianPyramid(init_gray1, gauss_pyr1, octaves, intervals, sigma);    //高斯金字塔
	
	std::cout << "[Step_three]Building difference of gaussian pyramid..." << std::endl;
	vector<Mat> dog_pyr1;
	DogPyramid(gauss_pyr1, dog_pyr1, octaves, intervals);     //差分金字塔
	
	std::cout << "[Step_four]Detecting local extrema..." << std::endl;
	vector<Keypoint> extrema1;
	
	DetectionLocalExtrema1(sem, ptext1, ptext2, dog_pyr1, extrema1, octaves, a_1[0], b_1[0], c_1[0], t_1[0], A_1, B_1, C_1, Z_1, intervals);   //极值点初探（排除较小点，找到极值点，消除边缘响应）
	std::cout << "【3】keypoints cout: " << extrema1.size() << " --" << std::endl;
	std::cout << "【4】extrema detection finished." << std::endl;
	std::cout << "【5】please look dir gausspyramid, dogpyramid and extrema.txt.--" << endl;

	std::cout << "【Step_five】CalculateScale..." << std::endl;
	CalculateScale(extrema1, sigma, intervals);   //计算特征点的尺度
	HalfFeatures(extrema1);            //图像缩放

	std::cout << "【Step_six】Orientation assignment..." << std::endl;
	OrientationAssignment1(sem, ptext1, ptext2, extrema1, features1, gauss_pyr1, a_1[1], b_1[1], c_1[1], t_1[1], cc_1, r_1);
	std::cout << "【6】features count: " << features1.size() << std::endl;

	std::cout << "【Step_seven】DescriptorRepresentation..." << std::endl;
	DescriptorRepresentation1(sem, ptext1, ptext2, features1, gauss_pyr1, a_1[2], b_1[2], c_1[2], t_1[2], cc_1, r_1);
	time_t now_time_2 = time(NULL);
	cout << "time_1 " << now_time_2 - now_time_1 << endl;
}

void Sift2(CSEM sem, double* ptext1, double* ptext2, const Mat& src2, vector<Keypoint>& features2, double* a_2, double* b_2, double* c_2, double* t_2, double cc_2, double r_2, Mat& A_2, Mat& B_2, Mat& C_2, Mat& Z_2,  double sigma, int intervals) {
	std::cout << "[Step_one]Create -1 octave gaussian pyramid image" << std::endl;

	std::cout << "sigma_2      = " << sigma << std::endl;
	std::cout << "intervals_2     = " << intervals << std::endl;
	cv::Mat init_gray2;

	CreateInitSmoothGray(src2, init_gray2, sigma);

	int octaves = log((double)min(init_gray2.rows, init_gray2.cols)) / log(2.0) - 2;             //计算高斯金字塔的层数
	std::cout << "[1]The height and width of init_gray_img = " << init_gray2.rows << "*" << init_gray2.cols << std::endl;
	std::cout << "[2]The octaves of the gauss pyramid      = " << octaves << std::endl;

	std::cout << "[Step_two]Building gaussian pyramid ..." << std::endl;
	std::vector<Mat> gauss_pyr2;
	GaussianPyramid(init_gray2, gauss_pyr2, octaves, intervals, sigma);    //高斯金字塔
	

	std::cout << "[Step_three]Building difference of gaussian pyramid..." << std::endl;
	vector<Mat> dog_pyr2;
	DogPyramid(gauss_pyr2, dog_pyr2, octaves, intervals);     //差分金字塔
	std::cout << "[Step_four]Detecting local extrema..." << std::endl;
	vector<Keypoint> extrema2;
	
	DetectionLocalExtrema2(sem, ptext1, ptext2, dog_pyr2, extrema2, octaves, a_2[0], b_2[0], c_2[0], t_2[0], A_2, B_2, C_2, Z_2, intervals);   //极值点初探（排除较小点，找到极值点，消除边缘响应）
	std::cout << "【3】keypoints cout: " << extrema2.size() << " --" << std::endl;
	std::cout << "【4】extrema detection finished." << std::endl;
	std::cout << "【5】please look dir gausspyramid, dogpyramid and extrema.txt.--" << endl;
	
	std::cout << "【Step_five】CalculateScale..." << std::endl;
	CalculateScale(extrema2, sigma, intervals);   //计算特征点的尺度
	HalfFeatures(extrema2);            //图像缩放

	std::cout << "【Step_six】Orientation assignment..." << std::endl;
	OrientationAssignment2(sem, ptext1, ptext2, extrema2, features2, gauss_pyr2, a_2[1], b_2[1], c_2[1], t_2[1], cc_2, r_2);
	std::cout << "【6】features count: " << features2.size() << std::endl;

	std::cout << "【Step_seven】DescriptorRepresentation..." << std::endl;
	DescriptorRepresentation2(sem, ptext1, ptext2, features2, gauss_pyr2, a_2[2], b_2[2], c_2[2], t_2[2], cc_2, r_2);

}
void threshing1(CSEM sem, double* ptext1, double* ptext2, int* DetectionLocalExtrema1, double thresh, const vector<Mat>& dog_pyr1, int octaves, double a_1, double b_1, double c_1, double t_1, int intervals = INTERVALS) {
	double a1[4000000] = { 0 };
	double a2[4000000] = { 0 };
	double a3[4000000] = { 0 };
	int ii = 0;
	for (int o = 0; o < octaves; o++) {
		//(intervals + 2) - 1
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			pixel_t* data = (pixel_t*)dog_pyr1[index].data;
			int step = dog_pyr1[index].cols;
			//dog_pyr1[index].rows - IMG_BORDER  dog_pyr1[index].cols - IMG_BORDE
			for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
					pixel_t val = *(data + y * step + x);
					DetectionLocalExtrema1[ii] = 1;
					secAbsCmp1_1(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, val);
					
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
					secAbsCmp1_2(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, t_1, thresh);
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
					secAbsCmp1_3(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
					secAbsCmp1_4(ii, ptext1, ptext2, a1, a2, a3);
					if (a1[ii] == -1 || a1[ii] == 0) DetectionLocalExtrema1[ii] = 0;
					ii++;
				}
			}
		}
	}
}
void threshing2(CSEM sem, double* ptext1, double* ptext2, int* DetectionLocalExtrema2, double thresh, const vector<Mat>& dog_pyr2, int octaves, double a_2, double b_2, double c_2, double t_2, int intervals = INTERVALS) {
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");

	double b1[4000000] = { 0 };
	double b2[4000000] = { 0 };
	double b3[4000000] = { 0 };
	int ii = 0;
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			pixel_t* data = (pixel_t*)dog_pyr2[index].data;
			int step = dog_pyr2[index].cols;
			for (int y = IMG_BORDER; y < dog_pyr2[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr2[index].cols - IMG_BORDER; x++) {
					pixel_t val = *(data + y * step + x);
					//val = round(val * 1000000) / 1000000;
					DetectionLocalExtrema2[ii] = 1;
					secAbsCmp2_1(ii, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, val);
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr2[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr2[index].cols - IMG_BORDER; x++) {
					secAbsCmp2_2(ii, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, t_2);
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr2[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr2[index].cols - IMG_BORDER; x++) {
					secAbsCmp2_3(ii, ptext1, ptext2, b1, b2, b3);
					if (b1[ii] == -1 || b1[ii] == 0) DetectionLocalExtrema2[ii] = 0;
					ptext1[ii] = b3[ii];
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
}
void isExtremum1(CSEM sem, double* ptext1, double* ptext2, int* DetectionLocalExtrema1, const vector<Mat>& dog_pyr1, int octaves, double a_1, double b_1, double c_1, double t_1, int intervals = INTERVALS){
	double a1[4000000] = { 0 };
	double a2[4000000] = { 0 };
	double a3[4000000] = { 0 };
	double a[4000000] = { 0 };
	double y_1 = 0;
	int flag = 1;
	int ii = 0;
	
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			pixel_t* data = (pixel_t*)dog_pyr1[index].data;
			int step = dog_pyr1[index].cols;
			for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
					pixel_t val = *(data + y * step + x);
					secCmp1_1(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, val, y_1, t_1);
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
					secCmp1_2(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
					secCmp1_3(ii, ptext1, ptext2, a1, a2, a3);
					a[ii] = a1[ii];
					ii++;
				}
			}
		}
	}
	double a4[4000000 * 27];
	double a5[4000000 * 27];
	double a6[4000000 * 27];
	
	ii = 0;
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			pixel_t* data = (pixel_t*)dog_pyr1[index].data;
			int step = dog_pyr1[index].cols;
			for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
					pixel_t val = *(data + y * step + x);
					for (int k = -1; k <= 1; k++) {
						int stp = dog_pyr1[index + k].cols;
						for (int l = -1; l <= 1; l++) {
							for (int m = -1; m <= 1; m++) {
								secCmp1_1(27 * ii + 9 * (k + 1) + 3 * (l + 1) + m + 1, ptext1, ptext2, a4, a5, a6, a_1, b_1, c_1, val, *((pixel_t*)dog_pyr1[index + k].data + stp * (y + l) + (x + m)), t_1);
								//cout << "o:" << 27 * ii + 9 * (k + 1) + 3 * (l + 1) + m + 1 << endl;
							}
						}
					}
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			pixel_t* data = (pixel_t*)dog_pyr1[index].data;
			int step = dog_pyr1[index].cols;
			for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
					for (int k = -1; k <= 1; k++) {
						int stp = dog_pyr1[index + k].cols;
						for (int l = -1; l <= 1; l++) {
							for (int m = -1; m <= 1; m++) {
								secCmp1_2(27 * ii + 9 * (k + 1) + 3 * (l + 1) + m + 1, ptext1, ptext2, a4, a5, a6, a_1, b_1, c_1);
							}
						}
					}
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			pixel_t* data = (pixel_t*)dog_pyr1[index].data;
			int step = dog_pyr1[index].cols;
			for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
					for (int k = -1; k <= 1; k++) {
						int stp = dog_pyr1[index + k].cols;
						for (int l = -1; l <= 1; l++) {
							for (int m = -1; m <= 1; m++) {
								secCmp1_3(27 * ii + 9 * (k + 1) + 3 * (l + 1) + m + 1, ptext1, ptext2, a4, a5, a6);
								if (a[ii] == 1 && a4[27 * ii + 9 * (k + 1) + 3 * (l + 1) + m + 1] == -1) {
									DetectionLocalExtrema1[ii] = 0;  flag = 0;  break;
								}
								else if (a[ii] != 1 && a4[27 * ii + 9 * (k + 1) + 3 * (l + 1) + m + 1] == 1) {
									DetectionLocalExtrema1[ii] = 0; flag = 0;  break;
								}
							}
							if(flag == 0) break;
						}
						if (flag == 0) break;
					}
					flag = 1;
					ii++;
				}
			}
		}
	}
	ii = 0;
}
void isExtremum2(CSEM sem, double* ptext1, double* ptext2, int* DetectionLocalExtrema2, const vector<Mat>& dog_pyr2, int octaves, double a_2, double b_2, double c_2, double t_2, int intervals = INTERVALS) {
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	double b1[4000000] = { 0 };
	double b2[4000000] = { 0 };
	double b3[4000000] = { 0 };
	double b[4000000] = { 0 };
	int flag = 1;
	double y_2 = 0;
	int ii = 0;
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			pixel_t* data = (pixel_t*)dog_pyr2[index].data;
			int step = dog_pyr2[index].cols;
			for (int y = IMG_BORDER; y < dog_pyr2[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr2[index].cols - IMG_BORDER; x++) {
					pixel_t val = *(data + y * step + x);
					secCmp2_1(ii, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, val, y_2, t_2);
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr2[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr2[index].cols - IMG_BORDER; x++) {
					secCmp2_2(ii, ptext1, ptext2, b1, b2, b3);
					b[ii] = b1[ii];
					ii++;
				}
			}
		}
		
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	double b4[4000000 * 27];
	double b5[4000000 * 27];
	double b6[4000000 * 27];
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			pixel_t* data = (pixel_t*)dog_pyr2[index].data;
			int step = dog_pyr2[index].cols;
			for (int y = IMG_BORDER; y < dog_pyr2[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr2[index].cols - IMG_BORDER; x++) {
					pixel_t val = *(data + y * step + x);
					for (int k = -1; k <= 1; k++) {
						int stp = dog_pyr2[index + k].cols;
						for (int l = -1; l <= 1; l++) {
							for (int m = -1; m <= 1; m++) {
								secCmp2_1(27 * ii + 9 * (k + 1) + 3 * (l + 1) + m + 1, ptext1, ptext2, b4, b5, b6, a_2, b_2, c_2, val, *((pixel_t*)dog_pyr2[index + k].data + stp * (y + l) + (x + m)), t_2);
							}
						}
					}
					ii++;
				}
			}
		}
		
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			pixel_t* data = (pixel_t*)dog_pyr2[index].data;
			int step = dog_pyr2[index].cols;
			for (int y = IMG_BORDER; y < dog_pyr2[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr2[index].cols - IMG_BORDER; x++) {
					for (int k = -1; k <= 1; k++) {
						int stp = dog_pyr2[index + k].cols;
						for (int l = -1; l <= 1; l++) {
							for (int m = -1; m <= 1; m++) {
								secCmp2_2(27 * ii + 9 * (k + 1) + 3 * (l + 1) + m + 1, ptext1, ptext2, b4, b5, b6);
								if (b[ii] == 1 && b4[27 * ii + 9 * (k + 1) + 3 * (l + 1) + m + 1] == -1) {
									DetectionLocalExtrema2[ii] = 0; flag = 0;  break;
								}
								else if (b[ii] != 1 && b4[27 * ii + 9 * (k + 1) + 3 * (l + 1) + m + 1] == 1) {
									DetectionLocalExtrema2[ii] = 0; flag = 0;  break;
								}
							}
							if (flag == 0) break;
						}
						if (flag == 0) break;
					}
					flag = 1;
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
}
void InterploationExtremum1(CSEM sem, double* ptext1, double* ptext2, int* DetectionLocalExtrema1, Keypoint* keypoint, const vector<Mat>& dog_pyr1, int octaves, double a_1, double b_1, double c_1, double t_1, Mat& A_1, Mat& B_1, Mat& C_1, Mat& Z_1, int intervals = INTERVALS, double dxthreshold = DXTHRESHOLD) {
	const double img_scale = 1.0 / (255 * 1);//;
	const double deriv_scale = img_scale * 0.5;
	const double second_deriv_scale = img_scale;
	const double cross_deriv_scale = img_scale * 0.25;
	double offset_x[4000000 * 3];
	double dx[4000000 * 3];
	double A1[4000000 * 3 * 3];
	double A2[4000000 * 3 * 3];
	double A3[4000000 * 3 * 3];
	double a1[4000000 * 3 * 3];
	double a2[4000000 * 3 * 3];
	double a3[4000000 * 3 * 3];
	int idx[4000000];
	int x_A[4000000];
	int y_A[4000000];
	int interval_A[4000000];
	int flag[4000000];
	double term[4000000];
	int ii = 0;

	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			pixel_t* data = (pixel_t*)dog_pyr1[index].data;
			int step = dog_pyr1[index].cols;
			for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema1[ii] == 1) {
						x_A[ii] = x;
						y_A[ii] = y;
						interval_A[ii] = i;
						flag[ii] = 1;
						idx[ii] = index;
					}
					ii++;
				}
			}
		}
	}
	ii = 0;
	for (int l = 0; l < MAX_INTERPOLATION_STEPS; l++) {
		for (int o = 0; o < octaves; o++) {
			for (int i = 1; i < (intervals + 2) - 1; i++) {
				int index = o * (intervals + 2) + i;
				for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
					for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
						if (DetectionLocalExtrema1[ii] == 1) {
							if (flag[ii] == 0){ }
							else {
								const Mat& img = dog_pyr1[idx[ii]];
								const Mat& prev = dog_pyr1[idx[ii] - 1];
								const Mat& next = dog_pyr1[idx[ii] + 1];
								dx[ii * 3] = (img.at<double>(y_A[ii], x_A[ii] + 1) - img.at<double>(y_A[ii], x_A[ii] - 1)) * deriv_scale;
								dx[ii * 3 + 1] = (img.at<double>(y_A[ii] + 1, x_A[ii]) - img.at<double>(y_A[ii] - 1, x_A[ii])) * deriv_scale;
								dx[ii * 3 + 2] = (next.at<double>(y_A[ii], x_A[ii]) - prev.at<double>(y_A[ii], x_A[ii])) * deriv_scale;
								
								double v2 = (double)img.at<double>(y_A[ii], x_A[ii]) * 2;
								double dxx = (img.at<double>(y_A[ii], x_A[ii] + 1) + img.at<double>(y_A[ii], x_A[ii] - 1) - v2) * second_deriv_scale;
								double dyy = (img.at<double>(y_A[ii] + 1, x_A[ii]) + img.at<double>(y_A[ii] - 1, x_A[ii]) - v2) * second_deriv_scale;
								double dss = (next.at<double>(y_A[ii], x_A[ii]) + prev.at<double>(y_A[ii], x_A[ii]) - v2) * second_deriv_scale;
								double dxy = (img.at<double>(y_A[ii] + 1, x_A[ii] + 1) - img.at<double>(y_A[ii] + 1, x_A[ii] - 1) -
									img.at<double>(y_A[ii] - 1, x_A[ii] + 1) + img.at<double>(y_A[ii] - 1, x_A[ii] - 1)) * cross_deriv_scale;
								double dxs = (next.at<double>(y_A[ii], x_A[ii] + 1) - next.at<double>(y_A[ii], x_A[ii] - 1) -
									prev.at<double>(y_A[ii], x_A[ii] + 1) + prev.at<double>(y_A[ii], x_A[ii] - 1)) * cross_deriv_scale;
								double dys = (next.at<double>(y_A[ii] + 1, x_A[ii]) - next.at<double>(y_A[ii] - 1, x_A[ii]) -
									prev.at<double>(y_A[ii] + 1, x_A[ii]) + prev.at<double>(y_A[ii] - 1, x_A[ii])) * cross_deriv_scale;
								double H[9] = { dxx, dxy, dxs,dxy, dyy, dys, dxs, dys, dss };
								Mat H_1 = ArrayToMat(H, 3, 3);
								secMatInv1_1(ii, ptext1, ptext2, A1, A2, A3, A_1, B_1, C_1, Z_1, H_1, 3, 3);
							}
						}
						ii++;
					}
				}
			}
		}
		ii = 0;
		if (sem.post() == false)  ;//printf("sem.post failed.\n");
		//else printf("A  sem.post ok\n");
		if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
		//else printf("A  sem.wait ok\n");
		for (int o = 0; o < octaves; o++) {
			for (int i = 1; i < (intervals + 2) - 1; i++) {
				int index = o * (intervals + 2) + i;
				for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
					for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
						if (DetectionLocalExtrema1[ii] == 1) {
							if (flag[ii] == 0) {}
							else {
								secMatInv1_2(ii, ptext1, ptext2, A1, A2, A3, A_1, B_1, C_1, 3, 3);
							}
						}
						ii++;
					}
				}
			}
		}
		ii = 0;
		if (sem.post() == false)  ;//printf("sem.post failed.\n");
		//else printf("A  sem.post ok\n");
		if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
		//else printf("A  sem.wait ok\n");
		for (int o = 0; o < octaves; o++) {
			for (int i = 1; i < (intervals + 2) - 1; i++) {
				int index = o * (intervals + 2) + i;
				for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
					for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
						if (DetectionLocalExtrema1[ii] == 1) {
							if (flag[ii] == 0) {  }
							else {
								secMatInv1_3(ii, ptext1, ptext2, A1, A2, A3, A_1, B_1, C_1, Z_1, 3, 3);
								for (int m = 0; m < 3; m++) {
									offset_x[ii * 3 + m] = 0.0;
									for (int n = 0; n < 3; n++) {
										secMul1_1(ii * 9 + m * 3 + n, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, A2[ii * 9 + m * 3 + n], dx[ii * 3 + n]);
									}
								}
							}
							
						}
						ii++;
					}
				}
			}
		}
		ii = 0;
		if (sem.post() == false)  ;//printf("sem.post failed.\n");
		//else printf("A  sem.post ok\n");
		if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
		//else printf("A  sem.wait ok\n");
		for (int o = 0; o < octaves; o++) {
			for (int i = 1; i < (intervals + 2) - 1; i++) {
				int index = o * (intervals + 2) + i;
				for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
					for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
						if (DetectionLocalExtrema1[ii] == 1) {
							if (flag[ii] == 0) {}
							else {
								for (int m = 0; m < 3; m++) {
									for (int n = 0; n < 3; n++) {
										secMul1_2(ii * 9 + m * 3 + n, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
										offset_x[ii * 3 + m] += a3[ii * 9 + m * 3 + n];
									}
									offset_x[ii * 3 + m] = -offset_x[ii * 3 + m];
									ptext1[ii * 3 + m] = offset_x[ii * 3 + m];
								}
							}
						}
						ii++;
					}
				}
			}
		}
		ii = 0;
		if (sem.post() == false)  ;//printf("sem.post failed.\n");
		//else printf("A  sem.post ok\n");
		if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
		//else printf("A  sem.wait ok\n");
		
		for (int o = 0; o < octaves; o++) {
			for (int i = 1; i < (intervals + 2) - 1; i++) {
				int index = o * (intervals + 2) + i;
				const Mat& mat = dog_pyr1[index];
				for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
					for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
						if (DetectionLocalExtrema1[ii] == 1) {
							if (flag[ii] == 0) { }
							else {
								for (int m = 0; m < 3; m++) {
									offset_x[ii * 3 + m] += ptext2[ii * 3 + m];
								}

								if (fabs(offset_x[ii * 3]) < 0.5 && fabs(offset_x[ii * 3 + 1]) < 0.5 && fabs(offset_x[ii * 3 + 2]) < 0.5) {
									flag[ii] = 0;
									//cout << "x:" << x << "  y:" << y << endl;

								}
								else {
									x_A[ii] += cvRound(offset_x[ii * 3]);
									y_A[ii] += cvRound(offset_x[ii * 3 + 1]);
									interval_A[ii] += cvRound(offset_x[ii * 3 + 2]);
									idx[ii] = index - i + interval_A[ii];
									if (interval_A[ii] < 1 || interval_A[ii] > INTERVALS || x_A[ii] >= mat.cols - 1 || x_A[ii] < 2 || y_A[ii] >= mat.rows - 1 || y_A[ii] < 2)
									{
										DetectionLocalExtrema1[ii] = 0;
									}
								}
							}
						}
						ii++;
					}
				}
			}
		}
		ii = 0;
	}
	ii = 0;


	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema1[ii] == 1) {
						if (flag[ii] == 1) {                    //窜改失败if (i >= MAX_INTERPOLATION_STEPS)
							DetectionLocalExtrema1[ii] = 0; ii++; continue;
						}
						else {
							offset_x[ii * 3] *= 0.5;
							offset_x[ii * 3 + 1] *= 0.5;
							offset_x[ii * 3 + 2] *= 0.5;
							const Mat& img = dog_pyr1[idx[ii]];
							const Mat& prev = dog_pyr1[idx[ii] - 1];
							const Mat& next = dog_pyr1[idx[ii] + 1];
							dx[ii * 3] = (img.at<double>(y_A[ii], x_A[ii] + 1) - img.at<double>(y_A[ii], x_A[ii] - 1)) * deriv_scale;
							dx[ii * 3 + 1] = (img.at<double>(y_A[ii] + 1, x_A[ii]) - img.at<double>(y_A[ii] - 1, x_A[ii])) * deriv_scale;
							dx[ii * 3 + 2] = (next.at<double>(y_A[ii], x_A[ii]) - prev.at<double>(y_A[ii], x_A[ii])) * deriv_scale;
							term[ii] = 0.0;
							for (int m = 0; m < 3; m++) {
								secMul1_1(ii * 3 + m, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, dx[ii * 3 + m], offset_x[ii * 3 + m]);
							}
						}
						
					}
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			pixel_t* data = (pixel_t*)dog_pyr1[index].data;
			int step = dog_pyr1[index].cols;
			for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema1[ii] == 1) {
						pixel_t val = *(data + y_A[ii] * step + x_A[ii]);
						for (int m = 0; m < 3; m++) {
							secMul1_2(ii * 3 + m, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
							term[ii] += a3[ii * 3 + m];
						}
						term[ii] = val * img_scale + 0.5 * term[ii];
						secAbsCmp1_1(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, term[ii]);
					}
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	double FabsDx_thresh = dxthreshold / INTERVALS;
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema1[ii] == 1) {
						secAbsCmp1_2(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, t_1, FabsDx_thresh);
					}
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema1[ii] == 1) {
						secAbsCmp1_3(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
					}
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("++++A  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema1[ii] == 1) {
						secAbsCmp1_4(ii, ptext1, ptext2, a1, a2, a3);
						if (a1[ii] == -1) { DetectionLocalExtrema1[ii] = 0; 
						}
						if (DetectionLocalExtrema1[ii] == 1) {
							keypoint[ii].x = x_A[ii];
							keypoint[ii].y = y_A[ii];
							keypoint[ii].offset_x = offset_x[ii * 3] * 2;
							keypoint[ii].offset_y = offset_x[ii * 3 + 1] * 2;
							keypoint[ii].interval = interval_A[ii];
							keypoint[ii].offset_interval = offset_x[ii * 3 + 2] * 2;
							keypoint[ii].octave = o;
							keypoint[ii].dx = (x + offset_x[ii * 3]*2) * pow(2.0, o);
							keypoint[ii].dy = (y + offset_x[ii * 3 + 1] * 2) * pow(2.0, o);
						}
					}
					ii++;
				}
			}
		}
	}
	ii = 0;
	
}
void InterploationExtremum2(CSEM sem, double* ptext1, double* ptext2, int* DetectionLocalExtrema2, Keypoint* keypoint, const vector<Mat>& dog_pyr2, int octaves, double a_2, double b_2, double c_2, double t_2, Mat& A_2, Mat& B_2, Mat& C_2, Mat& Z_2, int intervals = INTERVALS, double dxthreshold = DXTHRESHOLD) {
	const double img_scale = 1.0 / (255 * 1);// / (255 * 1);
	const double deriv_scale = img_scale * 0.5;
	const double second_deriv_scale = img_scale;
	const double cross_deriv_scale = img_scale * 0.25;
	double offset_x[4000000 * 3];
	double dx[4000000 * 3];
	double B1[4000000 * 3 * 3];
	double B2[4000000 * 3 * 3];
	double B3[4000000 * 3 * 3];
	double b1[4000000 * 3 * 3];
	double b2[4000000 * 3 * 3];
	double b3[4000000 * 3 * 3];
	int idx[4000000];
	int x_B[4000000];
	int y_B[4000000];
	int interval_B[4000000];
	int flag[4000000];
	double term[4000000];
	int ii = 0;
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr2[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr2[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema2[ii] == 1) {
						x_B[ii] = x;
						y_B[ii] = y;
						interval_B[ii] = i;
						flag[ii] = 1;
						idx[ii] = index;
					}
					ii++;
				}
			}
		}
	}
	ii = 0;
	for (int l = 0; l < MAX_INTERPOLATION_STEPS; l++) {
		if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
		//else printf("B  sem.wait ok\n");
		for (int o = 0; o < octaves; o++) {
			for (int i = 1; i < (intervals + 2) - 1; i++) {
				int index = o * (intervals + 2) + i;
				for (int y = IMG_BORDER; y < dog_pyr2[index].rows - IMG_BORDER; y++) {
					for (int x = IMG_BORDER; x < dog_pyr2[index].cols - IMG_BORDER; x++) {
						if (DetectionLocalExtrema2[ii] == 1) {
							if (flag[ii] == 0) { }
							else {
								const Mat& img = dog_pyr2[idx[ii]];
								const Mat& prev = dog_pyr2[idx[ii] - 1];
								const Mat& next = dog_pyr2[idx[ii] + 1];
								dx[ii * 3] = (img.at<double>(y_B[ii], x_B[ii] + 1) - img.at<double>(y_B[ii], x_B[ii] - 1)) * deriv_scale;
								dx[ii * 3 + 1] = (img.at<double>(y_B[ii] + 1, x_B[ii]) - img.at<double>(y_B[ii] - 1, x_B[ii])) * deriv_scale;
								dx[ii * 3 + 2] = (next.at<double>(y_B[ii], x_B[ii]) - prev.at<double>(y_B[ii], x_B[ii])) * deriv_scale;
								
								double v2 = (double)img.at<double>(y_B[ii], x_B[ii]) * 2;
								double dxx = (img.at<double>(y_B[ii], x_B[ii] + 1) + img.at<double>(y_B[ii], x_B[ii] - 1) - v2) * second_deriv_scale;
								double dyy = (img.at<double>(y_B[ii] + 1, x_B[ii]) + img.at<double>(y_B[ii] - 1, x_B[ii]) - v2) * second_deriv_scale;
								double dss = (next.at<double>(y_B[ii], x_B[ii]) + prev.at<double>(y_B[ii], x_B[ii]) - v2) * second_deriv_scale;
								double dxy = (img.at<double>(y_B[ii] + 1, x_B[ii] + 1) - img.at<double>(y_B[ii] + 1, x_B[ii] - 1) -
									img.at<double>(y_B[ii] - 1, x_B[ii] + 1) + img.at<double>(y_B[ii] - 1, x_B[ii] - 1)) * cross_deriv_scale;
								double dxs = (next.at<double>(y_B[ii], x_B[ii] + 1) - next.at<double>(y_B[ii], x_B[ii] - 1) -
									prev.at<double>(y_B[ii], x_B[ii] + 1) + prev.at<double>(y_B[ii], x_B[ii] - 1)) * cross_deriv_scale;
								double dys = (next.at<double>(y_B[ii] + 1, x_B[ii]) - next.at<double>(y_B[ii] - 1, x_B[ii]) -
									prev.at<double>(y_B[ii] + 1, x_B[ii]) + prev.at<double>(y_B[ii] - 1, x_B[ii])) * cross_deriv_scale;
								double H[9] = { dxx, dxy, dxs,dxy, dyy, dys, dxs, dys, dss };
								Mat H_2 = ArrayToMat(H, 3, 3);
								secMatInv2_1(ii, ptext1, ptext2, B1, B2, B3, A_2, B_2, C_2, Z_2, H_2, 3, 3);
							}
						}
						ii++;
					}
				}
			}
		}
		ii = 0;
		if (sem.post() == false)  ;//printf("sem.post failed.\n");
		//else printf("B  sem.post ok\n");
		if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
		//else printf("B  sem.wait ok\n");
		for (int o = 0; o < octaves; o++) {
			for (int i = 1; i < (intervals + 2) - 1; i++) {
				int index = o * (intervals + 2) + i;
				for (int y = IMG_BORDER; y < dog_pyr2[index].rows - IMG_BORDER; y++) {
					for (int x = IMG_BORDER; x < dog_pyr2[index].cols - IMG_BORDER; x++) {
						if (DetectionLocalExtrema2[ii] == 1) {
							if (flag[ii] == 0) {  }
							else {
								secMatInv2_2(ii, ptext1, ptext2, B1, B2, B3, A_2, B_2, C_2, Z_2, 3, 3);
								
							}
						}

						ii++;
					}
				}
			}
		}
		ii = 0;
		if (sem.post() == false)  ;//printf("sem.post failed.\n");
		//else printf("B  sem.post ok\n");
		if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
		//else printf("B  sem.wait ok\n");
		for (int o = 0; o < octaves; o++) {
			for (int i = 1; i < (intervals + 2) - 1; i++) {
				int index = o * (intervals + 2) + i;
				for (int y = IMG_BORDER; y < dog_pyr2[index].rows - IMG_BORDER; y++) {
					for (int x = IMG_BORDER; x < dog_pyr2[index].cols - IMG_BORDER; x++) {
						if (DetectionLocalExtrema2[ii] == 1) {
							if (flag[ii] == 0) { }
							else {
								for (int m = 0; m < 3; m++) {
									offset_x[ii * 3 + m] = 0.0;
									for (int n = 0; n < 3; n++) {
										secMul2_1(ii * 9 + m * 3 + n, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, B2[ii * 9 + m * 3 + n], dx[ii * 3 + n]);
										offset_x[ii * 3 + m] += b3[ii * 9 + m * 3 + n];
									}
									offset_x[ii * 3 + m] = -offset_x[ii * 3 + m];
								}
							}
						}
						ii++;
					}
				}
			}
		}
		ii = 0;
		if (sem.post() == false)  ;//printf("sem.post failed.\n");
		//else printf("B  sem.post ok\n");
		if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
		//else printf("B  sem.wait ok\n");
		for (int o = 0; o < octaves; o++) {
			for (int i = 1; i < (intervals + 2) - 1; i++) {
				int index = o * (intervals + 2) + i;
				const Mat& mat = dog_pyr2[index];
				for (int y = IMG_BORDER; y < dog_pyr2[index].rows - IMG_BORDER; y++) {
					for (int x = IMG_BORDER; x < dog_pyr2[index].cols - IMG_BORDER; x++) {
						if (DetectionLocalExtrema2[ii] == 1) {
							if (flag[ii] == 0) { }
							else {
								for (int m = 0; m < 3; m++) {
									ptext2[ii * 3 + m] = offset_x[ii * 3 + m];
									offset_x[ii * 3 + m] += ptext1[ii * 3 + m];
								}
								if (fabs(offset_x[ii * 3]) < 0.5 && fabs(offset_x[ii * 3 + 1]) < 0.5 && fabs(offset_x[ii * 3 + 2]) < 0.5) {
									flag[ii] = 0;
								}
								else {
									x_B[ii] += cvRound(offset_x[ii * 3]);
									y_B[ii] += cvRound(offset_x[ii * 3 + 1]);
									interval_B[ii] += cvRound(offset_x[ii * 3 + 2]);
									idx[ii] = index - i + interval_B[ii];
									if (interval_B[ii] < 1 || interval_B[ii] > INTERVALS || x_B[ii] >= mat.cols - 1 || x_B[ii] < 2 || y_B[ii] >= mat.rows - 1 || y_B[ii] < 2)
									{
										DetectionLocalExtrema2[ii] = 0;
									}
								}
							}
						}
						ii++;
					}
				}
			}
		}
		ii = 0;
		if (sem.post() == false)  ;//printf("sem.post failed.\n");
		//else printf("B  sem.post ok\n");
	}
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	ii = 0;
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			pixel_t* data = (pixel_t*)dog_pyr2[index].data;
			int step = dog_pyr2[index].cols;
			for (int y = IMG_BORDER; y < dog_pyr2[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr2[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema2[ii] == 1) {
						if (flag[ii] == 1) {
							DetectionLocalExtrema2[ii] = 0; ii++; continue;
						}
						else {
							offset_x[ii * 3] *= 0.5;
							offset_x[ii * 3 + 1] *= 0.5;
							offset_x[ii * 3 + 2] *= 0.5;
							const Mat& img = dog_pyr2[idx[ii]];
							const Mat& prev = dog_pyr2[idx[ii] - 1];
							const Mat& next = dog_pyr2[idx[ii] + 1];
							dx[ii * 3] = (img.at<double>(y_B[ii], x_B[ii] + 1) - img.at<double>(y_B[ii], x_B[ii] - 1)) * deriv_scale;
							dx[ii * 3 + 1] = (img.at<double>(y_B[ii] + 1, x_B[ii]) - img.at<double>(y_B[ii] - 1, x_B[ii])) * deriv_scale;
							dx[ii * 3 + 2] = (next.at<double>(y_B[ii], x_B[ii]) - prev.at<double>(y_B[ii], x_B[ii])) * deriv_scale;
							term[ii] = 0.0;
							for (int m = 0; m < 3; m++) {
								secMul2_1(ii * 3 + m, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, dx[ii * 3 + m], offset_x[ii * 3 + m]);
								term[ii] += b3[ii * 3 + m];
							}
							pixel_t val = *(data + y_B[ii] * step + x_B[ii]);
							term[ii] = val * img_scale + 0.5 * term[ii];
						}
						
					}
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	ii = 0;
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr2[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr2[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema2[ii] == 1) {
						secAbsCmp2_1(ii, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, term[ii]);
					}
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr2[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr2[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema2[ii] == 1) {
						secAbsCmp2_2(ii, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, t_2);
					}
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr2[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr2[index].cols - IMG_BORDER; x++){
					if (DetectionLocalExtrema2[ii] == 1) {
						secAbsCmp2_3(ii, ptext1, ptext2, b1, b2, b3);
						if (b1[ii] == -1) { DetectionLocalExtrema2[ii] = 0; //cout << "|D(x^)| " << "x: " << x << "  y:" << y << endl;
						}
						if (DetectionLocalExtrema2[ii] == 1) {
							keypoint[ii].x = x_B[ii];
							keypoint[ii].y = y_B[ii];
							keypoint[ii].offset_x = offset_x[ii * 3] * 2;
							keypoint[ii].offset_y = offset_x[ii * 3 + 1] * 2;
							keypoint[ii].interval = interval_B[ii];
							keypoint[ii].offset_interval = offset_x[ii * 3 + 2] * 2;
							keypoint[ii].octave = o;
							keypoint[ii].dx = (x_B[ii] + offset_x[ii * 3]*2) * pow(2.0, o);
							keypoint[ii].dy = (y_B[ii] + offset_x[ii * 3 + 1] * 2) * pow(2.0, o);
						}
					}
					ii++;
				}
			}
		}
	}
	
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
}
void passEdgeResponse1(CSEM sem, double* ptext1, double* ptext2, int* DetectionLocalExtrema1, Keypoint* keypoint, const vector<Mat>& dog_pyr1, vector<Keypoint>& extrema1, int octaves, double a_1, double b_1, double c_1, double t_1, double r = RATIO,int intervals = INTERVALS) {
	double Dxx[4000000];
	double Dyy[4000000];
	double Dxy[4000000];
	double Tr_h_1[4000000];
	double Det_h_1[4000000];
	double a1[4000000];
	double a2[4000000];
	double a3[4000000];
	double a4[4000000];
	double a5[4000000];
	int ii = 0;

	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			pixel_t* data = (pixel_t*)dog_pyr1[index].data;  //极值点的图片索引定位
			int step = dog_pyr1[index].cols;
			for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema1[ii] == 1) {
						pixel_t val = *(data + keypoint[ii].y * step + keypoint[ii].x);
						Dxx[ii] = DAt(keypoint[ii].x + 1, keypoint[ii].y) + DAt(keypoint[ii].x - 1, keypoint[ii].y) - 2 * val;
						Dyy[ii] = DAt(keypoint[ii].x, keypoint[ii].y + 1) + DAt(keypoint[ii].x, keypoint[ii].y - 1) - 2 * val;
						Dxy[ii] = (DAt(keypoint[ii].x + 1, keypoint[ii].y + 1) + DAt(keypoint[ii].x - 1, keypoint[ii].y - 1) - DAt(keypoint[ii].x - 1, keypoint[ii].y + 1) - DAt(keypoint[ii].x + 1, keypoint[ii].y - 1)) / 4.0;
						Tr_h_1[ii] = Dxx[ii] + Dyy[ii];
						secMul1_1(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, Dxx[ii], Dyy[ii]);
					}
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema1[ii] == 1) {
						secMul1_2(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
						a4[ii] = a3[ii];
						secMul1_1(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, Dxy[ii], Dxy[ii]);
					}
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	double thresh = 0;
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema1[ii] == 1) {
						secMul1_2(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
						a5[ii] = a3[ii];
						Det_h_1[ii] = a4[ii] - a5[ii];
						secCmp1_1(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, Det_h_1[ii], thresh, t_1);
					}
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	thresh = 0;
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema1[ii] == 1) {
						secCmp1_2(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
					}
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema1[ii] == 1) {
						secCmp1_3(ii, ptext1, ptext2, a1, a2, a3);
						if (a1[ii] <= 0) {
							DetectionLocalExtrema1[ii] = 0;
						}
						else {
							secMul1_1(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, Tr_h_1[ii], Tr_h_1[ii]);
						}
					}
					ii++;
				}
			}
		}
	}
	ii = 0;

	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema1[ii] == 1) {
						secMul1_2(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
						a1[ii] = r * a3[ii];
						a2[ii] = Det_h_1[ii] * (r + 1) * (r + 1);
						secCmp1_1(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, a1[ii], a2[ii], t_1);
					}
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema1[ii] == 1) {
						secCmp1_2(ii, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
					}
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	int n = 0;
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			pixel_t* data = (pixel_t*)dog_pyr1[index].data;  //极值点的图片索引定位
			int step = dog_pyr1[index].cols;
			for (int y = IMG_BORDER; y < dog_pyr1[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr1[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema1[ii] == 1) {
						secCmp1_3(ii, ptext1, ptext2, a1, a2, a3);
						if (a1[ii] == -1) {
							keypoint[ii].val = *(data + keypoint[ii].y * step + keypoint[ii].x);
							extrema1.push_back(keypoint[ii]);
						}
						else DetectionLocalExtrema1[ii] = 0;
					}
					ii++;
				}
			}
		}
	}
}
void passEdgeResponse2(CSEM sem, double* ptext1, double* ptext2, int* DetectionLocalExtrema2, Keypoint* keypoint, const vector<Mat>& dog_pyr2, vector<Keypoint>& extrema2, int octaves, double a_2, double b_2, double c_2, double t_2, double r = RATIO, int intervals = INTERVALS) {
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	double Dxx[4000000];
	double Dyy[4000000];
	double Dxy[4000000];
	double Tr_h_2[4000000];
	double Det_h_2[4000000];
	double b1[4000000];
	double b2[4000000];
	double b3[4000000];
	double b4[4000000];
	double b5[4000000];
	int ii = 0;
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			pixel_t* data = (pixel_t*)dog_pyr2[index].data;  //极值点的图片索引定位
			int step = dog_pyr2[index].cols;
			for (int y = IMG_BORDER; y < dog_pyr2[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr2[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema2[ii] == 1) {
						pixel_t val = *(data + keypoint[ii].y * step + keypoint[ii].x);
						Dxx[ii] = DAt(keypoint[ii].x + 1, keypoint[ii].y) + DAt(keypoint[ii].x - 1, keypoint[ii].y) - 2 * val;
						Dyy[ii] = DAt(keypoint[ii].x, keypoint[ii].y + 1) + DAt(keypoint[ii].x, keypoint[ii].y - 1) - 2 * val;
						Dxy[ii] = (DAt(keypoint[ii].x + 1, keypoint[ii].y + 1) + DAt(keypoint[ii].x - 1, keypoint[ii].y - 1) - DAt(keypoint[ii].x - 1, keypoint[ii].y + 1) - DAt(keypoint[ii].x + 1, keypoint[ii].y - 1)) / 4.0;
						Tr_h_2[ii] = Dxx[ii] + Dyy[ii];
						secMul2_1(ii, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, Dxx[ii], Dyy[ii]);
						b4[ii] = b3[ii];
					}
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr2[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr2[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema2[ii] == 1) {
						secMul2_1(ii, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, Dxy[ii], Dxy[ii]);
						b5[ii] = b3[ii];
						Det_h_2[ii] = b4[ii] - b5[ii];
					}
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	double thresh = 0;
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr2[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr2[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema2[ii] == 1) {
						secCmp2_1(ii, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, Det_h_2[ii], thresh, t_2);
					}
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr2[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr2[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema2[ii] == 1) {
						secCmp2_2(ii, ptext1, ptext2, b1, b2, b3);
					}
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr2[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr2[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema2[ii] == 1) {
						if (b1[ii] <= 0) {
							DetectionLocalExtrema2[ii] = 0;
						}
						else {
							secMul2_1(ii, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, Tr_h_2[ii], Tr_h_2[ii]);
							b1[ii] = r * b3[ii];
							b2[ii] = Det_h_2[ii] * (r + 1) * (r + 1);
						}
					}
					ii++;
				}
			}
		}
	}
	
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			for (int y = IMG_BORDER; y < dog_pyr2[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr2[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema2[ii] == 1) {
						secCmp2_1(ii, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, b1[ii], b2[ii], t_2);
					}
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int o = 0; o < octaves; o++) {
		for (int i = 1; i < (intervals + 2) - 1; i++) {
			int index = o * (intervals + 2) + i;
			pixel_t* data = (pixel_t*)dog_pyr2[index].data;  //极值点的图片索引定位
			int step = dog_pyr2[index].cols;
			for (int y = IMG_BORDER; y < dog_pyr2[index].rows - IMG_BORDER; y++) {
				for (int x = IMG_BORDER; x < dog_pyr2[index].cols - IMG_BORDER; x++) {
					if (DetectionLocalExtrema2[ii] == 1) {
						secCmp2_2(ii, ptext1, ptext2, b1, b2, b3);
						if (b1[ii] == -1) {
							keypoint[ii].val = *(data + keypoint[ii].y * step + keypoint[ii].x);
							extrema2.push_back(keypoint[ii]);
						}
						else DetectionLocalExtrema2[ii] = 0;
					}
					ii++;
				}
			}
		}
	}
	ii = 0;
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
}
void DetectionLocalExtrema1(CSEM sem, double* ptext1, double* ptext2, const vector<Mat>& dog_pyr1, vector<Keypoint>& extrema1, int octaves, double a_1, double b_1, double c_1, double t_1, Mat& A_1, Mat& B_1, Mat& C_1, Mat& Z_1, int intervals ) {
	double  thresh = 0.5 * DXTHRESHOLD / intervals;
	
	int DetectionLocalExtrema1[4000000] ;
	threshing1(sem, ptext1, ptext2, DetectionLocalExtrema1, thresh, dog_pyr1, octaves, a_1, b_1, c_1, t_1);
	cout << "threshing1 jieshu " << endl;
	int n = 0;
	for (int p = 0; p < 4000000; p++) {
		if (DetectionLocalExtrema1[p] == 1) n++;
	}
	cout << "n     " <<n<< endl;


	isExtremum1(sem, ptext1, ptext2, DetectionLocalExtrema1, dog_pyr1, octaves, a_1, b_1, c_1, t_1);
	cout << "isExtremum1 jieshu " << endl;
	for (int p = 0; p < 4000000; p++) {
		if (DetectionLocalExtrema1[p] == 1) n++;
	}
	cout << "n     " << n << endl;


	Keypoint keypoint[4839999];
	InterploationExtremum1(sem, ptext1, ptext2, DetectionLocalExtrema1, keypoint, dog_pyr1, octaves, a_1, b_1, c_1, t_1, A_1, B_1, C_1, Z_1);
	cout << "InterploationExtremum1 jieshu " << endl;
	
	passEdgeResponse1(sem, ptext1, ptext2, DetectionLocalExtrema1, keypoint, dog_pyr1, extrema1, octaves, a_1, b_1, c_1, t_1);
	cout << "passEdgeResponse1 jieshu " << endl;
}
void DetectionLocalExtrema2(CSEM sem, double* ptext1, double* ptext2, const vector<Mat>& dog_pyr2, vector<Keypoint>& extrema2, int octaves, double a_2, double b_2, double c_2, double t_2, Mat& A_2, Mat& B_2, Mat& C_2, Mat& Z_2, int intervals ) {
	double  thresh = 0.5 * DXTHRESHOLD / intervals;

	int DetectionLocalExtrema2[4000000] ;
	threshing2(sem, ptext1, ptext2, DetectionLocalExtrema2, thresh,dog_pyr2, octaves, a_2, b_2, c_2, t_2);
	cout << "threshing2 jieshu " << endl;
	
	isExtremum2(sem, ptext1, ptext2, DetectionLocalExtrema2, dog_pyr2, octaves, a_2, b_2, c_2, t_2);
	cout << "isExtremum2 jieshu " << endl;

	Keypoint keypoint[4839999];
	InterploationExtremum2(sem, ptext1, ptext2, DetectionLocalExtrema2, keypoint, dog_pyr2, octaves, a_2, b_2, c_2, t_2, A_2, B_2, C_2, Z_2);
	cout << "InterploationExtremum2 jieshu " << endl;
	passEdgeResponse2(sem, ptext1, ptext2, DetectionLocalExtrema2, keypoint, dog_pyr2, extrema2, octaves, a_2, b_2, c_2, t_2);
	cout << "passEdgeResponse2 jieshu " << endl;
}

void ori1( int j, int * radius,int * flag, CSEM sem, double* ptext1, double* ptext2, double* ori, double a_1, double b_1, double c_1, double* sgn_x, double* sgn_y, double* x_1, double t_1) {
	double a1[1600 *j * 9];
	double a2[1600 * j * 9];
	double a3[1600 * j * 9];
	double flag1[1600 * j];
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					flag1[i * 1600 + (m + 20) * 40 + (n + 20)] = 1;
					if (sgn_x[i * 1600 + (m + 20) * 40 + (n + 20)] >= 0 && sgn_y[i * 1600 + (m + 20) * 40 + (n + 20)] > 0) {
						double y_1[9] = { tan(CV_PI / 18), tan(CV_PI / 9), tan(CV_PI / 6), tan(CV_PI / 4.5) , tan(CV_PI / 3.6) , tan(CV_PI / 3) , tan(7 * CV_PI / 18) , tan(CV_PI / 2.25) , tan(CV_PI / 2) };
						for (int k = 0; k < 9; k++) {
							a1[i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k] = x_1[i * 1600 + (m + 20) * 40 + (n + 20)] - y_1[k];
							secMul1_1(i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, t_1, a1[i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k]);
						}

					}
					else if (sgn_x[i * 1600 + (m + 20) * 40 + (n + 20)] < 0 && sgn_y[i * 1600 + (m + 20) * 40 + (n + 20)] >= 0) {
						double y_1[9] = { tan(CV_PI / 1.8), tan(11 * CV_PI / 18) , tan(CV_PI / 1.5) , tan(13 * CV_PI / 18) , tan(14 * CV_PI / 18) , tan(CV_PI / 1.2), tan(16 * CV_PI / 18), tan(17 * CV_PI / 18), tan(18 * CV_PI / 18) };
						for (int k = 0; k < 9; k++) {
							a1[i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k] = x_1[i * 1600 + (m + 20) * 40 + (n + 20)] - y_1[k];
							secMul1_1(i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, t_1, a1[i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k]);
						}

					}
					else if (sgn_x[i * 1600 + (m + 20) * 40 + (n + 20)] <= 0 && sgn_y[i * 1600 + (m + 20) * 40 + (n + 20)] < 0) {
						double y_1[9] = { tan(1 * CV_PI / 18), tan(2 * CV_PI / 18), tan(3 * CV_PI / 18), tan(4 * CV_PI / 18) , tan(5 * CV_PI / 18) , tan(6 * CV_PI / 18) , tan(7 * CV_PI / 18) , tan(8 * CV_PI / 18) ,tan(9 * CV_PI / 18) };
						for (int k = 0; k < 9; k++) {
							a1[i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k] = x_1[i * 1600 + (m + 20) * 40 + (n + 20)] - y_1[k];
							secMul1_1(i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, t_1, a1[i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k]);
						}

					}
					else {
						double y_1[9] = { tan(28 * CV_PI / 18), tan(29 * CV_PI / 18), tan(30 * CV_PI / 18), tan(31 * CV_PI / 18), tan(32 * CV_PI / 18), tan(33 * CV_PI / 18), tan(34 * CV_PI / 18), tan(35 * CV_PI / 18), tan(36 * CV_PI / 18) };
						for (int k = 0; k < 9; k++) {
							a1[i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k] = x_1[i * 1600 + (m + 20) * 40 + (n + 20)] - y_1[k];
							secMul1_1(i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, t_1, a1[i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k]);
						}
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					for (int k = 0; k < 9; k++) {
						secMul1_2(i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
						ptext1[i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k] = a3[i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k];
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	int k;
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					for (k = 0; k < 9; k++) {
						a1[i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k] = sgn(a3[i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k] + ptext1[i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k]);
						if (flag1[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
							if (a1[i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k] == -1) {
								ori[i * 1600 + (m + 20) * 40 + (n + 20)] = k + 1;
								flag1[i * 1600 + (m + 20) * 40 + (n + 20)] = 0;
							}
						}
					}
					if (flag1[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) ori[i * 1600 + (m + 20) * 40 + (n + 20)] = k;
					
					if (sgn_x[i * 1600 + (m + 20) * 40 + (n + 20)] >= 0 && sgn_y[i * 1600 + (m + 20) * 40 + (n + 20)] > 0) {
						ori[i * 1600 + (m + 20) * 40 + (n + 20)] = ori[i * 1600 + (m + 20) * 40 + (n + 20)] * 10 - 5;
					}
					else if (sgn_x[i * 1600 + (m + 20) * 40 + (n + 20)] < 0 && sgn_y[i * 1600 + (m + 20) * 40 + (n + 20)] >= 0) {
						ori[i * 1600 + (m + 20) * 40 + (n + 20)] = (ori[i * 1600 + (m + 20) * 40 + (n + 20)] + 9) * 10 - 5;
					}
					else if (sgn_x[i * 1600 + (m + 20) * 40 + (n + 20)] <= 0 && sgn_y[i * 1600 + (m + 20) * 40 + (n + 20)] < 0) {
						ori[i * 1600 + (m + 20) * 40 + (n + 20)] = ori[i * 1600 + (m + 20) * 40 + (n + 20)] * 10 - 180 - 5;
					}
					else {
						ori[i * 1600 + (m + 20) * 40 + (n + 20)] = (ori[i * 1600 + (m + 20) * 40 + (n + 20)] - 9) * 10 - 5;
					}
				}
			}
		}
	}
	
}
void ori2(int j, int* radius, int* flag, CSEM sem, double* ptext1, double* ptext2, double* ori, double a_2, double b_2, double c_2, double* sgn_x, double* sgn_y, double* x_2,  double t_2) {
	double b1[1600 * j * 9];
	double b2[1600 * j * 9];
	double b3[1600 * j * 9];
	int flag2[1600 * j];
	double y_2[9] = { 0 };
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					flag2[i * 1600 + (m + 20) * 40 + (n + 20)] = 1;
					for (int k = 0; k < 9; k++) {
						b1[i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k] = x_2[i * 1600 + (m + 20) * 40 + (n + 20)] - y_2[k];
						secMul2_1(i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, t_2, b1[i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k]);
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	int k;
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					for (k = 0; k < 9; k++) {
						b1[i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k] = sgn(b3[i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k] + ptext1[i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k]);
						ptext1[i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k] = b3[i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k];
						if (flag2[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
							if (b1[i * 1600 * 9 + (m + 20) * 40 * 9 + (n + 20) * 9 + k] == -1) {
								ori[i * 1600 + (m + 20) * 40 + (n + 20)] = k + 1;
								flag2[i * 1600 + (m + 20) * 40 + (n + 20)] = 0;
							}
						}
					}
					if(flag2[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) ori[i * 1600 + (m + 20) * 40 + (n + 20)] = k;
					if (sgn_x[i * 1600 + (m + 20) * 40 + (n + 20)] >= 0 && sgn_y[i * 1600 + (m + 20) * 40 + (n + 20)] > 0) {
						ori[i * 1600 + (m + 20) * 40 + (n + 20)] = ori[i * 1600 + (m + 20) * 40 + (n + 20)] * 10 - 5;
					}
					else if (sgn_x[i * 1600 + (m + 20) * 40 + (n + 20)] < 0 && sgn_y[i * 1600 + (m + 20) * 40 + (n + 20)] >= 0) {
						ori[i * 1600 + (m + 20) * 40 + (n + 20)] = (ori[i * 1600 + (m + 20) * 40 + (n + 20)] + 9) * 10 - 5;
					}
					else if (sgn_x[i * 1600 + (m + 20) * 40 + (n + 20)] <= 0 && sgn_y[i * 1600 + (m + 20) * 40 + (n + 20)] < 0) {
						ori[i * 1600 + (m + 20) * 40 + (n + 20)] = ori[i * 1600 + (m + 20) * 40 + (n + 20)] * 10 - 180 - 5;
					}
					else {
						ori[i * 1600 + (m + 20) * 40 + (n + 20)] = (ori[i * 1600 + (m + 20) * 40 + (n + 20)] - 9) * 10 - 5;
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
}
void CalculateOrientationHistogram1(CSEM sem, double* ptext1, double* ptext2, double* hist, const vector<Mat>& gauss_pyr1, vector<Keypoint>& extrema1, double a_1, double b_1, double c_1, double t_1, double cc_1, double r_1) {
	int j = extrema1.size();
	double a1[1600 * j];
	double a2[1600 * j];
	double a3[1600 * j];
	double dx[1600 * j];
	double dy[1600 * j];
	double dx_dx_1[1600 * j];
	double dy_dy_1[1600 * j];
	double dx_dy_1[1600 * j];
	double y_x_1[1600 * j];
	double sgn_x[1600 * j];
	double sgn_y[1600 * j];
	int flag[1600 * j];
	int bins = ORI_HIST_BINS;
	double mag_1[1600 * j];
	double ori[1600 * j];
	double weight[j];
	int bin[j];
	double econs[j];
	int radius[j];
	const double PI2 = 2.0 * CV_PI;
	double power = 0.5;
	for (int i = 0; i < j; i++) {
		for (int k = 0; k < bins; k++) hist[i * bins + k] = 0.0;
		
		econs[i] = -1.0 / (2.0 * ORI_SIGMA_TIMES * extrema1[i].octave_scale * ORI_SIGMA_TIMES * extrema1[i].octave_scale);
		radius[i] = cvRound(ORI_WINDOW_RADIUS * extrema1[i].octave_scale);
		cout << "radius[i]:   " << radius[i] << endl;
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				flag[i * 1600 + (m + 20) * 40 + (n + 20)] = 1;
				if (extrema1[i].x + m > 0 && extrema1[i].x + m < gauss_pyr1[extrema1[i].octave * (INTERVALS + 3) + extrema1[i].interval].cols - 1 && extrema1[i].y + n > 0 && extrema1[i].y + n < gauss_pyr1[extrema1[i].octave * (INTERVALS + 3) + extrema1[i].interval].rows - 1) {
					pixel_t* data = (pixel_t*)gauss_pyr1[extrema1[i].octave * (INTERVALS + 3) + extrema1[i].interval].data;
					int step = gauss_pyr1[extrema1[i].octave * (INTERVALS + 3) + extrema1[i].interval].cols;
					dx[i * 1600 + (m + 20) * 40 + (n + 20)] = *(data + step * (extrema1[i].y + n) + extrema1[i].x + m + 1) - (*(data + step * (extrema1[i].y + n) + (extrema1[i].x + m - 1)));
					dy[i * 1600 + (m + 20) * 40 + (n + 20)] = *(data + step * (extrema1[i].y + n + 1) + extrema1[i].x + m) - (*(data + step * (extrema1[i].y + n - 1) + extrema1[i].x + m));
					secMul1_1(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, dx[i * 1600 + (m + 20) * 40 + (n + 20)], dx[i * 1600 + (m + 20) * 40 + (n + 20)]);
				}
				else {
					flag[i * 1600 + (m + 20) * 40 + (n + 20)] = 0;
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					secMul1_2(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
					dx_dx_1[i * 1600 + (m + 20) * 40 + (n + 20)] = a3[i * 1600 + (m + 20) * 40 + (n + 20)];
					secMul1_1(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, dy[i * 1600 + (m + 20) * 40 + (n + 20)], dy[i * 1600 + (m + 20) * 40 + (n + 20)]);
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					secMul1_2(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
					dy_dy_1[i * 1600 + (m + 20) * 40 + (n + 20)] = a3[i * 1600 + (m + 20) * 40 + (n + 20)];
					dx_dy_1[i * 1600 + (m + 20) * 40 + (n + 20)] = dx_dx_1[i * 1600 + (m + 20) * 40 + (n + 20)] + dy_dy_1[i * 1600 + (m + 20) * 40 + (n + 20)];
					secPRE1_1(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, dx_dy_1[i * 1600 + (m + 20) * 40 + (n + 20)], cc_1);
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					secPRE1_2(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, r_1, power);
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					secPRE1_3(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, a1, a2, a3, cc_1);
					mag_1[i * 1600 + (m + 20) * 40 + (n + 20)] = a1[i * 1600 + (m + 20) * 40 + (n + 20)];
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	double thresh = 0;
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					secCmp1_1(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, dx[i * 1600 + (m + 20) * 40 + (n + 20)], thresh, t_1);
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					secCmp1_2(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					secCmp1_3(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, a1, a2, a3);
					sgn_x[i * 1600 + (m + 20) * 40 + (n + 20)] = a1[i * 1600 + (m + 20) * 40 + (n + 20)];
					secCmp1_1(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, dy[i * 1600 + (m + 20) * 40 + (n + 20)], thresh, t_1);
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					secCmp1_2(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					secCmp1_3(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, a1, a2, a3);
					sgn_y[i * 1600 + (m + 20) * 40 + (n + 20)] = a1[i * 1600 + (m + 20) * 40 + (n + 20)];
					secDiv1_1(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, dy[i * 1600 + (m + 20) * 40 + (n + 20)], t_1);
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	double tx_1[1600*j];
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					secDiv1_2(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, tx_1, dx[i * 1600 + (m + 20) * 40 + (n + 20)], t_1);
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					secDiv1_3(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					secDiv1_4(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, a1, a2, a3, tx_1);
					y_x_1[i * 1600 + (m + 20) * 40 + (n + 20)] = a1[i * 1600 + (m + 20) * 40 + (n + 20)];
				}
			}
		}
	}
	ori1(j, radius, flag, sem, ptext1, ptext2, ori, a_1, b_1, c_1, sgn_x, sgn_y, y_x_1, t_1);
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					weight[i] = exp((m * m + n * n) * econs[i]);
					ori[i * 1600 + (m + 20) * 40 + (n + 20)] = ori[i * 1600 + (m + 20) * 40 + (n + 20)] * CV_PI / 180;
					bin[i] = cvFloor(bins * (CV_PI - ori[i * 1600 + (m + 20) * 40 + (n + 20)]) / PI2);
					bin[i] = bin[i] < bins ? bin[i] : 0;
					hist[i * bins + bin[i]] += mag_1[i * 1600 + (m + 20) * 40 + (n + 20)] * weight[i];
				}
			}
		}
		
	}
}
void CalculateOrientationHistogram2(CSEM sem, double* ptext1, double* ptext2, double* hist, const vector<Mat>& gauss_pyr2, vector<Keypoint>& extrema2, double a_2, double b_2, double c_2, double t_2, double cc_2, double r_2) {
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	int j = extrema2.size();
	double b1[1600 * j];
	double b2[1600 * j];
	double b3[1600 * j];
	double dx[1600 * j];
	double dy[1600 * j];
	double dx_dx_2[1600 * j];
	double dy_dy_2[1600 * j];
	double dx_dy_2[1600 * j];
	double sgn_x[1600 * j];
	double sgn_y[1600 * j];
	double y_x_2[1600 * j];
	int flag[1600 * j];
	int bins = ORI_HIST_BINS;
	double mag_2[1600 * j];
	double ori[1600 * j];
	double weight[j];
	int bin[j];
	double econs[j];
	int radius[j];
	const double PI2 = 2.0 * CV_PI;
	double power = 0.5;
	for (int i = 0; i < j; i++) {
		for (int k = 0; k < bins; k++) hist[i * bins + k] = 0.0;
		econs[i] = -1.0 / (2.0 * ORI_SIGMA_TIMES * extrema2[i].octave_scale * ORI_SIGMA_TIMES * extrema2[i].octave_scale);
		radius[i] = cvRound(ORI_WINDOW_RADIUS * extrema2[i].octave_scale);
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				flag[i * 1600 + (m + 20) * 40 + (n + 20)] = 1;
				if (extrema2[i].x + m > 0 && extrema2[i].x + m < gauss_pyr2[extrema2[i].octave * (INTERVALS + 3) + extrema2[i].interval].cols - 1 && extrema2[i].y + n > 0 && extrema2[i].y + n < gauss_pyr2[extrema2[i].octave * (INTERVALS + 3) + extrema2[i].interval].rows - 1) {
					pixel_t* data = (pixel_t*)gauss_pyr2[extrema2[i].octave * (INTERVALS + 3) + extrema2[i].interval].data;
					int step = gauss_pyr2[extrema2[i].octave * (INTERVALS + 3) + extrema2[i].interval].cols;
					dx[i * 1600 + (m + 20) * 40 + (n + 20)] = *(data + step * (extrema2[i].y + n) + (extrema2[i].x + m + 1)) - (*(data + step * (extrema2[i].y + n) + (extrema2[i].x + m - 1)));
					dy[i * 1600 + (m + 20) * 40 + (n + 20)] = *(data + step * (extrema2[i].y + n + 1) + extrema2[i].x + m) - (*(data + step * (extrema2[i].y + n - 1) + extrema2[i].x + m));
					secMul2_1(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, dx[i * 1600 + (m + 20) * 40 + (n + 20)], dx[i * 1600 + (m + 20) * 40 + (n + 20)]);
					dx_dx_2[i * 1600 + (m + 20) * 40 + (n + 20)] = b3[i * 1600 + (m + 20) * 40 + (n + 20)];
				}
				else {
					flag[i * 1600 + (m + 20) * 40 + (n + 20)] = 0;
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					secMul2_1(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, dy[i * 1600 + (m + 20) * 40 + (n + 20)], dy[i * 1600 + (m + 20) * 40 + (n + 20)]);
					dy_dy_2[i * 1600 + (m + 20) * 40 + (n + 20)] = b3[i * 1600 + (m + 20) * 40 + (n + 20)];
					dx_dy_2[i * 1600 + (m + 20) * 40 + (n + 20)] = dx_dx_2[i * 1600 + (m + 20) * 40 + (n + 20)] + dy_dy_2[i * 1600 + (m + 20) * 40 + (n + 20)];
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					secPRE2_1(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, dx_dy_2[i * 1600 + (m + 20) * 40 + (n + 20)], cc_2);
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					secPRE2_2(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, r_2, power);
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					secPRE2_3(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, b1, b2, b3, cc_2);
					mag_2[i * 1600 + (m + 20) * 40 + (n + 20)] = b1[i * 1600 + (m + 20) * 40 + (n + 20)];
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	double thresh = 0;
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					secCmp2_1(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, dx[i * 1600 + (m + 20) * 40 + (n + 20)], thresh, t_2);
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					secCmp2_2(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, b1, b2, b3);
					sgn_x[i * 1600 + (m + 20) * 40 + (n + 20)] = b1[i * 1600 + (m + 20) * 40 + (n + 20)];
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					secCmp2_1(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, dy[i * 1600 + (m + 20) * 40 + (n + 20)], thresh, t_2);
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					secCmp2_2(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, b1, b2, b3);
					sgn_y[i * 1600 + (m + 20) * 40 + (n + 20)] = b1[i * 1600 + (m + 20) * 40 + (n + 20)];
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	double tx_2[1600 * j];
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					secDiv2_1(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, tx_2, dy[i * 1600 + (m + 20) * 40 + (n + 20)], t_2);
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					secDiv2_2(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, dx[i * 1600 + (m + 20) * 40 + (n + 20)], t_2);
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					secDiv2_3(i * 1600 + (m + 20) * 40 + (n + 20), ptext1, ptext2, b1, b2, b3, tx_2);
					y_x_2[i * 1600 + (m + 20) * 40 + (n + 20)] = b1[i * 1600 + (m + 20) * 40 + (n + 20)];
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	ori2(j, radius, flag, sem, ptext1, ptext2, ori, a_2, b_2, c_2, sgn_x, sgn_y, y_x_2, t_2);
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 1600 + (m + 20) * 40 + (n + 20)] == 1) {
					weight[i] = exp((m * m + n * n) * econs[i]);
					ori[i * 1600 + (m + 20) * 40 + (n + 20)] = ori[i * 1600 + (m + 20) * 40 + (n + 20)] * CV_PI / 180;
					bin[i] = cvFloor(bins * (CV_PI - ori[i * 1600 + (m + 20) * 40 + (n + 20)]) / PI2);
					bin[i] = bin[i] < bins ? bin[i] : 0;
					hist[i * bins + bin[i]] += mag_2[i * 1600 + (m + 20) * 40 + (n + 20)] * weight[i];
				}
			}
		}
	}
	
}
void GaussSmoothOriHist(double* hist, vector<Keypoint>& extrema,int n ) {
	int j = extrema.size();
	for (int i = 0; i < j; i++) {
		
		for (int k = 0; k < ORI_SMOOTH_TIMES; k++) {
			double prev = hist[i * n + n - 1];
			double temp;
			double h0 = hist[i * n];
			
			for (int l = 0; l < n; l++) {
				temp = hist[i * n + l];
				hist[i * n + l] = 0.25 * prev + 0.5 * hist[i * n + l] + 0.25 * (l+ 1 >= n ? h0 : hist[(i * n + l) + 1]);
				prev = temp;
			}
		}
	}
}
void DominantDirection1(CSEM sem, double* ptext1, double* ptext2, double* hist, double* maxd, vector<Keypoint>& extrema1, double a_1, double b_1, double c_1, double t_1, int n) {
	int j = extrema1.size();
	double a1[j];
	double a2[j];
	double a3[j];
	for (int i = 0; i < j; i++) maxd[i] = hist[i * n];

	for (int m = 1; m < n; m++) {
		for (int i = 0; i < j; i++) {
			
			secCmp1_1(i, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, hist[i * n + m], maxd[i], t_1);
			
		}
		if (sem.post() == false)  ;//printf("sem.post failed.\n");
		//else printf("A  sem.post ok\n");
		if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
		//else printf("A  sem.wait ok\n");
		for (int i = 0; i < j; i++) {
			//cout << "i:" << i << "   hist:" << ptext1[i] << endl;
			secCmp1_2(i, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
		}
		if (sem.post() == false)  ;//printf("sem.post failed.\n");
		//else printf("A  sem.post ok\n");
		if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
		//else printf("A  sem.wait ok\n");
		for (int i = 0; i < j; i++) {
			
			secCmp1_3(i, ptext1, ptext2, a1, a2, a3);
			if(a1[i]==1) maxd[i] = hist[i * n + m];
			
		}
	}
}
void DominantDirection2(CSEM sem, double* ptext1, double* ptext2, double* hist, double* maxd, vector<Keypoint>& extrema2, double a_2, double b_2, double c_2, double t_2, int n) {
	int j = extrema2.size();
	double b1[j];
	double b2[j];
	double b3[j];
	for (int i = 0; i < j; i++) maxd[i] = hist[i * n];
	for (int m = 1; m < n; m++) {
		if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
		//else printf("B  sem.wait ok\n");
		for (int i = 0; i < j; i++) {
			secCmp2_1(i, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, hist[i * n + m], maxd[i], t_2);
			//cout << "i:" << i << "   hist:" << ptext1[i] << endl;
		}
		if (sem.post() == false)  ;//printf("sem.post failed.\n");
		//else printf("B  sem.post ok\n");
		if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
		//else printf("B  sem.wait ok\n");
		for (int i = 0; i < j; i++) {
			secCmp2_2(i, ptext1, ptext2, b1, b2, b3);
			if(b1[i] ==1.0) maxd[i] = hist[i * n + m];
		}
		if (sem.post() == false)  ;//printf("sem.post failed.\n");
		//else printf("B  sem.post ok\n");
	}
}
void CopyKeypoint(const Keypoint& src, Keypoint& dst) {
	dst.dx = src.dx;
	dst.dy = src.dy;

	dst.interval = src.interval;
	dst.octave = src.octave;
	dst.octave_scale = src.octave_scale;
	dst.offset_interval = src.offset_interval;

	dst.offset_x = src.offset_x;
	dst.offset_y = src.offset_y;

	dst.ori = src.ori;
	dst.scale = src.scale;
	dst.val = src.val;
	dst.x = src.x;
	dst.y = src.y;
}
void CalcOriFeatures1(CSEM sem, double* ptext1, double* ptext2, double* hist, vector<Keypoint>& extrema1, vector<Keypoint>& features1, double a_1, double b_1, double c_1, double t_1, int n, double* highest_peak) {
	double  PI2 = CV_PI * 2.0;
	int j = extrema1.size();
	double  bin[j * n];
	double  mag_thr[j];
	double  x_1;
	double  y_1;
	double  tx_1[j * n * 3];
	double a1[j * n * 3];
	double a2[j * n * 3];
	double a3[j * n * 3];
	int flag[j * n];
	int l;
	int r;
	for (int i = 0; i < j; i++) {
		mag_thr[i] = highest_peak[i] * ORI_PEAK_RATIO;
		for (int m = 0; m < n; m++) {
			flag[i * n + m]=1;
			l = (m == 0) ? n - 1 : m - 1; 
			r = (m + 1) % n;
			secCmp1_1(i * n * 3 + m * 3, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, hist[i * n + m], hist[i * n + l], t_1);
			secCmp1_1(i * n * 3 + m * 3 + 1, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, hist[i * n + m], hist[i * n + r], t_1);
			secCmp1_1(i * n * 3 + m * 3 + 2, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, hist[i * n + m], mag_thr[i], t_1);
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = 0; m < n; m++) {
			l = (m == 0) ? n - 1 : m - 1;
			r = (m + 1) % n;
			secCmp1_2(i * n * 3 + m * 3, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
			secCmp1_2(i * n * 3 + m * 3 + 1, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
			secCmp1_2(i * n * 3 + m * 3 + 2, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = 0; m < n; m++) {
			l = (m == 0) ? n - 1 : m - 1;
			r = (m + 1) % n;
			secCmp1_3(i * n * 3 + m * 3, ptext1, ptext2, a1, a2, a3);
			secCmp1_3(i * n * 3 + m * 3 + 1, ptext1, ptext2, a1, a2, a3);
			secCmp1_3(i * n * 3 + m * 3 + 2, ptext1, ptext2, a1, a2, a3);
			if (a1[i * n * 3 + m * 3] == 1 && a1[i * n * 3 + m * 3 + 1] == 1 && a1[i * n * 3 + m * 3 + 2] >= 0) {
				x_1 = hist[i * n + l] - hist[i * n + r];
				secDiv1_1(i * n * 3 + m * 3, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, x_1, t_1);
				flag[i * n + m] = 0;
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = 0; m < n; m++) {
			l = (m == 0) ? n - 1 : m - 1;
			r = (m + 1) % n;
			if (flag[i * n + m] == 0) {
				y_1 = hist[i * n + l] - 2 * hist[i * n + m] + hist[i * n + r];
				secDiv1_2(i * n * 3 + m * 3, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, tx_1,y_1, t_1);
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = 0; m < n; m++) {
			if (flag[i * n + m] == 0) {
				secDiv1_3(i * n * 3 + m * 3, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = 0; m < n; m++) {
			if (flag[i * n + m] == 0) {
				secDiv1_4(i * n * 3 + m * 3, ptext1, ptext2, a1, a2, a3,tx_1);
				
				ptext1[i * n * 3 + m * 3] = a1[i * n * 3 + m * 3];
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = 0; m < n; m++) {
			if (flag[i * n + m] == 0) {
				a2[i * n * 3 + m * 3] = ptext1[i * n * 3 + m * 3] + a1[i * n * 3 + m * 3];
				bin[i * n + m] = m + 0.5f * a2[i * n * 3 + m * 3];
				bin[i * n + m] = (bin[i * n + m] < 0) ? (bin[i * n + m] + n) : (bin[i * n + m] >= n ? (bin[i * n + m] - n) : bin[i * n + m]);
				Keypoint new_key;
				CopyKeypoint(extrema1[i], new_key);
				new_key.ori = ((PI2 * bin[i * n + m]) / n) - CV_PI;
				new_key.ori = cvFloor(new_key.ori * 180 / CV_PI) + 5;  
				new_key.ori = new_key.ori * CV_PI / 180;
				features1.push_back(new_key);
			}
		}
	}
}
void CalcOriFeatures2(CSEM sem, double* ptext1, double* ptext2, double* hist, vector<Keypoint>& extrema2, vector<Keypoint>& features2, double a_2, double b_2, double c_2, double t_2, int n, double* highest_peak) {
	double  PI2 = CV_PI * 2.0;
	int j = extrema2.size();
	double  bin[j * n];
	double  mag_thr[j];
	double  x_2;
	double  y_2;
	double  tx_2[j * n * 3];
	double b1[j * n * 3];
	double b2[j * n * 3];
	double b3[j * n * 3];
	int flag[j * n];
	int l;
	int r;
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		mag_thr[i] = highest_peak[i] * ORI_PEAK_RATIO;
		for (int m = 0; m < n; m++) {
			flag[i * n + m] = 1;
			l = (m == 0) ? n - 1 : m - 1;
			r = (m + 1) % n;
			secCmp2_1(i * n * 3 + m * 3, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, hist[i * n + m], hist[i * n + l], t_2);
			secCmp2_1(i * n * 3 + m * 3+1, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, hist[i * n + m], hist[i * n + r], t_2);
			secCmp2_1(i * n * 3 + m * 3+2, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, hist[i * n + m], mag_thr[i], t_2);
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = 0; m < n; m++) {
			secCmp2_2(i * n * 3 + m * 3, ptext1, ptext2, b1, b2, b3);
			secCmp2_2(i * n * 3 + m * 3 + 1, ptext1, ptext2, b1, b2, b3);
			secCmp2_2(i * n * 3 + m * 3 + 2, ptext1, ptext2, b1, b2, b3);
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = 0; m < n; m++) {
			l = (m == 0) ? n - 1 : m - 1;
			r = (m + 1) % n;
			if (b1[i * n * 3 + m * 3] == 1 && b1[i * n * 3 + m * 3 + 1] == 1 && b1[i * n * 3 + m * 3 + 2] >= 0) {
				x_2 = hist[i * n + l] - hist[i * n + r];
				secDiv2_1(i * n * 3 + m * 3, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, tx_2,x_2, t_2);
				flag[i * n + m] = 0;
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = 0; m < n; m++) {
			l = (m == 0) ? n - 1 : m - 1;
			r = (m + 1) % n;
			if (flag[i * n + m] == 0) {
				y_2 = hist[i * n + l] - 2 * hist[i * n + m] + hist[i * n + r];
				secDiv2_2(i * n * 3 + m * 3, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, y_2, t_2);
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = 0; m < n; m++) {
			if (flag[i * n + m] == 0) {
				secDiv2_3(i * n * 3 + m * 3, ptext1, ptext2, b1, b2, b3, tx_2);
				
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = 0; m < n; m++) {
			if (flag[i * n + m] == 0) {
				b2[i * n * 3 + m * 3] = b1[i * n * 3 + m * 3] + ptext1[i * n * 3 + m * 3];
				ptext1[i * n * 3 + m * 3] = b1[i * n * 3 + m * 3];
				bin[i * n + m] = m + 0.5f * b2[i * n * 3 + m * 3];
				bin[i * n + m] = (bin[i * n + m] < 0) ? (bin[i * n + m] + n) : (bin[i * n + m] >= n ? (bin[i * n + m] - n) : bin[i * n + m]);
				Keypoint new_key;
				CopyKeypoint(extrema2[i], new_key);
				new_key.ori = ((PI2 * bin[i * n + m]) / n) - CV_PI;
				new_key.ori = cvFloor(new_key.ori * 180 / CV_PI) + 5;
				new_key.ori = new_key.ori * CV_PI / 180;
				features2.push_back(new_key);
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
}
void OrientationAssignment1(CSEM sem, double* ptext1, double* ptext2, vector<Keypoint>& extrema1, vector<Keypoint>& features1, const vector<Mat>& gauss_pyr1, double a_1, double b_1, double c_1, double t_1, double cc_1, double r_1) {
	double  hist[extrema1.size() * ORI_HIST_BINS];
	double highest_peak_1[extrema1.size()];

	CalculateOrientationHistogram1(sem, ptext1, ptext2, hist, gauss_pyr1, extrema1, a_1, b_1, c_1, t_1, cc_1, r_1);
	GaussSmoothOriHist(hist, extrema1, ORI_HIST_BINS);
	DominantDirection1(sem, ptext1, ptext2, hist, highest_peak_1, extrema1, a_1, b_1, c_1, t_1, ORI_HIST_BINS);
	CalcOriFeatures1(sem, ptext1, ptext2, hist, extrema1, features1, a_1, b_1, c_1, t_1, ORI_HIST_BINS, highest_peak_1);
	//for (int i = 0; i < features1.size(); i++) cout << "1111x:" << features1[i].x << "  y:" << features1[i].y << "  ori:" << features1[i].ori << endl;
	
}
void OrientationAssignment2(CSEM sem, double* ptext1, double* ptext2, vector<Keypoint>& extrema2, vector<Keypoint>& features2, const vector<Mat>& gauss_pyr2, double a_2, double b_2, double c_2, double t_2, double cc_2, double r_2) {
	double  hist[extrema2.size() * ORI_HIST_BINS];
	double highest_peak_2[extrema2.size()];
	CalculateOrientationHistogram2(sem, ptext1, ptext2, hist, gauss_pyr2, extrema2, a_2, b_2, c_2, t_2, cc_2, r_2);

	GaussSmoothOriHist(hist, extrema2, ORI_HIST_BINS);
	DominantDirection2(sem, ptext1, ptext2, hist, highest_peak_2, extrema2, a_2, b_2, c_2, t_2, ORI_HIST_BINS);
	
	CalcOriFeatures2(sem, ptext1, ptext2, hist, extrema2, features2, a_2, b_2, c_2, t_2, ORI_HIST_BINS, highest_peak_2);
	//for (int i = 0; i < features2.size(); i++) cout << "2222x:" << features2[i].x << "  y:" << features2[i].y << "ori:" << features2[i].ori << endl;
	
}

void DescriptorOri_1(int j, int* radius, int* flag, int* flag1, CSEM sem, double* ptext1, double* ptext2, double* ori, double a_1, double b_1, double c_1, double* sgn_x, double* sgn_y, double* x_1, double t_1) {
	double a1[3600 * j * 9];
	double a2[3600 * j * 9];
	double a3[3600 * j * 9];
	double flag1_1[3600 * j];
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag1[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						flag1_1[i * 3600 + (m + 30) * 60 + (n + 30)] = 1;
						if (sgn_x[i * 3600 + (m + 30) * 60 + (n + 30)] >= 0 && sgn_y[i * 3600 + (m + 30) * 60 + (n + 30)] > 0) {
							double y_1[9] = { tan(CV_PI / 18), tan(CV_PI / 9), tan(CV_PI / 6), tan(CV_PI / 4.5) , tan(CV_PI / 3.6) , tan(CV_PI / 3) , tan(7 * CV_PI / 18) , tan(CV_PI / 2.25) , tan(CV_PI / 2) };
							for (int k = 0; k < 9; k++) {
								a1[i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k] = x_1[i * 3600 + (m + 30) * 60 + (n + 30)] - y_1[k];
								secMul1_1(i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, t_1, a1[i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k]);
							}
						}
						else if (sgn_x[i * 3600 + (m + 30) * 60 + (n + 30)] < 0 && sgn_y[i * 3600 + (m + 30) * 60 + (n + 30)] >= 0) {
							double y_1[9] = { tan(CV_PI / 1.8), tan(11 * CV_PI / 18) , tan(CV_PI / 1.5) , tan(13 * CV_PI / 18) , tan(14 * CV_PI / 18) , tan(CV_PI / 1.2), tan(16 * CV_PI / 18), tan(17 * CV_PI / 18), tan(18 * CV_PI / 18) };
							for (int k = 0; k < 9; k++) {
								a1[i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k] = x_1[i * 3600 + (m + 30) * 60 + (n + 30)] - y_1[k];
								secMul1_1(i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, t_1, a1[i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k]);
							}
						}
						else if (sgn_x[i * 3600 + (m + 30) * 60 + (n + 30)] <= 0 && sgn_y[i * 3600 + (m + 30) * 60 + (n + 30)] < 0) {
							double y_1[9] = { tan(1 * CV_PI / 18), tan(2 * CV_PI / 18), tan(3 * CV_PI / 18), tan(4 * CV_PI / 18) , tan(5 * CV_PI / 18) , tan(6 * CV_PI / 18) , tan(7 * CV_PI / 18) , tan(8 * CV_PI / 18) ,tan(9 * CV_PI / 18) };
							for (int k = 0; k < 9; k++) {
								a1[i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k] = x_1[i * 3600 + (m + 30) * 60 + (n + 30)] - y_1[k];
								secMul1_1(i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, t_1, a1[i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k]);
							}
						}
						else {
							double y_1[9] = { tan(28 * CV_PI / 18), tan(29 * CV_PI / 18), tan(30 * CV_PI / 18), tan(31 * CV_PI / 18), tan(32 * CV_PI / 18), tan(33 * CV_PI / 18), tan(34 * CV_PI / 18), tan(35 * CV_PI / 18), tan(36 * CV_PI / 18) };
							for (int k = 0; k < 9; k++) {
								a1[i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k] = x_1[i * 3600 + (m + 30) * 60 + (n + 30)] - y_1[k];
								secMul1_1(i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, t_1, a1[i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k]);
							}
						}
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag1[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						for (int k = 0; k < 9; k++) {
							secMul1_2(i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
							ptext1[i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k] = a3[i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k];
						}
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	int k;
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag1[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						for (k = 0; k < 9; k++) {
							a1[i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k] = sgn(a3[i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k] + ptext1[i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k]);
							if (flag1_1[i * 3600 + (m + 30) * 60 + (n + 30)] == 1) {
								if (a1[i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k] == -1) {
									ori[i * 3600 + (m + 30) * 60 + (n + 30)] = k + 1;
									flag1_1[i * 3600 + (m + 30) * 60 + (n + 30)] = 0;
								}
							}
						}
						if (flag1_1[i * 3600 + (m + 30) * 60 + (n + 30)] == 1) ori[i * 3600 + (m + 30) * 60 + (n + 30)] = k;

						if (sgn_x[i * 3600 + (m + 30) * 60 + (n + 30)] >= 0 && sgn_y[i * 3600 + (m + 30) * 60 + (n + 30)] > 0) {
							ori[i * 3600 + (m + 30) * 60 + (n + 30)] = ori[i * 3600 + (m + 30) * 60 + (n + 30)] * 10 - 5;
						}
						else if (sgn_x[i * 3600 + (m + 30) * 60 + (n + 30)] < 0 && sgn_y[i * 3600 + (m + 30) * 60 + (n + 30)] >= 0) {
							ori[i * 3600 + (m + 30) * 60 + (n + 30)] = (ori[i * 3600 + (m + 30) * 60 + (n + 30)] + 9) * 10 - 5;
						}
						else if (sgn_x[i * 3600 + (m + 30) * 60 + (n + 30)] <= 0 && sgn_y[i * 3600 + (m + 30) * 60 + (n + 30)] < 0) {
							ori[i * 3600 + (m + 30) * 60 + (n + 30)] = ori[i * 3600 + (m + 30) * 60 + (n + 30)] * 10 - 180 - 5;
						}
						else {
							ori[i * 3600 + (m + 30) * 60 + (n + 30)] = (ori[i * 3600 + (m + 30) * 60 + (n + 30)] - 9) * 10 - 5;
						}

					}
				}
			}
		}
	}
}
void DescriptorOri_2(int j, int* radius, int* flag, int* flag2, CSEM sem, double* ptext1, double* ptext2, double* ori, double a_2, double b_2, double c_2, double* sgn_x, double* sgn_y, double* x_2, double t_2) {
	double b1[3600 * j * 9];
	double b2[3600 * j * 9];
	double b3[3600 * j * 9];
	double flag2_2[3600 * j];
	double y_2[9] = { 0 };
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag2[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						flag2_2[i * 3600 + (m + 30) * 60 + (n + 30)] = 1;
						for (int k = 0; k < 9; k++) {
							b1[i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k] = x_2[i * 3600 + (m + 30) * 60 + (n + 30)] - y_2[k];
							secMul2_1(i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, t_2, b1[i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k]);
						}
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	int k;
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag2[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						for (k = 0; k < 9; k++) {
							b1[i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k] = sgn(b3[i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k] + ptext1[i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k]);
							ptext1[i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k] = b3[i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k];
							if (flag2_2[i * 3600 + (m + 30) * 60 + (n + 30)] == 1) {
								if (b1[i * 3600 * 9 + (m + 30) * 60 * 9 + (n + 30) * 9 + k] == -1) {
									ori[i * 3600 + (m + 30) * 60 + (n + 30)] = k + 1;
									flag2_2[i * 3600 + (m + 30) * 60 + (n + 30)] = 0;
								}
							}
						}
						if (flag2_2[i * 3600 + (m + 30) * 60 + (n + 30)] == 1) ori[i * 3600 + (m + 30) * 60 + (n + 30)] = k;
						if (sgn_x[i * 3600 + (m + 30) * 60 + (n + 30)] >= 0 && sgn_y[i * 3600 + (m + 30) * 60 + (n + 30)] > 0) {
							ori[i * 3600 + (m + 30) * 60 + (n + 30)] = ori[i * 3600 + (m + 30) * 60 + (n + 30)] * 10 - 5;
						}
						else if (sgn_x[i * 3600 + (m + 30) * 60 + (n + 30)] < 0 && sgn_y[i * 3600 + (m + 30) * 60 + (n + 30)] >= 0) {
							ori[i * 3600 + (m + 30) * 60 + (n + 30)] = (ori[i * 3600 + (m + 30) * 60 + (n + 30)] + 9) * 10 - 5;
						}
						else if (sgn_x[i * 3600 + (m + 30) * 60 + (n + 30)] <= 0 && sgn_y[i * 3600 + (m + 30) * 60 + (n + 30)] < 0) {
							ori[i * 3600 + (m + 30) * 60 + (n + 30)] = ori[i * 3600 + (m + 30) * 60 + (n + 30)] * 10 - 180 - 5;
						}
						else {
							ori[i * 3600 + (m + 30) * 60 + (n + 30)] = (ori[i * 3600 + (m + 30) * 60 + (n + 30)] - 9) * 10 - 5;
						}
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
}
void CalculateDescrHist1(CSEM sem, double* ptext1, double* ptext2, double* hist, vector<Keypoint>& features1, const vector<Mat>& gauss_pyr1, double a_1, double b_1, double c_1, double t_1, double cc_1, double r_1, int bins, int width) {
	int j = features1.size();
	double cos_ori[j];
	double sin_ori[j];
	double sub_hist_width[j];
	int    radius[j];
	double grad_ori[3600 * j];
	double grad_mag[3600 * j];
	double rot_x[3600 * j];
	double rot_y[3600 * j];
	double xbin[3600 * j];
	double ybin[3600 * j];
	double obin[3600 * j];
	double weight[3600 * j];

	double a1[3600 * j];
	double a2[3600 * j];
	double a3[3600 * j];
	double dx[3600 * j];
	double dy[3600 * j];
	double dx_dx_1[3600 * j];
	double dy_dy_1[3600 * j];
	double dx_dy_1[3600 * j];
	double y_x_1[3600 * j];
	double sgn_x[3600 * j];
	double sgn_y[3600 * j];
	int flag[3600 * j];
	int flag1[3600 * j];
	double sigma = 0.5 * width;
	double conste = -1.0 / (2 * sigma * sigma);
	double PI2 = CV_PI * 2;
	double power = 0.5;
	
	for (int i = 0; i < j; i++) {
		sub_hist_width[i] = DESCR_SCALE_ADJUST * features1[i].octave_scale;
		radius[i] = (sub_hist_width[i] * sqrt(2.0) * (width + 1)) / 2.0 + 0.5;
		cout << "des  radius[i]:   " << radius[i] << endl;
		cos_ori[i] = cos(features1[i].ori);
		sin_ori[i] = sin(features1[i].ori);
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				flag[i * 3600 + (m + 30) * 60 + (n + 30)] = 1;
				flag1[i * 3600 + (m + 30) * 60 + (n + 30)] = 1;
				rot_x[i * 3600 + (m + 30) * 60 + (n + 30)] = (cos_ori[i] * n - sin_ori[i] * m) / sub_hist_width[i];
				rot_y[i * 3600 + (m + 30) * 60 + (n + 30)] = (sin_ori[i] * n + cos_ori[i] * m) / sub_hist_width[i];
				xbin[i * 3600 + (m + 30) * 60 + (n + 30)] = rot_x[i * 3600 + (m + 30) * 60 + (n + 30)] + width / 2 - 0.5;
				ybin[i * 3600 + (m + 30) * 60 + (n + 30)] = rot_y[i * 3600 + (m + 30) * 60 + (n + 30)] + width / 2 - 0.5;
				if (xbin[i * 3600 + (m + 30) * 60 + (n + 30)] > -1.0 && xbin[i * 3600 + (m + 30) * 60 + (n + 30)] < width && ybin[i * 3600 + (m + 30) * 60 + (n + 30)] > -1.0 && ybin[i * 3600 + (m + 30) * 60 + (n + 30)] < width) {
					flag[i * 3600 + (m + 30) * 60 + (n + 30)] = 0;
					if (features1[i].x+n > 0 && features1[i].x + n < gauss_pyr1[features1[i].octave * (INTERVALS + 3) + features1[i].interval].cols - 1 && features1[i].y+m > 0 && features1[i].y + m < gauss_pyr1[features1[i].octave * (INTERVALS + 3) + features1[i].interval].rows - 1) {
						flag1[i * 3600 + (m + 30) * 60 + (n + 30)] = 0;
						pixel_t* data = (pixel_t*)gauss_pyr1[features1[i].octave * (INTERVALS + 3) + features1[i].interval].data;
						int step = gauss_pyr1[features1[i].octave * (INTERVALS + 3) + features1[i].interval].cols;
						dx[i * 3600 + (m + 30) * 60 + (n + 30)] = *(data + step * (features1[i].y + m) + (features1[i].x + n + 1)) - (*(data + step * (features1[i].y + m) + (features1[i].x + n - 1)));
						dy[i * 3600 + (m + 30) * 60 + (n + 30)] = *(data + step * (features1[i].y + m + 1) + features1[i].x + n) - (*(data + step * (features1[i].y + m - 1) + features1[i].x + n));
						secMul1_1(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, dx[i * 3600 + (m + 30) * 60 + (n + 30)], dx[i * 3600 + (m + 30) * 60 + (n + 30)]);
					}
				}
				
			}
			
		}
		
	}
	
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++) {
			for (int n = -radius[i]; n <= radius[i]; n++) {
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag1[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						secMul1_2(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
						dx_dx_1[i * 3600 + (m + 30) * 60 + (n + 30)] = a3[i * 3600 + (m + 30) * 60 + (n + 30)];
						secMul1_1(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, dy[i * 3600 + (m + 30) * 60 + (n + 30)], dy[i * 3600 + (m + 30) * 60 + (n + 30)]);
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++) {
			for (int n = -radius[i]; n <= radius[i]; n++) {
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag1[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						secMul1_2(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
						dy_dy_1[i * 3600 + (m + 30) * 60 + (n + 30)] = a3[i * 3600 + (m + 30) * 60 + (n + 30)];
						dx_dy_1[i * 3600 + (m + 30) * 60 + (n + 30)] = dx_dx_1[i * 3600 + (m + 30) * 60 + (n + 30)] + dy_dy_1[i * 3600 + (m + 30) * 60 + (n + 30)];
						secPRE1_1(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, dx_dy_1[i * 3600 + (m + 30) * 60 + (n + 30)], cc_1);
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++) {
			for (int n = -radius[i]; n <= radius[i]; n++) {
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag1[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						secPRE1_2(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, r_1, power);
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++) {
			for (int n = -radius[i]; n <= radius[i]; n++) {
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag1[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						secPRE1_3(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, a1, a2, a3, cc_1);
						grad_mag[i * 3600 + (m + 30) * 60 + (n + 30)] = a1[i * 3600 + (m + 30) * 60 + (n + 30)];
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	double thresh = 0;
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++) {
			for (int n = -radius[i]; n <= radius[i]; n++) {
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag1[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						secCmp1_1(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, dx[i * 3600 + (m + 30) * 60 + (n + 30)], thresh, t_1);
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++) {
			for (int n = -radius[i]; n <= radius[i]; n++) {
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag1[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						secCmp1_2(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++) {
			for (int n = -radius[i]; n <= radius[i]; n++) {
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag1[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						secCmp1_3(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, a1, a2, a3);
						sgn_x[i * 3600 + (m + 30) * 60 + (n + 30)] = a1[i * 3600 + (m + 30) * 60 + (n + 30)];
						secCmp1_1(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, dy[i * 3600 + (m + 30) * 60 + (n + 30)], thresh, t_1);
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++) {
			for (int n = -radius[i]; n <= radius[i]; n++) {
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag1[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						secCmp1_2(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++) {
			for (int n = -radius[i]; n <= radius[i]; n++) {
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag1[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						secCmp1_3(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, a1, a2, a3);
						sgn_y[i * 3600 + (m + 30) * 60 + (n + 30)] = a1[i * 3600 + (m + 30) * 60 + (n + 30)];
						secDiv1_1(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, dy[i * 3600 + (m + 30) * 60 + (n + 30)], t_1);
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	double tx_1[3600 * j];
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++) {
			for (int n = -radius[i]; n <= radius[i]; n++) {
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag1[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						secDiv1_2(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, tx_1,dx[i * 3600 + (m + 30) * 60 + (n + 30)], t_1);
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++) {
			for (int n = -radius[i]; n <= radius[i]; n++) {
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag1[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						secDiv1_3(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++) {
			for (int n = -radius[i]; n <= radius[i]; n++) {
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag1[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						secDiv1_4(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, a1, a2, a3,tx_1);
						y_x_1[i * 3600 + (m + 30) * 60 + (n + 30)] = a1[i * 3600 + (m + 30) * 60 + (n + 30)];

					}
				}
			}
		}
	}
	
	DescriptorOri_1(j, radius, flag, flag1, sem, ptext1, ptext2, grad_ori, a_1, b_1, c_1, sgn_x, sgn_y, y_x_1, t_1);
	double mag[3600 * j];
	for (int i = 0; i < j; i++) {
		for (int l = 0; l < width * width * bins; l++) hist[i * width * width * bins + l] = 0.0;
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					
					if (flag1[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						grad_ori[i * 3600 + (m + 30) * 60 + (n + 30)] = grad_ori[i * 3600 + (m + 30) * 60 + (n + 30)] * CV_PI / 180;
						grad_ori[i * 3600 + (m + 30) * 60 + (n + 30)] = (CV_PI - grad_ori[i * 3600 + (m + 30) * 60 + (n + 30)]) - features1[i].ori;
						while (grad_ori[i * 3600 + (m + 30) * 60 + (n + 30)] < 0.0)
							grad_ori[i * 3600 + (m + 30) * 60 + (n + 30)] += PI2;
						while (grad_ori[i * 3600 + (m + 30) * 60 + (n + 30)] >= PI2)
							grad_ori[i * 3600 + (m + 30) * 60 + (n + 30)] -= PI2;
						obin[i * 3600 + (m + 30) * 60 + (n + 30)] = grad_ori[i * 3600 + (m + 30) * 60 + (n + 30)] * (bins / PI2);
						weight[i * 3600 + (m + 30) * 60 + (n + 30)] = exp(conste * (rot_x[i * 3600 + (m + 30) * 60 + (n + 30)] * rot_x[i * 3600 + (m + 30) * 60 + (n + 30)] + rot_y[i * 3600 + (m + 30) * 60 + (n + 30)] * rot_y[i * 3600 + (m + 30) * 60 + (n + 30)]));
						InterpHistEntry(i,hist, xbin[i * 3600 + (m + 30) * 60 + (n + 30)], ybin[i * 3600 + (m + 30) * 60 + (n + 30)], obin[i * 3600 + (m + 30) * 60 + (n + 30)], grad_mag[i * 3600 + (m + 30) * 60 + (n + 30)] * weight[i * 3600 + (m + 30) * 60 + (n + 30)], bins, width);
					}
				}
			}
		}
		for (int l = 0; l < width * width * bins; l++)
			features1[i].descriptor[l] = hist[i * width * width * bins + l];
	}
}
void CalculateDescrHist2(CSEM sem, double* ptext1, double* ptext2, double* hist, vector<Keypoint>& features2, const vector<Mat>& gauss_pyr2, double a_2, double b_2, double c_2, double t_2, double cc_2, double r_2, int bins, int width) {
	int j = features2.size();
	double cos_ori[j];
	double sin_ori[j];
	double sub_hist_width[j];
	int    radius[j];
	double grad_ori[3600 * j];
	double grad_mag[3600 * j];
	double rot_x[3600 * j];
	double rot_y[3600 * j];
	double xbin[3600 * j];;
	double ybin[3600 * j];
	double obin[3600 * j];
	double weight[3600 * j];

	double b1[3600 * j];
	double b2[3600 * j];
	double b3[3600 * j];
	double dx[3600 * j];
	double dy[3600 * j];
	double dx_dx_2[3600 * j];
	double dy_dy_2[3600 * j];
	double dx_dy_2[3600 * j];
	double y_x_2[3600 * j];
	double sgn_x[3600 * j];
	double sgn_y[3600 * j];
	int flag[3600 * j];
	int flag2[3600 * j];
	double sigma = 0.5 * width;
	double conste = -1.0 / (2 * sigma * sigma);
	double PI2 = CV_PI * 2;
	double power = 0.5;
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		sub_hist_width[i] = DESCR_SCALE_ADJUST * features2[i].octave_scale;
		radius[i] = (sub_hist_width[i] * sqrt(2.0) * (width + 1)) / 2.0 + 0.5;
		cos_ori[i] = cos(features2[i].ori);
		sin_ori[i] = sin(features2[i].ori);
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				flag[i * 3600 + (m + 30) * 60 + (n + 30)] = 1;
				flag2[i * 3600 + (m + 30) * 60 + (n + 30)] = 1;
				rot_x[i * 3600 + (m + 30) * 60 + (n + 30)] = (cos_ori[i] * n - sin_ori[i] * m) / sub_hist_width[i];
				rot_y[i * 3600 + (m + 30) * 60 + (n + 30)] = (sin_ori[i] * n + cos_ori[i] * m) / sub_hist_width[i];
				xbin[i * 3600 + (m + 30) * 60 + (n + 30)] = rot_x[i * 3600 + (m + 30) * 60 + (n + 30)] + width / 2 - 0.5;
				ybin[i * 3600 + (m + 30) * 60 + (n + 30)] = rot_y[i * 3600 + (m + 30) * 60 + (n + 30)] + width / 2 - 0.5;
				if (xbin[i * 3600 + (m + 30) * 60 + (n + 30)] > -1.0 && xbin[i * 3600 + (m + 30) * 60 + (n + 30)] < width && ybin[i * 3600 + (m + 30) * 60 + (n + 30)] > -1.0 && ybin[i * 3600 + (m + 30) * 60 + (n + 30)] < width) {
					flag[i * 3600 + (m + 30) * 60 + (n + 30)] = 0;
					if (features2[i].x + n > 0 && features2[i].x + n < gauss_pyr2[features2[i].octave * (INTERVALS + 3) + features2[i].interval].cols - 1 && features2[i].y + m > 0 && features2[i].y + m < gauss_pyr2[features2[i].octave * (INTERVALS + 3) + features2[i].interval].rows - 1) {
						flag2[i * 3600 + (m + 30) * 60 + (n + 30)] = 0;
						pixel_t* data = (pixel_t*)gauss_pyr2[features2[i].octave * (INTERVALS + 3) + features2[i].interval].data;
						int step = gauss_pyr2[features2[i].octave * (INTERVALS + 3) + features2[i].interval].cols;
						dx[i * 3600 + (m + 30) * 60 + (n + 30)] = *(data + step * (features2[i].y + m) + (features2[i].x + n + 1)) - (*(data + step * (features2[i].y + m) + (features2[i].x + n - 1)));
						dy[i * 3600 + (m + 30) * 60 + (n + 30)] = *(data + step * (features2[i].y + m + 1) + features2[i].x + n) - (*(data + step * (features2[i].y + m - 1) + features2[i].x + n));
						secMul2_1(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, dx[i * 3600 + (m + 30) * 60 + (n + 30)], dx[i * 3600 + (m + 30) * 60 + (n + 30)]);
						dx_dx_2[i * 3600 + (m + 30) * 60 + (n + 30)] = b3[i * 3600 + (m + 30) * 60 + (n + 30)];
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++) {
			for (int n = -radius[i]; n <= radius[i]; n++) {
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag2[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						secMul2_1(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, dy[i * 3600 + (m + 30) * 60 + (n + 30)], dy[i * 3600 + (m + 30) * 60 + (n + 30)]);
						dy_dy_2[i * 3600 + (m + 30) * 60 + (n + 30)] = b3[i * 3600 + (m + 30) * 60 + (n + 30)];
						dx_dy_2[i * 3600 + (m + 30) * 60 + (n + 30)] = dx_dx_2[i * 3600 + (m + 30) * 60 + (n + 30)] + dy_dy_2[i * 3600 + (m + 30) * 60 + (n + 30)];
						//cout << "222x:" << "  dx_dy_2:" << dx_dy_2[i * 3600 + (m + 30) * 60 + (n + 30)] << endl;
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++) {
			for (int n = -radius[i]; n <= radius[i]; n++) {
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag2[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						secPRE2_1(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, dx_dy_2[i * 3600 + (m + 30) * 60 + (n + 30)], cc_2);
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++) {
			for (int n = -radius[i]; n <= radius[i]; n++) {
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag2[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						secPRE2_2(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2,r_2, power);
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++) {
			for (int n = -radius[i]; n <= radius[i]; n++) {
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag2[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						secPRE2_3(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, b1, b2, b3,cc_2);
						grad_mag[i * 3600 + (m + 30) * 60 + (n + 30)] = b1[i * 3600 + (m + 30) * 60 + (n + 30)];
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	double thresh = 0;
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++) {
			for (int n = -radius[i]; n <= radius[i]; n++) {
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag2[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						secCmp2_1(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, dx[i * 3600 + (m + 30) * 60 + (n + 30)], thresh, t_2);
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++) {
			for (int n = -radius[i]; n <= radius[i]; n++) {
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag2[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						secCmp2_2(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, b1, b2, b3);
						sgn_x[i * 3600 + (m + 30) * 60 + (n + 30)] = b1[i * 3600 + (m + 30) * 60 + (n + 30)];
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++) {
			for (int n = -radius[i]; n <= radius[i]; n++) {
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag2[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						secCmp2_1(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, dy[i * 3600 + (m + 30) * 60 + (n + 30)], thresh, t_2);
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++) {
			for (int n = -radius[i]; n <= radius[i]; n++) {
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag2[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						secCmp2_2(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, b1, b2, b3);
						sgn_y[i * 3600 + (m + 30) * 60 + (n + 30)] = b1[i * 3600 + (m + 30) * 60 + (n + 30)];
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	double tx_2[3600 * j];
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++) {
			for (int n = -radius[i]; n <= radius[i]; n++) {
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag2[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						secDiv2_1(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, tx_2, dy[i * 3600 + (m + 30) * 60 + (n + 30)], t_2);
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++) {
			for (int n = -radius[i]; n <= radius[i]; n++) {
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag2[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						secDiv2_2(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, dx[i * 3600 + (m + 30) * 60 + (n + 30)], t_2);
					}
				}
			}
		}
	}
	
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int m = -radius[i]; m <= radius[i]; m++) {
			for (int n = -radius[i]; n <= radius[i]; n++) {
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag2[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						secDiv2_3(i * 3600 + (m + 30) * 60 + (n + 30), ptext1, ptext2, b1, b2, b3, tx_2);
						y_x_2[i * 3600 + (m + 30) * 60 + (n + 30)] = b1[i * 3600 + (m + 30) * 60 + (n + 30)];
					}
				}
			}
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	DescriptorOri_2(j, radius, flag, flag2, sem, ptext1, ptext2, grad_ori, a_2, b_2, c_2, sgn_x, sgn_y, y_x_2, t_2);
	for (int i = 0; i < j; i++) {
		for (int l = 0; l < width * width * bins; l++) hist[i * width * width * bins + l] = 0.0;
		for (int m = -radius[i]; m <= radius[i]; m++)
		{
			for (int n = -radius[i]; n <= radius[i]; n++)
			{
				
				if (flag[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
					if (flag2[i * 3600 + (m + 30) * 60 + (n + 30)] == 0) {
						grad_ori[i * 3600 + (m + 30) * 60 + (n + 30)] = grad_ori[i * 3600 + (m + 30) * 60 + (n + 30)] * CV_PI / 180;
						grad_ori[i * 3600 + (m + 30) * 60 + (n + 30)] = (CV_PI - grad_ori[i * 3600 + (m + 30) * 60 + (n + 30)]) - features2[i].ori;
						while (grad_ori[i * 3600 + (m + 30) * 60 + (n + 30)] < 0.0)
							grad_ori[i * 3600 + (m + 30) * 60 + (n + 30)] += PI2;
						while (grad_ori[i * 3600 + (m + 30) * 60 + (n + 30)] >= PI2)
							grad_ori[i * 3600 + (m + 30) * 60 + (n + 30)] -= PI2;
						obin[i * 3600 + (m + 30) * 60 + (n + 30)] = grad_ori[i * 3600 + (m + 30) * 60 + (n + 30)] * (bins / PI2);
						weight[i * 3600 + (m + 30) * 60 + (n + 30)] = exp(conste * (rot_x[i * 3600 + (m + 30) * 60 + (n + 30)] * rot_x[i * 3600 + (m + 30) * 60 + (n + 30)] + rot_y[i * 3600 + (m + 30) * 60 + (n + 30)] * rot_y[i * 3600 + (m + 30) * 60 + (n + 30)]));
						InterpHistEntry(i, hist, xbin[i * 3600 + (m + 30) * 60 + (n + 30)], ybin[i * 3600 + (m + 30) * 60 + (n + 30)], obin[i * 3600 + (m + 30) * 60 + (n + 30)], grad_mag[i * 3600 + (m + 30) * 60 + (n + 30)] * weight[i * 3600 + (m + 30) * 60 + (n + 30)], bins, width);
					}
				}
			}
		}
		for (int l = 0; l < width * width * bins; l++) 
			features2[i].descriptor[l] = hist[i * width * width * bins + l];
	}
}
void NormalizeDescr1(CSEM sem, double* ptext1, double* ptext2, vector<Keypoint>& features1, double a_1, double b_1, double c_1, double t_1, double cc_1, double r_1) {
	int j = features1.size();
	int d = 128;
	double len_sq[j];
	double a1[j * 128];
	double a2[j * 128];
	double a3[j * 128];
	double power = 0.5;
	for (int i = 0; i < j; i++) {
		len_sq[i] = 0.0;
		for (int l = 0; l < d; l++) {
			//cout << i <<  "   111  descriptor:" << l << "  " << features1[i].descriptor[l] << endl;
			secMul1_1(i * 128 + l, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, features1[i].descriptor[l], features1[i].descriptor[l]);
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (int l = 0; l < d; l++) {
			secMul1_2(i * 128 + l, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
			len_sq[i] += a3[i * 128 + l];
		}
	}
	for (int i = 0; i < j; i++) {
		secPRE1_1(i, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, len_sq[i], cc_1);
		

	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		secPRE1_2(i, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, r_1, power);
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		secPRE1_3(i, ptext1, ptext2, a1, a2, a3, cc_1);
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		a2[i] = ptext1[i] + a1[i];
		ptext1[i] = a1[i];
		for (int l = 0; l < d; l++) {
			features1[i].descriptor[l] /= a2[i];
			//cout << i << "   111  descriptor:" << l << "  " << features1[i].descriptor[l] << endl;
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	
}
void NormalizeDescr2(CSEM sem, double* ptext1, double* ptext2, vector<Keypoint>& features2, double a_2, double b_2, double c_2, double t_2, double cc_2, double r_2) {
	int j = features2.size();
	int d = 128;
	double cur[j];
	double len_sq[j];
	double b1[j * 128];
	double b2[j * 128];
	double b3[j * 128];
	double power = 0.5;
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		len_sq[i] = 0.0;
		for (int l = 0; l < d; l++) {
			//cout << i << "   222  descriptor:  " <<l<<"  "<< features2[i].descriptor[l] << endl;
			secMul2_1(i * 128 + l, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, features2[i].descriptor[l], features2[i].descriptor[l]);
			len_sq[i] += b3[i * 128 + l];
		}
		//cout << "222  len_sq:  " << len_sq[i] << endl;
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");	
	for (int i = 0; i < j; i++) {
		secPRE2_1(i, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, len_sq[i], cc_2);
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		secPRE2_2(i, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, r_2, power);
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		secPRE2_3(i, ptext1, ptext2, b1, b2, b3, cc_2);
		ptext1[i] = b1[i];
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		b2[i] = ptext1[i] + b1[i];
		//cout << i << "   222  descriptor:  " << "  " << features2[i].descriptor[0] << endl;
		for (int l = 0; l < d; l++) {
			features2[i].descriptor[l] /= b2[i];
			//cout << i << "   222  descriptor:  " << l << "  " << features2[i].descriptor[l] << endl;
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");

}
void HistToDescriptor1(CSEM sem, double* ptext1, double* ptext2, double* hist, vector<Keypoint>& features1, double a_1, double b_1, double c_1, double t_1, double cc_1, double r_1,int bins, int width) {
	int j = features1.size();
	int int_val, r, c, o,k,l;
	double val, thr_255;
	thr_255 = 255;
	double a1[j * 128];
	double a2[j * 128];
	double a3[j * 128];
	double thr = DESCR_MAG_THR;
	//for (int i = 0; i < j; i++) 
	//	for (int l = 0; l < width * width * bins; l++)
	//		//cout << "111  descriptor:" << features1[i].descriptor[l] << endl;
	
	NormalizeDescr1(sem, ptext1, ptext2, features1, a_1, b_1, c_1, t_1, cc_1, r_1);
	for (int i = 0; i < j; i++) {
		for (l = 0; l < 128; l++) {
			secCmp1_1(i * 128 + l, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, features1[i].descriptor[l], thr, t_1);
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (l = 0; l < 128; l++) {
			secCmp1_2(i * 128 + l, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (l = 0; l < 128; l++) {
			secCmp1_3(i * 128 + l, ptext1, ptext2, a1, a2, a3);
			
			if (a1[i * 128 + l] == 1) features1[i].descriptor[l] = 0.5 * DESCR_MAG_THR;
		}
	}
	
	NormalizeDescr1(sem, ptext1, ptext2, features1, a_1, b_1, c_1, t_1, cc_1, r_1);
	for (int i = 0; i < j; i++) {
		for (l = 0; l < 128; l++) {
			val = INT_DESCR_FCTR * features1[i].descriptor[l];
			secCmp1_1(i * 128 + l, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1, val, thr_255, t_1);
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (l = 0; l < 128; l++) {
			secCmp1_2(i * 128 + l, ptext1, ptext2, a1, a2, a3, a_1, b_1, c_1);
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("A  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (l = 0; l < 128; l++) {
			secCmp1_3(i * 128 + l, ptext1, ptext2, a1, a2, a3);
			if (a1[i * 128 + l] != -1) features1[i].descriptor[l] = 255;
			else features1[i].descriptor[l] = (int)(INT_DESCR_FCTR * features1[i].descriptor[l]);
			//cout << i << "   111  descriptor:" << l << "  " << features1[i].descriptor[l] << endl;
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("A  sem.post ok\n");


}
void HistToDescriptor2(CSEM sem, double* ptext1, double* ptext2, double* hist, vector<Keypoint>& features2, double a_2, double b_2, double c_2, double t_2, double cc_2, double r_2, int bins, int width) {
	int j = features2.size();
	int int_val, r, c, o, k, l;
	double val, thr_255;
	thr_255 = 0;
	double b1[j * 128];
	double b2[j * 128];
	double b3[j * 128];
	double thr = DESCR_MAG_THR;
	//for (int i = 0; i < j; i++) 
	//	for (int l = 0; l < width * width * bins; l++)
	//		//cout << "222  descriptor:" << features2[i].descriptor[l] << endl;

	
	NormalizeDescr2(sem, ptext1, ptext2, features2, a_2, b_2, c_2, t_2, cc_2, r_2);
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (l = 0; l < 128; l++) {
			
			secCmp2_1(i * 128 + l, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, features2[i].descriptor[l], thr, t_2);
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (l = 0; l < 128; l++) {
			secCmp2_2(i * 128 + l, ptext1, ptext2, b1, b2, b3);
			if (b1[i * 128 + l] == 1) features2[i].descriptor[l] = 0.5 * DESCR_MAG_THR;
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	NormalizeDescr2(sem, ptext1, ptext2, features2, a_2, b_2, c_2, t_2, cc_2, r_2);
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (l = 0; l < 128; l++) {
			val = INT_DESCR_FCTR * features2[i].descriptor[l];
			//cout << i << "   222  descriptor:  " << l << "  " <<"features2[i].descriptor[l]-----" << features2[i].descriptor[l]   <<"val-----" << val << endl;
			secCmp2_1(i * 128 + l, ptext1, ptext2, b1, b2, b3, a_2, b_2, c_2, val, thr_255, t_2);
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");
	if (sem.wait() == false)  ;//printf("sem.wait failed.\n");
	//else printf("B  sem.wait ok\n");
	for (int i = 0; i < j; i++) {
		for (l = 0; l < 128; l++) {
			secCmp2_2(i * 128 + l, ptext1, ptext2, b1, b2, b3);
			if (b1[i * 128 + l] != -1) features2[i].descriptor[l] = 255;
			else features2[i].descriptor[l] = (int)(INT_DESCR_FCTR * features2[i].descriptor[l]);
			//cout << i << "   222  descriptor:  " << l << "  " << features2[i].descriptor[l] << endl;
		}
	}
	if (sem.post() == false)  ;//printf("sem.post failed.\n");
	//else printf("B  sem.post ok\n");

}
void DescriptorRepresentation1(CSEM sem, double* ptext1, double* ptext2, vector<Keypoint>& features1, const vector<Mat>& gauss_pyr1, double a_1, double b_1, double c_1, double t_1, double cc_1, double r_1, int bins, int width){
	double hist[features1.size() * width * width * bins];
	CalculateDescrHist1(sem, ptext1, ptext2, hist, features1, gauss_pyr1, a_1, b_1, c_1, t_1, cc_1, r_1, bins, width);
	HistToDescriptor1(sem, ptext1, ptext2, hist, features1, a_1, b_1, c_1, t_1, cc_1, r_1, bins, width);
}
void DescriptorRepresentation2(CSEM sem, double* ptext1, double* ptext2, vector<Keypoint>& features2, const vector<Mat>& gauss_pyr2, double a_2, double b_2, double c_2, double t_2, double cc_2, double r_2, int bins, int width) {
	double hist[features2.size() * width * width * bins];
	CalculateDescrHist2(sem, ptext1, ptext2, hist, features2, gauss_pyr2, a_2, b_2, c_2, t_2, cc_2, r_2, bins, width);
	HistToDescriptor2(sem, ptext1, ptext2, hist, features2, a_2, b_2, c_2, t_2, cc_2, r_2, bins, width);
}

void InterpHistEntry(int ii,double* hist, double xbin, double ybin, double obin, double mag, int bins, int d) {
	double d_r, d_c, d_o, v_r, v_c, v_o;
	int r0, c0, o0, rb, cb, ob, r, c, o;
	r0 = cvFloor(ybin);  
	c0 = cvFloor(xbin);
	o0 = cvFloor(obin);
	d_r = ybin - r0;   
	d_c = xbin - c0;
	d_o = obin - o0;

	for (r = 0; r <= 1; r++) {
		rb = r0 + r;
		if (rb >= 0 && rb < d) {
			v_r = mag * ((r == 0) ? 1.0 - d_r : d_r);
			for (c = 0; c <= 1; c++) {
				cb = c0 + c;
				if (cb >= 0 && cb < d) {
					v_c = v_r * ((c == 0) ? 1.0 - d_c : d_c);
					for (o = 0; o <= 1; o++) {
						ob = (o0 + o) % bins;        
						v_o = v_c * ((o == 0) ? 1.0 - d_o : d_o);
						hist[ii * 128 + rb * 32 + cb * 8 + ob] += v_o;
					}
				}
			}
		}
	}
	
}
void CreateInitSmoothGray(const Mat& src, Mat& dst, double sigma) {

	//dst.convertTo(dst, DataType<double>::type, 1, 0);

	double sig_diff;
	sig_diff = sqrt(max(sigma * sigma - INIT_SIGMA * INIT_SIGMA * 4, 0.01));
	resize(src, dst, Size(src.cols * 2, src.rows * 2), 0, 0, INTER_LINEAR);

	GaussianBlur(dst, dst, Size(), sig_diff, sig_diff);

}

void GaussianPyramid(const Mat& src, vector<Mat>& gauss_pyr, int octaves, int intervals, double sigma)
{
	//std::cout << "sigma is     = " << sigma << std::endl;
	double* sigmas = new double[intervals + 3];
	double k = pow(2.0, 1.0 / intervals);      //pow（x,y）,x的y次方；k=2的（1/s)次方
											   //  std::cout << "k is     = " << k << std::endl;
	sigmas[0] = sigma;
	//  std::cout << "sigma0 is     = " << sigmas[0] << std::endl;
	double sig_prev;
	double sig_total;
	//用循环来表示第0组内各层图片的尺度，这个可以表示为0组的各层图像尺度；
	for (int i = 1; i < intervals + 3; i++)
	{
		sig_prev = pow(k, i - 1) * sigma;     //第i-1层尺度
		sig_total = sig_prev * k;             //第i层尺度
		sigmas[i] = sqrt(sig_total * sig_total - sig_prev * sig_prev);  //组内每层尺度坐标
		
	}

	for (int o = 0; o < octaves; o++)
	{
		//每组多三层
		for (int i = 0; i < intervals + 3; i++)
		{
			Mat mat;
			if (o == 0 && i == 0)    //第0组，第0层，就是原图像
			{
				src.copyTo(mat);
			}
			else if (i == 0)           //新的一组的首层图像是由上一组最后一层图像下采样得到
			{
				const Mat& src = gauss_pyr[(o - 1) * (intervals + 3) + intervals];
				resize(src, mat, Size(src.cols / 2, src.rows / 2), 0, 0, INTER_NEAREST);
			}
			else        //对上一层图像gauss_pyr[i-1]进行参数为sig[i]的高斯平滑，得到当前层图像
			{
				GaussianBlur(gauss_pyr[o * (intervals + 3) + i - 1], mat, Size(), sigmas[i], sigmas[i]);
				//  std::cout << "sigmas[i] is     = " << sigmas[i] << std::endl;
			}
			gauss_pyr.push_back(mat);  //将mat中的像素添加到gauss_pyr数组里面；也就是说每组从第0层到第5层的图片像素装进vector数组中；

		}

	}

	delete[] sigmas;
}

void DogPyramid(const vector<Mat>& gauss_pyr, vector<Mat>& dog_pyr, int octaves, int intervals)
{

	for (int o = 0; o < octaves; o++)
	{
		for (int i = 1; i < intervals + 3; i++)
		{
			Mat mat;
			subtract(gauss_pyr[o * (intervals + 3) + i], gauss_pyr[o * (intervals + 3) + i - 1], mat, noArray(), DataType<double>::type);
			dog_pyr.push_back(mat);
		}
	}

}

void CalculateScale(vector<Keypoint>& features, double sigma, int intervals)
{
	double intvl = 0;
	for (int i = 0; i < features.size(); i++)
	{
		intvl = features[i].interval + features[i].offset_interval;
		features[i].scale = sigma * pow(2.0, features[i].octave + intvl / intervals);
		features[i].octave_scale = sigma * pow(2.0, intvl / intervals);
		//cout << "features[i].interval:" << features[i].interval << " features[i].offset_interval:" << features[i].offset_interval << "  octave_scale:" << features[i].octave_scale << endl;
	}

}

void HalfFeatures(vector<Keypoint>& features)
{
	for (int i = 0; i < features.size(); i++)
	{
		features[i].dx /= 2;
		features[i].dy /= 2;
		features[i].scale /= 2;
	}
}
void write_features(const vector<Keypoint>& features, const char* file)
{
	ofstream dout(file);
	dout << features.size() << " " << FEATURE_ELEMENT_LENGTH << endl;
	for (int i = 0; i < features.size(); i++)
	{
		dout << features[i].dy << " " << features[i].dx << " " << features[i].scale << " " << features[i].ori << endl;
		for (int j = 0; j < FEATURE_ELEMENT_LENGTH; j++)
		{
			if (j % 20 == 0)
				dout << endl;
			dout << features[i].descriptor[j] << " ";
		}
		dout << endl;
	}
	dout.close();
}