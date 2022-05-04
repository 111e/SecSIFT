#include <cstdio>
#include <unistd.h>
#include "tool.h"
#include"SIFT.h"
#include <time.h>
#include <sys/time.h>
using namespace cv;
using namespace std;


void zhiluan(Mat& src1, Mat& src2) {
	src1.copyTo(src2);
	int d = 50;
	/*Rect rect1(0, 0, 200, 300);
	Rect rect2(200, 0, 200, 300);
	src1(rect1).copyTo(src2(rect2));
	src1(rect2).copyTo(src2(rect1));*/

	/*Rect rect1(0, 0, d, d);
	Rect rect2(0, d, d, d);
	Rect rect3(d, 0, d, d);
	Rect rect4(d, d, d, d);
	src1(rect1).copyTo(src2(rect4));
	src1(rect2).copyTo(src2(rect3));
	src1(rect3).copyTo(src2(rect2));
	src1(rect4).copyTo(src2(rect1));*/

	/*Rect rect1(0, 0, d, d);
	Rect rect2(0, d, d, d);
	Rect rect3(0, 2 * d, d, d);
	Rect rect4(d, 0, d, d);
	Rect rect5(d, d, d, d);
	Rect rect6(d, 2 * d, d, d);
	Rect rect7(2 * d, 0, d, d);
	Rect rect8(2 * d, d, d, d);
	Rect rect9(2 * d, 2 * d, d, d);
	Rect rect10(3 * d, 0, d, d);
	Rect rect11(3 * d, d, d, d);
	Rect rect12(3 * d, 2 * d, d, d);

	src1(rect1).copyTo(src2(rect12));
	src1(rect2).copyTo(src2(rect11));
	src1(rect3).copyTo(src2(rect10));
	src1(rect4).copyTo(src2(rect9));
	src1(rect5).copyTo(src2(rect8));
	src1(rect6).copyTo(src2(rect7));
	src1(rect7).copyTo(src2(rect6));
	src1(rect8).copyTo(src2(rect5));
	src1(rect9).copyTo(src2(rect4));
	src1(rect10).copyTo(src2(rect3));
	src1(rect11).copyTo(src2(rect2));
	src1(rect12).copyTo(src2(rect1));
*/

	Rect rect1(0, 0, d, d);
	Rect rect2(0, d, d, d);
	Rect rect3(0, 2 * d, d, d);
	Rect rect4(d, 0, d, d);
	Rect rect5(d, d, d, d);
	Rect rect6(d, 2 * d, d, d);
	Rect rect7(2 * d, 0, d, d);
	Rect rect8(2 * d, d, d, d);
	Rect rect9(2 * d, 2 * d, d, d);
	Rect rect10(3 * d, 0, d, d);
	Rect rect11(3 * d, d, d, d);
	Rect rect12(3 * d, 2 * d, d, d);

	Rect rect13(0, 3*d+0, d, d);
	Rect rect14(0, 3 * d + d, d, d);
	Rect rect15(0, 3 * d + 2 * d, d, d);
	Rect rect16(d, 3 * d + 0, d, d);
	Rect rect17(d, 3 * d + d, d, d);
	Rect rect18(d, 3 * d + 2 * d, d, d);
	Rect rect19(2 * d, 3 * d + 0, d, d);
	Rect rect20(2 * d, 3 * d + d, d, d);
	Rect rect21(2 * d, 3 * d + 2 * d, d, d);
	Rect rect22(3 * d, 3 * d + 0, d, d);
	Rect rect23(3 * d, 3 * d + d, d, d);
	Rect rect24(3 * d, 3 * d + 2 * d, d, d);


	Rect rect25(4*d+0, 0, d, d);
	Rect rect26(4 * d + 0, d, d, d);
	Rect rect27(4 * d + 0, 2 * d, d, d);
	Rect rect28(4 * d + d, 0, d, d);
	Rect rect29(4 * d + d, d, d, d);
	Rect rect30(4 * d + d, 2 * d, d, d);
	Rect rect31(4 * d + 2 * d, 0, d, d);
	Rect rect32(4 * d + 2 * d, d, d, d);
	Rect rect33(4 * d + 2 * d, 2 * d, d, d);
	Rect rect34(4 * d + 3 * d, 0, d, d);
	Rect rect35(4 * d + 3 * d, d, d, d);
	Rect rect36(4 * d + 3 * d, 2 * d, d, d);

	Rect rect37(4 * d + 0, 3 * d + 0, d, d);
	Rect rect38(4 * d + 0, 3 * d + d, d, d);
	Rect rect39(4 * d + 0, 3 * d + 2 * d, d, d);
	Rect rect40(4 * d + d, 3 * d + 0, d, d);
	Rect rect41(4 * d + d, 3 * d + d, d, d);
	Rect rect42(4 * d + d, 3 * d + 2 * d, d, d);
	Rect rect43(4 * d + 2 * d, 3 * d + 0, d, d);
	Rect rect44(4 * d + 2 * d, 3 * d + d, d, d);
	Rect rect45(4 * d + 2 * d, 3 * d + 2 * d, d, d);
	Rect rect46(4 * d + 3 * d, 3 * d + 0, d, d);
	Rect rect47(4 * d + 3 * d, 3 * d + d, d, d);
	Rect rect48(4 * d + 3 * d, 3 * d + 2 * d, d, d);

	src1(rect1).copyTo(src2(rect12));
	src1(rect2).copyTo(src2(rect11));
	src1(rect3).copyTo(src2(rect10));
	src1(rect4).copyTo(src2(rect9));
	src1(rect5).copyTo(src2(rect8));
	src1(rect6).copyTo(src2(rect7));
	src1(rect7).copyTo(src2(rect6));
	src1(rect8).copyTo(src2(rect5));
	src1(rect9).copyTo(src2(rect4));
	src1(rect10).copyTo(src2(rect3));
	src1(rect11).copyTo(src2(rect2));
	src1(rect12).copyTo(src2(rect1));

	src1(rect13).copyTo(src2(rect24));
	src1(rect14).copyTo(src2(rect23));
	src1(rect15).copyTo(src2(rect22));
	src1(rect16).copyTo(src2(rect21));
	src1(rect17).copyTo(src2(rect20));
	src1(rect18).copyTo(src2(rect19));
	src1(rect19).copyTo(src2(rect18));
	src1(rect20).copyTo(src2(rect17));
	src1(rect21).copyTo(src2(rect16));
	src1(rect22).copyTo(src2(rect15));
	src1(rect23).copyTo(src2(rect14));
	src1(rect24).copyTo(src2(rect13));

	src1(rect25).copyTo(src2(rect36));
	src1(rect26).copyTo(src2(rect35));
	src1(rect27).copyTo(src2(rect34));
	src1(rect28).copyTo(src2(rect33));
	src1(rect29).copyTo(src2(rect32));
	src1(rect30).copyTo(src2(rect31));
	src1(rect31).copyTo(src2(rect30));
	src1(rect32).copyTo(src2(rect29));
	src1(rect33).copyTo(src2(rect28));
	src1(rect34).copyTo(src2(rect27));
	src1(rect35).copyTo(src2(rect26));
	src1(rect36).copyTo(src2(rect25));

	src1(rect37).copyTo(src2(rect48));
	src1(rect38).copyTo(src2(rect47));
	src1(rect39).copyTo(src2(rect46));
	src1(rect40).copyTo(src2(rect45));
	src1(rect41).copyTo(src2(rect44));
	src1(rect42).copyTo(src2(rect43));
	src1(rect43).copyTo(src2(rect42));
	src1(rect44).copyTo(src2(rect41));
	src1(rect45).copyTo(src2(rect40));
	src1(rect46).copyTo(src2(rect39));
	src1(rect47).copyTo(src2(rect38));
	src1(rect48).copyTo(src2(rect37));


	//Rect rect1(0, 0, d,  d);
	//Rect rect2(d, 0, d, d);
	//Rect rect3(2*d,0, d,d);
	//Rect rect4(3*d, 0, d, d);
	//Rect rect5(4 * d, 0, d, d);
	//Rect rect6(5 * d, 0, d, d);
	//Rect rect7(6 * d, 0, d, d);
	//Rect rect8(7 * d, 0, d, d);


	//Rect rect9(0, d, d, d);
	//Rect rect10(d,d, d, d);
	//Rect rect11(2 * d, d, d, d);
	//Rect rect12(3 * d, d, d, d);
	//Rect rect13(4 * d, d, d, d);
	//Rect rect14(5 * d, d, d, d);
	//Rect rect15(6 * d, d, d, d);
	//Rect rect16(7 * d, d, d, d);

	//Rect rect17(0, 2 * d, d, d);
	//Rect rect18(d, 2 * d, d, d);
	//Rect rect19(2 * d, 2 * d, d, d);
	//Rect rect20(3 * d, 2 * d, d, d);
	//Rect rect21(4 * d, 2 * d, d, d);
	//Rect rect22(5 * d, 2 * d, d, d);
	//Rect rect23(6 * d, 2 * d, d, d);
	//Rect rect24(7 * d, 2 * d, d, d);

	//Rect rect25(0, 3 * d, d, d);
	//Rect rect26(d, 3 * d, d, d);
	//Rect rect27(2 * d, 3 * d, d, d);
	//Rect rect28(3 * d, 3 * d, d, d);
	//Rect rect29(4 * d, 3 * d, d, d);
	//Rect rect30(5 * d, 3 * d, d, d);
	//Rect rect31(6 * d, 3 * d, d, d);
	//Rect rect32(7 * d, 3 * d, d, d);

	//Rect rect33(0, 4 * d, d, d);
	//Rect rect34(d, 4 * d, d, d);
	//Rect rect35(2 * d, 4 * d, d, d);
	//Rect rect36(3 * d, 4 * d, d, d);
	//Rect rect37(4 * d, 4 * d, d, d);
	//Rect rect38(5 * d, 4 * d, d, d);
	//Rect rect39(6 * d, 4 * d, d, d);
	//Rect rect40(7 * d, 4 * d, d, d);

	//Rect rect41(0, 5 * d, d, d);
	//Rect rect42(d, 5 * d, d, d);
	//Rect rect43(2 * d, 5 * d, d, d);
	//Rect rect44(3 * d, 5 * d, d, d);
	//Rect rect45(4 * d, 5 * d, d, d);
	//Rect rect46(5 * d, 5 * d, d, d);
	//Rect rect47(6 * d, 5 * d, d, d);
	//Rect rect48(7 * d, 5* d, d, d);

	//



	//src1(rect1).copyTo(src2(rect20));
	//src1(rect2).copyTo(src2(rect19));
	//src1(rect3).copyTo(src2(rect18));
	//src1(rect4).copyTo(src2(rect17));
	//src1(rect5).copyTo(src2(rect16));
	//src1(rect6).copyTo(src2(rect15));
	//src1(rect7).copyTo(src2(rect14));
	//src1(rect8).copyTo(src2(rect13));
	//src1(rect9).copyTo(src2(rect12));
	//src1(rect10).copyTo(src2(rect11));
	//src1(rect11).copyTo(src2(rect10));
	//src1(rect12).copyTo(src2(rect9));
	//src1(rect13).copyTo(src2(rect8));
	//src1(rect14).copyTo(src2(rect7));
	//src1(rect15).copyTo(src2(rect6));
	//src1(rect16).copyTo(src2(rect5));
	//src1(rect17).copyTo(src2(rect4));
	//src1(rect18).copyTo(src2(rect3));
	//src1(rect19).copyTo(src2(rect2));
	//src1(rect20).copyTo(src2(rect1));

	//src1(rect21).copyTo(src2(rect40));
	//src1(rect22).copyTo(src2(rect39));
	//src1(rect23).copyTo(src2(rect38));
	//src1(rect24).copyTo(src2(rect37));
	//src1(rect25).copyTo(src2(rect36));
	//src1(rect26).copyTo(src2(rect35));
	//src1(rect27).copyTo(src2(rect34));
	//src1(rect28).copyTo(src2(rect33));
	//src1(rect29).copyTo(src2(rect32));
	//src1(rect30).copyTo(src2(rect31));
	//src1(rect31).copyTo(src2(rect30));
	//src1(rect32).copyTo(src2(rect29));
	//src1(rect33).copyTo(src2(rect28));
	//src1(rect34).copyTo(src2(rect27));
	//src1(rect35).copyTo(src2(rect26));
	//src1(rect36).copyTo(src2(rect25));
	//src1(rect37).copyTo(src2(rect24));
	//src1(rect38).copyTo(src2(rect23));
	//src1(rect39).copyTo(src2(rect22));
	//src1(rect40).copyTo(src2(rect21));

	//src1(rect41).copyTo(src2(rect48));
	//src1(rect42).copyTo(src2(rect47));
	//src1(rect43).copyTo(src2(rect46));
	//src1(rect44).copyTo(src2(rect45));
	//src1(rect45).copyTo(src2(rect44));
	//src1(rect46).copyTo(src2(rect43));
	//src1(rect47).copyTo(src2(rect42));
	//src1(rect48).copyTo(src2(rect41));

}
int main()
{
	int shmid1, shmid2, i;
    pid_t pid;
	CSEM sem;
	struct timeval start1, end1;
	gettimeofday(&start1, NULL);
	Mat src1 = imread("/home/amax/zxl/projects/shm_secsift/img/rotate/0.jpg");
	cv::resize(src1, src1, cv::Size(round(src1.cols * 0.5), round(src1.rows * 0.5)));


	

	if (src1.channels() == 3 || src1.channels() == 4)
		cvtColor(src1, src1, COLOR_BGR2GRAY);
	else
		src1.copyTo(src1);
	Mat dst1;
	src1.convertTo(dst1, DataType<double>::type, 1, 0);



	//旋转
	Mat src2 = imread("/home/amax/zxl/projects/shm_secsift/img/rotate/30.jpg");
	cv::resize(src2, src2, cv::Size(round(src2.cols * 0.5 * 0.7), round(src2.rows * 0.5 * 0.7)));
	//模糊
	
	/*Mat src2 = imread("/home/amax/zxl/projects/shm_secsift/img/blur/img2.jpg");
	cv::resize(src2, src2, cv::Size(400, 300));*/

	//3维
	/*Mat src2 = imread("/home/amax/zxl/projects/shm_secsift/img/img3.jpg");
	cv::resize(src2, src2, cv::Size(round(src2.cols * 0.05), round(src2.rows * 0.05)));*/

	//Mat src = imread("/home/amax/zxl/projects/shm_secsift/img/img1.jpg");
	//cv::resize(src, src, cv::Size(400, 300));
	//Mat src2;
	////src.copyTo(src2);
	//zhiluan(src, src2);

	if (src2.channels() == 3 || src2.channels() == 4)
		cvtColor(src2, src2, COLOR_BGR2GRAY);
	else
		src2.copyTo(src2);
	Mat dst2;
	src2.convertTo(dst2, DataType<double>::type, 1, 0);



	double  a[3], a_1[3], a_2[3], b[3], b_1[3], b_2[3], c[3], c_1[3], c_2[3], t[3], t_1[3], t_2[3] = { 0 };
	generate_random(a, a_1, a_2, b, b_1, b_2, c, c_1, c_2, t, t_1, t_2);
	
	int Xr = 3;
	int Xc = 3;
	Mat A(Xr, Xc, CV_64FC1);
	Mat A_1(Xr, Xc, CV_64FC1);
	Mat A_2(Xr, Xc, CV_64FC1);
	randu(A, Scalar::all(-10), Scalar::all(10));
	randu(A_1, Scalar::all(-10), Scalar::all(10));
	A_2 = A - A_1;

	Mat B(Xr, Xc, CV_64FC1);
	Mat B_1(Xr, Xc, CV_64FC1);
	Mat B_2(Xr, Xc, CV_64FC1);
	randu(B, Scalar::all(-10), Scalar::all(10));
	randu(B_1, Scalar::all(-10), Scalar::all(10));
	B_2 = B - B_1;

	Mat C(Xr, Xc, CV_64FC1);
	Mat C_1(Xr, Xc, CV_64FC1);
	Mat C_2(Xr, Xc, CV_64FC1);
	C = A * B;
	randu(C_1, Scalar::all(-10), Scalar::all(10));
	C_2 = C - C_1;

	Mat Z(Xr, Xc, CV_64FC1);
	Mat Z_1(Xr, Xc, CV_64FC1);
	Mat Z_2(Xr, Xc, CV_64FC1);
	randu(Z_1, Scalar::all(-10), Scalar::all(10));
	randu(Z_2, Scalar::all(-10), Scalar::all(10));
	Z = Z_1 + Z_2;

	double r_1 = 0 + rand() % 18;
	double r_2 = 0 + rand() % 20;
	double cc = r_1 * r_2;
	double cc_1 = 0 + rand() % 15;
	double cc_2 = cc - cc_1;


	
	Mat src3(dst1.rows, dst1.cols, DataType<double>::type);
	randu(src3, Scalar::all(0), Scalar::all(255));

	gettimeofday(&end1, NULL);
	printf("jiami   us: %ld\n", 2 * ((end1.tv_sec * 1000000 + end1.tv_usec) - (start1.tv_sec * 1000000 + start1.tv_usec)));
	
	if ((shmid1 = shmget((key_t)0x5003, 59550000000, 0666 | IPC_CREAT)) == -1)
	{
		printf("shmat(0x5003) failed\n"); return -1;
	}
	if ((shmid2 = shmget((key_t)0x5002, 59550000000, 0666 | IPC_CREAT)) == -1)
	{
		printf("shmat(0x5002) failed\n"); return -1;
	}
	double* ptext1 = 0;   // 用于指向共享内存的指针
	ptext1 = (double*)shmat(shmid1, 0, 0);
	double* ptext2 = 0;   // 用于指向共享内存的指针
	ptext2 = (double*)shmat(shmid2, 0, 0);

	if (sem.init(0x5000) == false)  printf("sem.init failed.\n");
	else printf("sem.init ok\n");


	for (i = 0; i < 2; i++)
	{
		if ((pid = fork()) == 0)
		{
			break;//子进程出口
		}
	}

	if (i == 0)   // 兄进程读数据
	{

		vector<Keypoint> features11, features33;
		
		Mat src11 = src3;// 0.8 * dst1; //src3;//
		Mat src33 = 0.8 * dst2;
		


		Sift1(sem, ptext1, ptext2, src11, features11, a_1, b_1, c_1, t_1, cc_1, r_1, A_1, B_1, C_1, Z_1);
		write_features(features11, "descriptor1.txt");
		std::cout << "Sift1 finish-------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
		
		/*Sift1(sem, ptext1, ptext2, src33, features33, a_1, b_1, c_1, t_1, cc_1, r_1, A_1, B_1, C_1, Z_1);
		write_features(features33, "descriptor3.txt");
		std::cout << "Sift1 finish-------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
		*/
	}
	else if (i == 1)
	{
		vector<Keypoint> features22, features44;
		Mat src22 = dst1 - src3; //0.2 * dst1; //dst1 - src3;
		Mat src44 = 0.2 * dst2;
		//cout << "src2:" << src2 << endl;

		Sift2(sem, ptext1, ptext2, src22, features22, a_2, b_2, c_2, t_2, cc_2, r_2, A_2, B_2, C_2, Z_2);
		write_features(features22, "descriptor2.txt");
		std::cout << "Sift2 finish--------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
		
		/*Sift2(sem, ptext1, ptext2, src44, features44, a_2, b_2, c_2, t_2, cc_2, r_2, A_2, B_2, C_2, Z_2);
		write_features(features44, "descriptor4.txt");
		std::cout << "Sift2 finish*****----------------------------------------------------------------------------------------------------------------------------" << std::endl;
		*/
	}
	
	
    waitKey(0);
	
	if (shmctl(shmid1, IPC_RMID, 0) == -1)
	{
		printf("shmctl(0x5003) failed\n"); return -1;
	}
	if (shmctl(shmid2, IPC_RMID, 0) == -1)
	{
		printf("shmctl(0x5002) failed\n"); return -1;
	}
    return 0;
}