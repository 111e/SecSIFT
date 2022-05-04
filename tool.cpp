#include <sys/sem.h>
#include "tool.h"
#include<iomanip>
#include <opencv2/opencv.hpp>
using namespace cv;

void generate_random(double* a, double* a_1, double* a_2, double* b, double* b_1, double* b_2, double* c, double* c_1, double* c_2, double* t, double* t_1, double* t_2) {
	/*a = 0 + rand() % 17;
	a_1 = 15;
	a_2 = a - a_1;
	b = 0 + rand() % 17;
	b_1 = 15;
	b_2 = b - b_1;
	c = a * b;
	c_1 = 0 + rand() %33;
	c_2 = c - c_1;
	t = 0 + rand() % 1125;
	t_1 = 1120;
	t_2 = t - t_1;*/

    /*int i = 0;
    a[i] = 0 + rand() % 17;
    a_1[i] = 15;
    a_2[i] = a[i] - a_1[i];
    b[i] = 0 + rand() % 17;
    b_1[i] = 15;
    b_2[i] = b[i] - b_1[i];
    c[i] = a[i] * b[i];
    c_1[i] = 0 + rand() % 33;
    c_2[i] = c[i] - c_1[i];
    t[i] = 0 + rand() % 125;
    t_1[i] = 1120;
    t_2[i] = t[i] - t_1[i];*/



    int i = 0;
    a[i] = 0 + rand() % 17;
    a_1[i] = 15;
    a_2[i] = a[i] - a_1[i];
    b[i] = 0 + rand() % 17;
    b_1[i] = 15;
    b_2[i] = b[i] - b_1[i];
    c[i] = a[i] * b[i];
    c_1[i] = 0 + rand() % 33;
    c_2[i] = c[i] - c_1[i];
    t[i] = 0 + rand() % 125;
    t_1[i] = 1120;
    t_2[i] = t[i] - t_1[i];
   /* std::cout << "*****************************a_1[i]" << std::fixed << a_1[i] << std::endl;
    std::cout << "a_2[i]"<< std::fixed<< a_2[i] << std::endl;
    std::cout << "b_1[i]" << std::fixed << b_1[i] << std::endl;
    std::cout << "b_2[i]" << std::fixed << b_2[i] << std::endl;
    std::cout << "c_1[i]" << std::fixed << c_1[i] << std::endl;
    std::cout << "c_2[i]" << std::fixed << c_2[i] << std::endl;
    std::cout << "t_1[i]" << std::fixed << t_1[i] << std::endl;
    std::cout << "t_2[i]" << std::fixed << t_2[i] << std::endl;
    std::cout << "a[i]" << std::fixed << a[i] << std::endl;
    std::cout << "b[i]" << std::fixed << b[i] << std::endl;
    std::cout << "c[i]" << std::fixed << c[i] << std::endl;*/


    for (int i = 1; i < 3; i++) {
        a[i] = 0 + rand() % 17;
        a_1[i] = 15;
        a_2[i] = a[i] - a_1[i];
        b[i] = 0 + rand() % 17;
        b_1[i] = 15;
        b_2[i] = b[i] - b_1[i];
        c[i] = a[i] * b[i];
        c_1[i] = 0 + rand() % 33;
        c_2[i] = c[i] - c_1[i];
        t[i] = 0 + rand() % 25;
        t_1[i] = 20;
        t_2[i] = t[i] - t_1[i];
    }


}
int sgn(double xy) {
    if (xy > 0) return 1;
    else if (xy < 0) return -1;
    else return 0;
}

void MatToArray(Mat& img, double* array)
{
    for (int i = 0; i < img.rows; ++i)
    {
        for (int j = 0; j < img.cols; ++j)
        {
            array[i * img.cols + j] = img.at<double>(i, j);
        }
    }
}

Mat ArrayToMat(double* array, int row, int col) {
    Mat img(row, col, CV_64FC1);
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < col; ++j)
        {
            img.at<double>(i, j) = array[i * col + j];
        }
    }
    return img;
}
bool CSEM::init(key_t key)
{
    // 获取信号灯。
    if ((sem_id = semget(key, 1, 0640)) == -1)
    {
        // 如果信号灯不存在，创建它。
        if (errno == 2)
        {
            if ((sem_id = semget(key, 1, 0640 | IPC_CREAT)) == -1) { perror("init 1 semget()"); return false; }

            // 信号灯创建成功后，还需要把它初始化成可用的状态。
            union semun sem_union;
            sem_union.val = 1;
            if (semctl(sem_id, 0, SETVAL, sem_union) < 0) { perror("init semctl()"); return false; }
        }
        else
        {
            perror("init 2 semget()"); return false;
        }
    }

    return true;
}

bool CSEM::destroy()
{
    if (semctl(sem_id, 0, IPC_RMID) == -1) { perror("destroy semctl()"); return false; }

    return true;
}

bool CSEM::wait()
{
    struct sembuf sem_b;
    sem_b.sem_num = 0;
    sem_b.sem_op = -1;
    sem_b.sem_flg = SEM_UNDO;
    if (semop(sem_id, &sem_b, 1) == -1) { perror("wait semop()"); return false; }

    return true;
}

bool CSEM::post()
{
    struct sembuf sem_b;
    sem_b.sem_num = 0;
    sem_b.sem_op = 1;
    sem_b.sem_flg = SEM_UNDO;
    if (semop(sem_id, &sem_b, 1) == -1) { perror("post semop()"); return false; }

    return true;
}