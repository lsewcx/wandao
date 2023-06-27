#include "use.cpp"
#include <iostream>
#include <stdio.h>
#include <ctime>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
PerspectiveMapping ipm;

using namespace cv;
int main()
{
    bool flag = false; // false跑照片  true跑视频 启用视频时需要关闭照片才能运行哦
    ipm.init(Size(320, 240),
             Size(320, 400), flag);
    return 0;
}