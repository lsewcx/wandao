#include "use.cpp"
#include <iostream>
#include <ctime>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
PerspectiveMapping ipm;

using namespace std;
using namespace cv;

std::string path;
int main()
{
    bool flag = false; // false跑照片  true跑视频 启用视频时需要关闭照片才能运行哦
    if (flag == false)
    {
        path = "H:/opencvdaima/wandao/1812.jpg"; // 照片路径
        ipm.init(Size(320, 240),
                 Size(320, 400), flag, path);
    }
    else
    {
        path = "H:/opencvdaima/sample.mp4"; // 视频路径
        ipm.init(Size(320, 240),
                 Size(320, 400), flag, path);
    }

    return 0;
}