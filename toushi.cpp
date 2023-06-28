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

std::string picturepath;
std::string videopath;
cv::Mat src_tupian;
cv::Mat serchpicture;
cv::Mat _dstImg;

int main()
{
    picturepath = "H:/opencvdaima/wandao/1812.jpg";
    videopath = "H:/opencvdaima/wandao/sample.mp4"; // 视频路径
    // 透视变换的部分
    bool flag = true; // false跑照片  true跑视频 启用视频时需要关闭照片才能运行哦
    if (flag == false)
    {
        ipm.init(Size(320, 240),
                Size(320, 400), flag, picturepath);
    }
    else
    {
        ipm.init(Size(320, 240),
            Size(320, 400), flag, videopath);
    }
    return 0;
}