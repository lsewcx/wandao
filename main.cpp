/**
 * @note 此文件为主函数在此调用 如要返回图像值void 改Mat即可
 */
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
    picturepath = "H:/opencvdaima/wandao/1812.jpg"; // 图片路径
    videopath = "H:/opencvdaima/wandao/sample.mp4"; // 视频路径
    // 透视变换的部分
    bool flag = false; // false跑视频  true跑照片 启用视频时需要关闭照片才能运行哦
    if (flag == true)
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