/**
 * @note 此文件为主函数在此调用 如要返回图像值void 改Mat即可
 */

#include <iostream>
#include <ctime>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "public.hpp"
#include "use.cpp"

PerspectiveMapping ipm;
using namespace std;
using namespace cv;

std::string picturepath;
std::string videopath;
cv::Mat src_tupian;
cv::Mat serchpicture;
cv::Mat _dstImg;
cv::Mat frame;
cv::Mat fram1;



int main()
{
    RoadType roadType = RoadType::BaseHandle; 
    picturepath = "H:/opencvdaima/wandao/1812.jpg"; // 图片路径
    videopath = "H:/opencvdaima/wandao/sample.mp4"; // 视频路径


    // 透视变换的部分
    // bool flag = false; // false跑视频  true跑照片 启用视频时需要关闭照片才能运行哦
    // if (flag == true)
    // {
    //     ipm.init(Size(320, 240),
    //              Size(320, 400), flag, picturepath);
    // }
    // else
    // {
    //     ipm.init(Size(320, 240),
    //              Size(320, 400), flag,videopath);
    // }


    VideoCapture cap;
    cap.open(videopath);
    while(cap.isOpened())
    {
        cv::Mat frame;
        cap.read(frame);
        cv::Mat frame1= binarization.imageBinaryzation(frame);
        cv::Mat frame2= path1.pathSearch(frame1);
        cv::Mat frame3=binarization.imageBinaryzation(frame2);
        cv::imshow("image123",frame3);


         // [04] 出库和入库识别与路径规划

        // [11] 环岛识别与处理

        if (roadType == RoadType::RingHandle ||
          roadType == RoadType::BaseHandle)
        {
            if (ringRecognition.ringRecognition(trackRecognition,frame3))
            {
                roadType = RoadType::RingHandle;
            
                Mat imageRing =
                    Mat::zeros(Size(COLSIMAGE, ROWSIMAGE), CV_8UC3); // 初始化图像
                ringRecognition.drawImage(trackRecognition, imageRing);
                imshow("imageRecognition", imageRing);
            }
            else
            roadType = RoadType::BaseHandle;
        }

        trackRecognition.trackRecognition(frame3); // 赛道线识别
        trackRecognition.drawImage(frame); // 图像显示赛道线识别结果
        controlCenterCal.controlCenterCal(trackRecognition); // 根据赛道边缘信息拟合运动控制中心
        controlCenterCal.drawImage(trackRecognition, frame);
        imshow("imageTrack", frame);
        switch (roadType)
        {
            case RoadType::BaseHandle: // 基础赛道处理 // 基础赛道处理
                putText(frame, "[1] Track", Point(10, 20),
                        cv::FONT_HERSHEY_TRIPLEX, 0.3, cv::Scalar(0, 0, 255), 1,
                        CV_AA); // 显示赛道识别类型
                break;
            case RoadType::RingHandle: // 环岛赛道处理 // 环岛赛道处理
                putText(frame, "[2] Ring", Point(10, 20),
                        cv::FONT_HERSHEY_TRIPLEX, 0.3, cv::Scalar(0, 255, 0), 1,
                        CV_AA); // 显示赛道识别类型
                break;
        }


        char key = waitKey(10);//读取视频
        if (key == 27)
        {
            break;
        }
    }
}