/**
 * @file main.cpp
 * @note 此文件为主函数在此调用 如要返回图像值 void 改Mat即可
 * @author lse
 */

#include <iostream>
#include <ctime>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "public.hpp"
#include "use.cpp"
#include <exception>
#include "json.hpp"
#include "yyds.hpp"
// #include "imgprocess.cpp"

struct Params
{
    float speedLow = 1.0;       // 智能车最低速
    float speedHigh = 1.0;      // 智能车最高速
    float speedDown = 0.6;      // 特殊区域降速速度
    float speedBridge = 1.0;    // 坡道（桥）行驶速度
    float speedSlowzone = 1.0;  // 慢行区行驶速度
    float speedGarage = 1.0;    // 出入车库速度
    float runP1 = 0.9;          // 一阶比例系数：直线控制量
    float runP2 = 0.018;        // 二阶比例系数：弯道控制量
    float runP3 = 0.0;          // 三阶比例系数：弯道控制量
    float turnP = 3.5;          // 一阶比例系数：转弯控制量
    float turnD = 3.5;          // 一阶微分系数：转弯控制量
    bool debug = false;         // 调试模式使能
    bool saveImage = false;     // 存图使能
    uint16_t rowCutUp = 10;     // 图像顶部切行
    uint16_t rowCutBottom = 10; // 图像顶部切行
    float disGarageEntry = 0.7; // 车库入库距离(斑马线Image占比)
    bool GarageEnable = true;   // 出入库使能
    bool BridgeEnable = true;   // 坡道使能
    bool FreezoneEnable = true; // 泛行区使能
    bool RingEnable = true;     // 环岛使能
    bool CrossEnable = true;    // 十字使能
    bool GranaryEnable = true;  // 粮仓使能
    bool DepotEnable = true;    // 修车厂使能
    bool FarmlandEnable = true; // 农田使能
    bool SlowzoneEnable = true; // 慢行区使能
    bool flag = false;
    uint16_t circles = 2;                           // 智能车运行圈数
    string pathVideo = "../res/samples/sample.mp4"; // 视频路径
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Params, flag, speedLow, speedHigh, speedDown, speedBridge, speedSlowzone, speedGarage,
                                   runP1, runP2, runP3, turnP, turnD, debug, saveImage, rowCutUp, rowCutBottom, disGarageEntry,
                                   GarageEnable, BridgeEnable, FreezoneEnable, RingEnable, CrossEnable, GranaryEnable, DepotEnable, FarmlandEnable, SlowzoneEnable, circles, pathVideo); // 添加构造函数
};

PerspectiveMapping ipm;
Params params;
// image imgprocess;  // 图像处理

using namespace std;
using namespace cv;
using nlohmann::json;

std::string picturepath;
std::string videopath;
cv::Mat src_tupian;
cv::Mat serchpicture;
cv::Mat _dstImg;
cv::Mat frame;
cv::Mat fram1;

struct MyException : public exception
{
    const char *what() const throw()
    {
        return "error";
    }
};

void video(RoadType roadType)
{
    VideoCapture cap;
    cap.open(videopath);
    if (!cap.isOpened())
        throw MyException();
    while (cap.isOpened())
    {
        cv::Mat frame;
        cap.read(frame);

        // std::shared_ptr<DetectionResult> resultAI =
        // detection->getLastFrame();   // 获取Paddle多线程模型预测数据
        // Mat frame = resultAI->rgb_frame; // 获取原始摄像头图像

        cv::Mat frame1 = binarization.imageBinaryzation(frame);
        // cv::Mat frame2= path1.pathSearch(frame1);
        // cv::Mat frame3=binarization.imageBinaryzation(frame2);
        cv::imshow("image123", frame1);
        cv::Mat frame_imgpro;
        trackRecognition.trackRecognition(frame1); // 赛道线识别
        trackRecognition.drawImage(frame);         // 图像显示赛道线识别结果

        
        // imgprocess.process(frame, frame_imgpro);
        // imshow("imag123", frame_imgpro);

        // [11] 环岛识别与处理
        if (roadType == RoadType::RingHandle ||
            roadType == RoadType::BaseHandle)
        {
            if (ringRecognition.ringRecognition(trackRecognition, frame1))
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

        //[05] 农田区域检测
        // if (motionController.params.FarmlandEnable) // 赛道元素是否使能
        // {
        //   if (roadType == RoadType::FarmlandHandle ||
        //       roadType == RoadType::BaseHandle)
        //   {
        //     if (farmlandDetection.farmlandDetection(trackRecognition,
        //                                             resultAI->predictor_results))
        //     {
        //       roadType = RoadType::FarmlandHandle;
        //       if (motionController.params.debug)
        //       {
        //         Mat imageFarmland =
        //             Mat::zeros(Size(COLSIMAGE, ROWSIMAGE), CV_8UC3); // 初始化图像
        //         farmlandDetection.drawImage(trackRecognition, imageFarmland);
        //         imshow("imageRecognition", imageFarmland);
        //         imshowRec = true;
        //         savePicture(imageFarmland);
        //       }
        //     }
        //     else
        //       roadType = RoadType::BaseHandle;
        //   }
        // }

        controlCenterCal.controlCenterCal(trackRecognition); // 根据赛道边缘信息拟合运动控制中心
        controlCenterCal.drawImage(trackRecognition, frame);
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
            // case RoadType::FarmlandHandle: // 农田区域处理 // 坡道处理
            //     putText(imgaeCorrect, "[10] Farmland", Point(10, 20),
            //         cv::FONT_HERSHEY_TRIPLEX, 0.3, cv::Scalar(0, 255, 0), 1,
            //         CV_AA); // 显示赛道识别类型
            //     break;
        }
        imshow("imageTrack", frame);

        char key = waitKey(30); // 读取视频修改waitkey里面的参数可以修改图片播放的速度
        if (key == 27)
        {
            break;
        }
    }
}

int main()
{
    RoadType roadType = RoadType::BaseHandle;
    picturepath = "H:/opencvdaima/wandao/1812.jpg"; // 图片路径
    // videopath = "H:/opencvdaima/wandao/sample.mp4"; // 视频路径
    // videopath = "C:/Users/lsewcx/Desktop/新项目.mp4"; // 视频路径
    videopath = "C:/Users/lsewcx/Desktop/sample .mp4"; //

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

    video(roadType);

    // 读取json文件方法
    //  string jsonPath="H:/opencvdaima/wandao/motion.json";
    //  std::ifstream config_is(jsonPath);
    //  json js_value;
    //  config_is >> js_value;
    //  params = js_value.get<Params>();
    //  cout<<params.flag;

    // 异常抛出
    //  try
    //  {
    //      video(roadType);
    //  }
    //  catch(MyException& e)
    //  {
    //      std::cout << e.what() << std::endl;
    //  }
}