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
#include "cross_recognition.cpp" //十字道路识别与路径规划类
#include <ctime>
#include "Timer.cpp"
#include <cmath>
#include <thread>

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
    bool flag1 = false;
    uint16_t circles = 2;                           // 智能车运行圈数
    string pathVideo = "../res/samples/sample.mp4"; // 视频路径
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Params, flag1, speedLow, speedHigh, speedDown, speedBridge, speedSlowzone, speedGarage,
                                   runP1, runP2, runP3, turnP, turnD, debug, saveImage, rowCutUp, rowCutBottom, disGarageEntry,
                                   GarageEnable, BridgeEnable, FreezoneEnable, RingEnable, CrossEnable, GranaryEnable, DepotEnable, FarmlandEnable, SlowzoneEnable, circles, pathVideo); // 添加构造函数
};

PerspectiveMapping ipm;
Params params;
CrossroadRecognition crossroadRecognition;
Timer timer;
bool crosscounter = false;
bool ringst = true;
cv::Mat HighLight(cv::Mat input, int light);
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
int num = 0;
RoadType roadtemptype = RoadType::CrossHandle;

struct MyException : public exception
{
    const char *what() const throw()
    {
        return "error";
    }
};

cv::Mat removeHighlight(const cv::Mat &inputImage)
{
    cv::Mat hsvImage;
    cv::cvtColor(inputImage, hsvImage, cv::COLOR_BGR2HSV); // 转换为HSV颜色空间

    cv::Mat mask;
    cv::inRange(hsvImage, cv::Scalar(0, 0, 180), cv::Scalar(255, 255, 255), mask); // 创建掩膜以定位高亮区域

    cv::Mat result;
    cv::bitwise_not(mask, mask);                           // 反转掩膜
    cv::bitwise_and(inputImage, inputImage, result, mask); // 应用掩膜

    return result;
}

cv::Mat removeReflection(const cv::Mat &inputImage, int blockSize, double C)
{
    // 将输入图像转换为灰度图像
    cv::Mat grayImage;
    cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);

    // 应用自适应阈值化方法
    cv::Mat binaryImage;
    cv::adaptiveThreshold(grayImage, binaryImage, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, blockSize, C);

    return binaryImage;
}

cv::Mat removeReflection(const cv::Mat &inputImage, double threshold)
{
    // 将输入图像转换为灰度图像
    cv::Mat grayImage;
    cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);

    // 计算图像梯度
    cv::Mat gradX, gradY;
    cv::Sobel(grayImage, gradX, CV_32F, 1, 0);
    cv::Sobel(grayImage, gradY, CV_32F, 0, 1);

    // 计算梯度幅值
    cv::Mat gradient;
    cv::magnitude(gradX, gradY, gradient);

    // 对梯度幅值进行二值化
    cv::Mat binaryImage;
    cv::threshold(gradient, binaryImage, threshold, 255, cv::THRESH_BINARY);

    return binaryImage;
}

cv::Mat removeReflection(const cv::Mat &inputImage, int blockSize, int c)
{
    // 将输入图像转换为灰度图像
    cv::Mat grayImage;
    cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);

    // 对图像进行自适应阈值化
    cv::Mat binaryImage;
    cv::adaptiveThreshold(grayImage, binaryImage, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, blockSize, c);

    // 使用形态学操作进行反射消除
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_OPEN, kernel);

    return binaryImage;
}

cv::Mat reduceIllumination(const cv::Mat &inputImage, int threshold)
{
    // 将输入图像转换为灰度图像
    cv::Mat grayImage;
    cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);

    // 对灰度图像进行自适应阈值化
    cv::Mat binaryImage;
    cv::adaptiveThreshold(grayImage, binaryImage, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 21, 5);

    // 创建掩膜图像
    cv::Mat mask = binaryImage < threshold;

    // 根据掩膜图像对输入图像进行减弱处理
    cv::Mat reducedImage;
    inputImage.copyTo(reducedImage, mask);

    return reducedImage;
}

cv::Mat enhanceExposure(const cv::Mat &inputImage, double gamma)
{
    cv::Mat outputImage;
    // 将图像转换为浮点型
    inputImage.convertTo(outputImage, CV_64F);

    // 对图像进行幂次变换
    cv::pow(outputImage, gamma, outputImage);

    // 将图像转换回8位无符号整型
    outputImage.convertTo(outputImage, CV_8U);

    return outputImage;
}

void illuminationChange(cv::Mat &image)
{
    // 将图像转换为Lab颜色空间
    cv::Mat labImage;
    cv::cvtColor(image, labImage, cv::COLOR_BGR2Lab);

    // 分离通道
    std::vector<cv::Mat> labChannels(3);
    cv::split(labImage, labChannels);

    // 对L通道进行直方图均衡化
    cv::equalizeHist(labChannels[0], labChannels[0]);

    // 合并通道
    cv::merge(labChannels, labImage);

    // 将图像转换回BGR颜色空间
    cv::cvtColor(labImage, image, cv::COLOR_Lab2BGR);
}

uint16_t circlesThis = 1;    // 智能车当前运行的圈数
uint16_t countercircles = 0; // 圈数计数器
double totalBrightness = 0;
int frameCount = 0;
void video(RoadType roadType)
{
    VideoCapture cap;
    cap.open(videopath);
    if (!cap.isOpened())
        throw MyException();
    while (cap.isOpened())
    {
        cv::Mat frame;
        cv::Mat frame123;
        cv::Mat frame1234;
        cap.read(frame);
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        double frameBrightness = cv::mean(gray)[0];

        // 累加亮度值
        totalBrightness += frameBrightness;
        frameCount++;

        // 显示当前帧亮度
        std::cout << "Frame " << frameCount << " brightness: " << frameBrightness << std::endl;
        int light1 = -50;
        cv::Mat fm = frame.clone();
        cv::Mat result1 = HighLight(frame, light1);
        //  illuminationChange(frame);
        // reduceReflection(frame);
        // cv::equalizeHist(frame,frame123);
        // cv::Mat outimagr = removeHighlight(frame);
        // cv::imshow("iameufua",outimagr);
        // cv::Mat frame2= path1.pathSearch(frame1);

        // 自适应
        //  int blockSize = 11; // 块大小（奇数值），用于自适应阈值化
        //  double C = 2; // 常数C，用于控制阈值偏移
        //  cv::Mat outputImage = removeReflection(frame, blockSize, C);

        // 二值化
        //  double threshold = 40; // 阈值，根据实际情况进行调整
        //  cv::Mat outputImage = removeReflection(frame, threshold);

        // 二值化
        //  int blockSize = 21; // 块大小（奇数值），用于自适应阈值化
        //  int c = 3; // 常数C，用于控制阈值偏移
        //  cv::Mat outputImage = removeReflection(frame, blockSize, c);

        // int threshold = 128; // 阈值，用于控制减弱效果
        // cv::Mat outputImage = reduceIllumination(frame, threshold);

        // 曝光
        // double gamma = 0.01; // 曝光增强参数
        // cv::Mat outputImage = enhanceExposure(frame, gamma);
        cv::Mat frame12345 = binarization.imageBinaryzation(result1);
        cv::Mat frame12345678 = binarization.imageBinaryzation(fm);
        cv::imshow("1231313", frame12345);
        cv::imshow("12313134545", frame12345678);

        // cv::imshow("Output Image", outputImage);

        // cv::Mat frame1 = binarization.imageBinaryzation(frame);
        // cv::Mat frame2 = path1.pathSearch(frame1);
        // cv::imshow("1231313",frame2);
        // cv::equalizeHist(frame1, frame123);

        // // 使用双边滤波器处理反射
        // cv::Mat reflection;
        // cv::bilateralFilter(frame123, reflection, -1, 15, 15);
        // cv::Mat reflection_removed = frame123 - reflection;

        // // 局部对比度增强
        // cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        // clahe->setClipLimit(4.0);
        // cv::Mat enhanced_image;
        // clahe->apply(reflection_removed, enhanced_image);

        // // 转换回BGR颜色空间
        // cv::Mat final_image;
        // cv::cvtColor(enhanced_image, final_image, cv::COLOR_Lab2BGR);
        // cv::imshow("image12345", final_image);

        // // cv::GaussianBlur(frame1,frame123,cv::Size(5,5),0);
        // cv::cvtColor(frame123,frame1234,cv::COLOR_GRAY2BGR);
        // // cv::imshow("fadmamda",frame123);
        // std::vector<cv::Mat> lab_channels;
        // cv::split(frame1234,lab_channels);
        // cv::Mat reflection;
        // cv::bilateralFilter(lab_channels[0], reflection, -1, 15, 15);
        // cv::Mat reflection_removed = lab_channels[0] - reflection;

        // std::vector<cv::Mat> output_channels;
        // output_channels.push_back(reflection_removed);
        // output_channels.push_back(lab_channels[1]);
        // output_channels.push_back(lab_channels[2]);

        // cv::Mat output_image;
        // cv::merge(output_channels, output_image);

        // // 转换回BGR颜色空间
        // cv::Mat final_image;
        // cv::cvtColor(output_image, final_image, cv::COLOR_Lab2BGR);
        // cv::imshow("image12345", outimagr);

        // Mat gray;
        // cvtColor(frame, gray, COLOR_BGR2GRAY);
        // GaussianBlur(gray, gray, Size(5, 5), 0);
        // cv::Mat frame3=binarization.imageBinaryzation(frame2);
        // cv::imshow("image1234", frame1);
        // cv::imshow("image123", final_image);
        cv::Mat frame_imgpro;
        trackRecognition.trackRecognition(frame12345); // 赛道线识别
        // cv::imshow("2131313",frame12345);
        trackRecognition.drawImage(frame); // 图像显示赛道线识别结果

        if (roadType == RoadType::GarageHandle ||
            roadType == RoadType::BaseHandle)
        {
            countercircles++; // 圈数计数
            if (countercircles > 200)
                countercircles = 200;
        }

        // [11] 环岛识别与处理
        if (roadType == RoadType::RingHandle ||
            roadType == RoadType::BaseHandle && ringst)
        {
            if (ringRecognition.ringRecognition(trackRecognition, frame))
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

        // [12] 十字道路处理
        if (roadType == RoadType::CrossHandle || roadType == RoadType::BaseHandle)
        {
            if (crossroadRecognition.crossroadRecognition(
                    trackRecognition))
            {
                roadType = RoadType::CrossHandle;
                crosscounter = true;
                timer.reset();
                timer.start();
                Mat imageCross =
                    Mat::zeros(Size(COLSIMAGE, ROWSIMAGE), CV_8UC3); // 初始化图像
                crossroadRecognition.drawImage(trackRecognition, imageCross);
                imshow("imageRecognition", imageCross);
            }
            else
                roadType = RoadType::BaseHandle;
        }

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
        case RoadType::CrossHandle: // 十字道路处理 // 十字道路处理
            putText(frame, "[3] Cross", Point(10, 20),
                    cv::FONT_HERSHEY_TRIPLEX, 0.3, cv::Scalar(0, 255, 0), 1,
                    CV_AA); // 显示赛道识别类型
            break;
        }
        imshow("imageTrack", frame);

        char key = waitKey(30); // 读取视频修改waitkey里面的参数可以修改图片播放的速度
        if (key == 27)
        {
            break;
        }
    }
    double averageBrightness = totalBrightness / frameCount;

    std::cout << "Average brightness: " << averageBrightness << std::endl;
}

int main()
{
    RoadType roadType = RoadType::BaseHandle;
    picturepath = "H:/opencvdaima/wandao/1812.jpg"; // 图片路径
    // videopath = "H:/opencvdaima/wandao/sample.mp4"; // 视频路径
    // videopath = "H:/opencvdaima/wandao/video1.mp4"; // 视频路径
    videopath = "C:/Users/lsewcx/Desktop/新项目.mp4"; // 视频路径
    // videopath = "H:/opencvdaima/wandao/1233.mp4"; //

    // video(roadType);

    // 读取json文件方法
    //  string jsonPath="H:/opencvdaima/wandao/123/motion.json";
    //  std::ifstream config_is(jsonPath);
    //  json js_value;
    //  config_is >> js_value;
    //  params = js_value.get<Params>();

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

    // 异常抛出
    try
    {
        video(roadType);
    }
    catch (MyException &e)
    {
        std::cout << e.what() << std::endl;
    }
}

// 图像高光选取
cv::Mat HighLight(cv::Mat input, int light)
{
    // 生成灰度图
    cv::Mat gray = cv::Mat::zeros(input.size(), CV_32FC1);
    cv::Mat f = input.clone();
    f.convertTo(f, CV_32FC3);
    vector<cv::Mat> pics;
    split(f, pics);
    gray = 0.299f * pics[2] + 0.587 * pics[1] + 0.114 * pics[0];
    gray = gray / 255.f;

    // 确定高光区
    cv::Mat thresh = cv::Mat::zeros(gray.size(), gray.type());
    thresh = gray.mul(gray);
    // 取平均值作为阈值
    Scalar t = mean(thresh);
    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
    mask.setTo(255, thresh >= t[0]);

    // 参数设置
    int max = 4;
    float bright = light / 100.0f / max;
    float mid = 1.0f + max * bright;

    // 边缘平滑过渡
    cv::Mat midrate = cv::Mat::zeros(input.size(), CV_32FC1);
    cv::Mat brightrate = cv::Mat::zeros(input.size(), CV_32FC1);
    for (int i = 0; i < input.rows; ++i)
    {
        uchar *m = mask.ptr<uchar>(i);
        float *th = thresh.ptr<float>(i);
        float *mi = midrate.ptr<float>(i);
        float *br = brightrate.ptr<float>(i);
        for (int j = 0; j < input.cols; ++j)
        {
            if (m[j] == 255)
            {
                mi[j] = mid;
                br[j] = bright;
            }
            else
            {
                mi[j] = (mid - 1.0f) / t[0] * th[j] + 1.0f;
                br[j] = (1.0f / t[0] * th[j]) * bright;
            }
        }
    }

    // 高光提亮，获取结果图
    cv::Mat result = cv::Mat::zeros(input.size(), input.type());
    for (int i = 0; i < input.rows; ++i)
    {
        float *mi = midrate.ptr<float>(i);
        float *br = brightrate.ptr<float>(i);
        uchar *in = input.ptr<uchar>(i);
        uchar *r = result.ptr<uchar>(i);
        for (int j = 0; j < input.cols; ++j)
        {
            for (int k = 0; k < 3; ++k)
            {
                float temp = pow(float(in[3 * j + k]) / 255.f, 1.0f / mi[j]) * (1.0 / (1 - br[j]));
                if (temp > 1.0f)
                    temp = 1.0f;
                if (temp < 0.0f)
                    temp = 0.0f;
                uchar utemp = uchar(255 * temp);
                r[3 * j + k] = utemp;
            }
        }
    }
    return result;
}