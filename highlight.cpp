#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

cv::Mat HighLight(cv::Mat input, int light);

int main()
{
    VideoCapture cap;
    string videopath = "C:/Users/lsewcx/Desktop/新项目.mp4";
    cap.open(videopath);
    while (cap.isOpened())
    {
        cv::Mat src;
        cap.read(src);
        int light1 = 50;
        int light2 = -50;
        cv::Mat result1 = HighLight(src, light1);
        cv::Mat result2 = HighLight(src, light2);
        imshow("original", src);
        // imshow("result1", result1);
        imshow("result2", result2);
        char key = waitKey(30); // 读取视频修改waitkey里面的参数可以修改图片播放的速度
        if (key == 27)
        {
            break;
        }
    }
    return 0;
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