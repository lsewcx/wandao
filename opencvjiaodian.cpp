
/**
 * @author lse
 * @note opencv各种方法
 * @date 2023-7-27
 */
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

namespace lse
{
    class tuxiang
    {
    public:
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
    };
}