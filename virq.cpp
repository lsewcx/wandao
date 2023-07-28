/**
 * @author lse
 * @note 虚函数使用方法用来重写另一段class中的方法
 **/
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
class bass
{
public:
    virtual void abc()
    {
        int a = 0;
        a++;
        std::cout << a << endl;
    }
};
class deas : public bass
{
public:
    void abc() override
    {
        int b = 0;
        b = 8;
        std::cout << b << endl;
    }
};
int main()
{
    bass *xuhanshu = new deas();
    xuhanshu->abc();
    delete xuhanshu;
    deas hu;
    hu.abc();
    system("pause");
    return 0;
}