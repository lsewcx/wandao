#include <thread>
#include <iostream>
using namespace std;


void threadMain()
{
    cout<<"start"<<this_thread::get_id()<<endl;
}
int main()
{
    cout<<"Main" << this_thread::get_id()<<endl;
    //线程创建启动
    thread th(threadMain);
    th.join();
    return 0;
}