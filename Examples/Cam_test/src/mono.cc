#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<opencv2/core/core.hpp>
#include"../../../include/System.h"

#include <unistd.h> // Linux系统下的头文件，用于键盘输入
#include<termios.h>
#include<fcntl.h>

using namespace std;

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM):mpSLAM(pSLAM){}

    void GrabImage(cv::Mat&image);

    ORB_SLAM3::System* mpSLAM;
};

// 函数声明
bool kbhit();
char getch();

int main(int argc, char **argv)
{
    if(argc != 3)
    {
        cerr << endl << "Usage:Mono path_to_vocabulary path_to_settings" << endl;        
        return 1;
    }    

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    //不用Pangolin展示
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::MONOCULAR,false);

    ImageGrabber igb(&SLAM);

    cv::VideoCapture cap(0);

    if(!cap.isOpened()){
        cerr << "Error opening video stream or file" << endl;
        return 1;
    }

    cv::Mat frame;
    bool stop = false;
    while(!stop){
        cap >> frame;
        if(frame.empty()) break;

        igb.GrabImage(frame);

        // 检测键盘输入，按下特定键时停止并保存
        if (kbhit()) {
            char c = getch();
            if (c == 's' || c == 'S') {
                cout << "Saving trajectory..." << endl;
                SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
                stop = true;
            }
        }
    }

    // Stop all threads
    SLAM.Shutdown();

    return 0;
}

void ImageGrabber::GrabImage(cv::Mat&image)
{
    mpSLAM->TrackMonocular(image, chrono::system_clock::now().time_since_epoch().count()/1e9);
}

// 实现 kbhit 函数
bool kbhit() {
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if(ch != EOF) {
        ungetc(ch, stdin);
        return true;
    }

    return false;
}

// 实现 getch 函数
char getch() {
    char buf = 0;
    struct termios old = {0};
    if (tcgetattr(0, &old) < 0)
        perror("tcsetattr()");
    old.c_lflag &= ~ICANON;
    old.c_lflag &= ~ECHO;
    old.c_cc[VMIN] = 1;
    old.c_cc[VTIME] = 0;
    if (tcsetattr(0, TCSANOW, &old) < 0)
        perror("tcsetattr ICANON");
    if (read(0, &buf, 1) < 0)
    {
        perror ("read()");
    }
    old.c_lflag |= ICANON;
    old.c_lflag |= ECHO;
    if (tcsetattr(0, TCSADRAIN, &old) < 0)
        perror ("tcsetattr ~ICANON");
    return (buf);
}
