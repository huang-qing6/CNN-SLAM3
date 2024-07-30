#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>

#include "NewExtractor.h"

using namespace cv;
using namespace std;

namespace ORB_SLAM3
{
    /*const int PATCH_SIZE = 31;
    const int HALF_PATCH_SIZE = 15;
    const int EDGE_THRESHOLD = 19;*/

    //dma para
    bool first_init = true; 
    int num_transfers;
    int tx_channel, rx_channel;
    size_t tx_size, rx_size;
    //char *tx_buf, *rx_buf;
    axidma_dev_t axidma_dev;
    const array_t *tx_chans, *rx_chans;
    struct axidma_video_frame transmit_frame, *tx_frame, receive_frame, *rx_frame;

    KeypointAndDesc::KeypointAndDesc() {}
    KeypointAndDesc::KeypointAndDesc(const KeypointAndDesc &kp){
        for(int i=0; i<8; i++)
            this->desc[i] = kp.desc[i];
        this->posX = kp.posX;
        this->posY = kp.posY;
        this->response = kp.response;
    }
    KeypointAndDesc::~KeypointAndDesc() {}

 //初始化图像金字塔
FPGAextractor::FPGAextractor( int _nfeatures, float _scaleFactor, int _nlevels,
                               int _iniThFAST, int _minThFAST):
            nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
            iniThFAST(_iniThFAST), minThFAST(_minThFAST)
{
        // 待定传出/传入
        int imgsize = 1920 * 1080; // 图像FPS
        int keypoints = 512; // 传回特征点数量

    #pragma region 初始化DMA驱动
        printf("*****DMA Driver Initalized*******\n");
        if(first_init){
            tx_channel = -1; // default tx_channel num
            rx_channel = -1; // default rx_channel num
            axidma_dev = axidma_init();
                if (axidma_dev == NULL) {
                fprintf(stderr, "Failed to initialize the AXI DMA device.\n");
                goto ret;
            }

            // vDMA 初始化，暂时不用
            tx_frame = NULL;
            rx_frame = NULL;
            tx_size = imgsize * sizeof(int);
            rx_size = keypoints * sizeof(KeypointAndDesc);
            first_init = false;

            // get the channel number
            tx_chans = axidma_get_dma_tx(axidma_dev);
            rx_chans = axidma_get_dma_rx(axidma_dev);
            if (tx_chans->len < 1) {
                fprintf(stderr, "Error: No transmit channels were found.\n");
            }
            if (rx_chans->len < 1) {
                fprintf(stderr, "Error: No receive channels were found.\n");
            }

            /** 作为常规设计根据调用驱动返回来定义tx、rx名字比较符合软件设计，
             *  也可自己在channel初始化直接定义tx = 0, rx = 1
             *  */
            if (tx_channel == -1 && rx_channel == -1) {
                tx_channel = tx_chans->data[0];
                rx_channel = rx_chans->data[0];
            }

            printf("init driver finshed!\n");
        }

        // 初始化cma地址
        tx_Buf = (uchar*)cma_alloc(tx_size , 0);
        tx_BufPAddr = cma_get_phy_addr((void*)tx_Buf);
        rx_Buf = (KeypointAndDesc*)cma_alloc(rx_size , 0);
        rx_BufPAddr = cma_get_phy_addr((void*)rx_Buf);
        printf("srcBuf: %x, srcBufPAddr: %x\n", tx_Buf, tx_BufPAddr);
        printf("dstBuf: %x, dstBufPAddr: %x\n", rx_Buf, rx_BufPAddr);
    
        printf("*****DMA Driver initalized finished!******\n");
    #pragma endregion

    #pragma region 图像金字塔初始化 想删掉
            /*为什么保留了orbextractor？ 答：懒得改了的屎山代码
        printf("*****ORBExtractor constructor*****\n");
            mvScaleFactor.resize(nlevels);
            mvLevelSigma2.resize(nlevels);
            mvScaleFactor[0]=1.0f;
            mvLevelSigma2[0]=1.0f;
            for(int i=1; i<nlevels; i++)
            {
                mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
                mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
            }

            mvInvScaleFactor.resize(nlevels);
            mvInvLevelSigma2.resize(nlevels);
            for(int i=0; i<nlevels; i++)
            {
                mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
                mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
            }

            mvImagePyramid.resize(nlevels);

            mnFeaturesPerLevel.resize(nlevels);
            float factor = 1.0f / scaleFactor;
            float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

            int sumFeatures = 0;
            for( int level = 0; level < nlevels-1; level++ )
            {
                mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
                sumFeatures += mnFeaturesPerLevel[level];
                nDesiredFeaturesPerScale *= factor;
            }
            mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);
            printf("*****ORBExtractor Constructor******\n");*/
    #pragma endregion
        ret:
            printf("*****init dma channel faild, close!~******\n");

    }

    void  FPGAextractor::extract(const Mat &img, vector<KeypointAndDesc> &allKpAndDesc){
        double count;
        int bytesRecvd;
        int Keypoint;

        //memcpy 复制 img.data的地址size个字节给srcbuf
        memcpy(tx_Buf , (void*)img.data, tx_size); //从img.data输入到tx_Buf，tx_size大小的数据

        // Initialize the buffer region we're going to transmit
        //init_data(tx_Buf, rx_Buf, tx_size, rx_size);

        // Perform the DMA transaction
        axidma_twoway_transfer(axidma_dev, tx_channel, tx_Buf, tx_size, tx_frame,
            rx_channel, rx_Buf, rx_size, rx_frame, true);

        // TODO 接收数据部分还没写

        // 读取数据需要切换
        /*bytesRecvd = desc.getBytesRecvd(); 
        Keypoint = (bytesRecvd / sizeof(KeypointAndDesc));*/

        // int offsetInKpAndDesc = 0;
        // allKpAndDesc.reserve(nlevels); // change the nlevels 

        //allKpAndDesc = vector<KeypointAndDesc>(dstBuf + offsetInKpAndDesc, dstBuf + offsetInKpAndDesc + Keypoint); //保存关键数据！！！
        allKpAndDesc = vector<KeypointAndDesc>(rx_Buf , rx_Buf + Keypoint); //new 保存关键数据！！！
    }

    int FPGAextractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                                  OutputArray _descriptors, std::vector<int> &vLappingArea) //相比slam2 新增vLapping参数
    {
        double count;
        auto t0 = chrono::steady_clock::now();

        if(_image.empty())
            return 0;

        Mat image = _image.getMat();
        assert(image.type() == CV_8UC1);

        vector<KeypointAndDesc> allKeypointAndDescs;

        printf("using dma!\n");
        FPGAextractor::extract(image, allKeypointAndDescs);
        printf("finish!\n");

        Mat descriptors;

        int nkeypoints =  (int)allKeypointAndDescs.size();//如果完全不需要处理特征点就可以这样子

        //如果有特征就构建cv_8u的32列描述子
        if( nkeypoints == 0 )
            _descriptors.release();
        else
        {
            _descriptors.create(nkeypoints, 32, CV_8U);
            descriptors = _descriptors.getMat();
        }

        //orb_slam2是有一个清空初始化，这里直接利用nkeypoints
        _keypoints = vector<cv::KeyPoint>(nkeypoints);

        //Modified for speeding up stereo fisheye matching
        int monoIndex = 0, stereoIndex = nkeypoints-1;

        //vector<KeypointAndDesc>& keypoints = distributedKeypointAndDescs;
        vector<KeypointAndDesc>& keypoints = allKeypointAndDescs;

        int nkeypointsLevel = (int)keypoints.size();

        Mat desc = cv::Mat(nkeypointsLevel, 32, CV_8U);

        //float scale = mvScaleFactor; //getScale(level, firstLevel, scaleFactor);
        int i = 0;
        for (vector<KeypointAndDesc>::iterator keypoint = keypoints.begin(), keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint){
            if(keypoint->posX >= vLappingArea[0] && keypoint->posX <= vLappingArea[1]){
                //_keypoints.at(stereoIndex) = (*keypoint); //没有处理好
                desc.row(i).copyTo(descriptors.row(stereoIndex));
                stereoIndex--;
            }
            else{
                //_keypoints.at(monoIndex) = (*keypoint);
                desc.row(i).copyTo(descriptors.row(monoIndex));
                monoIndex++;            
            }
        } 
        return monoIndex;
    }
}//namespace ORB_SLAM