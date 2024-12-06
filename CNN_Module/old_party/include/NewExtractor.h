#ifndef NEWEXTRACTOR_H
#define NEWEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>

extern "C" {
#include "cma/libxlnk_cma.h" // allocate CMA
#include "libaxidma.h" // Interface to the AXI DMA
}



#include <chrono>

namespace ORB_SLAM3
{
class KeypointAndDesc{
public:
    uint32_t desc[8] = {0};
    uint32_t posX = 0;
    uint32_t posY = 0;
    uint32_t response = 0;

    KeypointAndDesc();
    KeypointAndDesc(const KeypointAndDesc &kp);

    ~KeypointAndDesc();
};
    
class FPGAextractor{
public:
    FPGAextractor() {}
    FPGAextractor(int nfeatures, float scaleFactor, int nlevels,
                int iniThFAST, int minThFAST);

    ~FPGAextractor() {}

    void extract(const cv::Mat &img, std::vector<KeypointAndDesc> &allKpAndDesc);

    int operator()( cv::InputArray _image, cv::InputArray _mask,
                std::vector<cv::KeyPoint>& _keypoints,
                cv::OutputArray _descriptors, std::vector<int> &vLappingArea);

    int inline GetLevels(){
        return nlevels;}

    float inline GetScaleFactor(){
        return scaleFactor;}

    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }

    void printProfileInfo();

    std::vector<cv::Mat> mvImagePyramid;
private:
    //cma para
    uchar *tx_Buf;
    KeypointAndDesc *rx_Buf;
    unsigned long tx_BufPAddr, rx_BufPAddr;

    //org para
    int nfeatures;
    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;

    std::vector<int> mnFeaturesPerLevel;

    std::vector<int> umax;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;    
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;
};

} // namespace ORB_SLAM3
#endif