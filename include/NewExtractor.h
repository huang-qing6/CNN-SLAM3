#ifndef NEWEXTRACTOR_H
#define NEWEXTRACTOR_H

#include "torch/script.h"
#include "torch/torch.h"

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>

// Compile will fail for opimizier since pytorch defined this, from GNN_SLAMv2
/*#ifdef EIGEN_MPL2_ONLY
#undef EIGEN_MPL2_ONLY
#endif*/

namespace ORB_SLAM3{

    /* 这里删掉了描述子的部分 */

class CNNextractor{
    
public:
    
    CNNextractor(){};
    CNNextractor(int nfeatures, float scaleFactor, int nlevels,
                int initThFAST, int minThFAST);

    ~CNNextractor(){};

    int operator()( cv::InputArray _image, cv::InputArray _mask,
                std::vector<cv::KeyPoint>& _keypoints,
                cv::OutputArray _descriptors, std::vector<int> &vLappingArea);


    /* 保留但不实例化，保证其他部分不需要更改 */
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

    std::vector<cv::Mat> mvImagePyramid;

protected:
    /* 这里删除图像金字塔的部分函数，只保留了参数 */

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

    // 新增 CNN module
    /*std::shared_ptr<*/torch::jit::script::Module/*>*/ module; // 感觉还是需要弄上shraed ptr

};

} // namespace CNN_SLAM3(OG:ORB_SLAM3)
#endif