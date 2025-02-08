#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>

#include "CNNextractor.h"

#include <chrono>

using namespace cv;
using namespace std;
namespace F = torch::nn::functional;

namespace ORB_SLAM3{

    const int PATCH_SIZE = 31;
    const int HALF_PATCH_SIZE = 15;
    const int EDGE_THRESHOLD = 19;

    /**非极大值抑制,
     * Input:
     *  keypoint [W H]
     *  desc [256 W*H]
     *  keypoint_res 特征点结果
     *  descriptors 描述子结果
     */ 
    void CNN_nms(std::vector<cv::KeyPoint> keypoint, cv::Mat conf, cv::Mat desc, 
            std::vector<cv::KeyPoint>& keypoint_res, cv::Mat& descriptors, 
            int border, int dist_thresh, int img_width, int img_height, float ratio_width, float ratio_height){
        std::vector<cv::Point2f> pts_raw;

        for (int i = 0; i < keypoint.size(); i++){
            int u = (int) keypoint[i].pt.x;
            int v = (int) keypoint[i].pt.y;

            pts_raw.push_back(cv::Point2f(u, v));
        }

        cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
        cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);

        grid.setTo(0);
        inds.setTo(0);

        for (int i = 0; i < pts_raw.size(); i++)
        {   
            int uu = (int) pts_raw[i].x;
            int vv = (int) pts_raw[i].y;

            grid.at<char>(vv, uu) = 1;
            inds.at<unsigned short>(vv, uu) = i;
        }

        cv::copyMakeBorder(grid, grid, dist_thresh, dist_thresh, dist_thresh, dist_thresh, cv::BORDER_CONSTANT, 0);      
        for (int i = 0; i < pts_raw.size(); i++)
        {   
            int uu = (int) pts_raw[i].x + dist_thresh;
            int vv = (int) pts_raw[i].y + dist_thresh;

            if (grid.at<char>(vv, uu) != 1)
                continue;

            for(int k = -dist_thresh; k < (dist_thresh+1); k++)
                for(int j = -dist_thresh; j < (dist_thresh+1); j++)
                {
                    if(j==0 && k==0) continue;

                    if ( conf.at<float>(vv + k, uu + j) < conf.at<float>(vv, uu) )
                        grid.at<char>(vv + k, uu + j) = 0;
                    
                }
            grid.at<char>(vv, uu) = 2;
        }

        size_t valid_cnt = 0;
        std::vector<int> select_indice;

        for (int v = 0; v < (img_height + dist_thresh); v++){
            for (int u = 0; u < (img_width + dist_thresh); u++)
            {
                if (u -dist_thresh>= (img_width - border) || u-dist_thresh < border || v-dist_thresh >= (img_height - border) || v-dist_thresh < border)
                continue;

                if (grid.at<char>(v,u) == 2)
                {
                    int select_ind = (int) inds.at<unsigned short>(v-dist_thresh, u-dist_thresh);
                    keypoint_res.push_back(cv::KeyPoint(pts_raw[select_ind].x * ratio_width, pts_raw[select_ind].y * ratio_height, 1.0f));

                    select_indice.push_back(select_ind);
                    valid_cnt++;
                }
            }
        }
        
        descriptors.create(select_indice.size(), 256, CV_32F);

        for (int i=0; i<select_indice.size(); i++)
        {
            for (int j=0; j<256; j++)
            {
                // 保存的特征点对应的描述子信息
                descriptors.at<float>(i, j) = desc.at<float>(select_indice[i], j);
            }
        }       

    }

    CNNextractor::CNNextractor(int _nfeatures, float _scaleFactor, int _nlevels,
                               float _iniThFAST, float _minThFAST):
            nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
            iniThFAST(_iniThFAST), minThFAST(_minThFAST)
    {
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

        //This is for orientation
        // pre-compute the end of a row in a circular patch
        umax.resize(HALF_PATCH_SIZE + 1);

        int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
        int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
        const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
        for (v = 0; v <= vmax; ++v)
            umax[v] = cvRound(sqrt(hp2 - v * v));

        // Make sure we are symmetric
        for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
        {
            while (umax[v0] == umax[v0 + 1])
                ++v0;
            umax[v] = v0;
            ++v0;
        }

        // 初始化CNN，指定路径可以更灵活
        try { 
            module = torch::jit::load("./Netsuper_ptq.pt"); 
        } catch (const c10::Error& e) { 
            std::cerr << "Error loading the model.\n"; 
        }
    }

    int CNNextractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                                  OutputArray _descriptors, std::vector<int> &vLappingArea)
    {
        if(_image.empty())
            return -1;

        // 与原图像进行比例转换
        int img_width = 360;
        int img_height = 240;
        Mat image = _image.getMat();
        Size size = image.size();
        assert(image.type() == CV_8UC1);
        Mat img;
        image.convertTo(img, CV_32FC1, 1.f/255, 0);
        // 记录尺度转换比例
        float ratio_width = float(image.cols) / float(img_width);
        float ratio_height = float(image.rows) / float(img_height);
        cv::resize(img, img, cv::Size(img_width, img_height));

    // 1.图像传至PL
        
    // 2.从PL获取特征点和描述子的raw_data    

    /*      Test 等同步骤 1 2      */
    // 预处理输入
    torch::DeviceType device_type;
    device_type = torch::kCPU;
    torch::Device device(device_type);
    /*std::vector<int64_t> dims = {1, img_height, img_width, 1}; // 根据实际输入形状调整 
    auto img_var = torch::from_blob(img.data, dims, torch::kFloat32); // 将输入数据移动到 CUDA 设备 */
    auto img_var = torch::from_blob(img.data, {1, 1, img.rows, img.cols}, torch::kFloat32);
    img_var = img_var.to(torch::kCPU); // 调整输入的维度顺序（NCHW -> NHWC） 
    //img_var = img_var.permute({0, 3, 1, 2}); // 转置
    std::vector<torch::jit::IValue> inputs; 
    // 进行推理
    inputs.push_back(img_var); 
    auto output = module.forward(inputs).toTuple();
    // 保存结果
    auto keypoint_prob  = output->elements()[0].toTensor().to(torch::kCPU);
    auto desc_raw = output->elements()[1].toTensor().to(torch::kCPU);

    // 3.处理raw_keypoint 现在假设输出的是keypoints_test //[1 65 H/8 W/8] => [nkeypoints 2]
        keypoint_prob = torch::softmax(keypoint_prob,1); // softmax,[1 65 H/8 W/8]
        keypoint_prob = keypoint_prob.slice(1,0,64); // 65通道减少 [1, 64, H/8, W/8]
        keypoint_prob = keypoint_prob.permute({0, 2, 3, 1});  // [B, H/8, W/8, 64]

        // torch::reshape,sp里有实现 pixel_shuffle
        int Hc = keypoint_prob.size(1);
        int Wc = keypoint_prob.size(2);
        keypoint_prob = keypoint_prob.contiguous().view({-1, Hc, Wc, 8, 8});
        keypoint_prob = keypoint_prob.permute({0, 1, 3, 2, 4});
        keypoint_prob = keypoint_prob.contiguous().view({-1, Hc * 8, Wc * 8});  // [B, H, W]
        keypoint_prob = keypoint_prob.squeeze(); // [H W]

        float threshold = 0.007; // 阈值
        auto kpts = (keypoint_prob > threshold); // [H W]
        kpts = torch::nonzero(kpts);  // [nkeypoints 2]  (y, x)

    // 4.处理raw_desc 现在假设输出的是desc_test [1*256*H/8*W/8] => [n_keypoints 256]
        // 新上采样，等同interpolate
        auto fkpts = kpts.to(torch::kFloat);
        auto grid = torch::zeros({1, 1, kpts.size(0), 2}).to(device);  // [1 1 nkeypoints 2]
        grid[0][0].slice(1, 0, 1) = 2.0 * fkpts.slice(1, 1, 2) / keypoint_prob.size(1) - 1;  // x
        grid[0][0].slice(1, 1, 2) = 2.0 * fkpts.slice(1, 0, 1) / keypoint_prob.size(0) - 1;  // y
        desc_raw = torch::grid_sampler(desc_raw, grid, 0, 0, false);  // [1 256 1 nkeypoints]
        desc_raw = desc_raw.squeeze(0).squeeze(1);  // [256 nkeypoints]   

        // 新标准化 
        auto dn = torch::norm(desc_raw, 2, 1);
        desc_raw = desc_raw.div(torch::unsqueeze(dn, 1));
        desc_raw = desc_raw.transpose(0, 1).contiguous();  // [n_keypoints 256]
        desc_raw = desc_raw.to(torch::kCPU);

    // 5.nms处理 CV_位数 类型(F C) 通道数，如CV_32FC1 需要考虑W H的输出，所以首先进行置换
        // 5.1 创建keypoints_no_nms
        std::vector<cv::KeyPoint> keypoints_no_nms;
        for(int i = 0; i < kpts.size(0); i++){
            float response = keypoint_prob[kpts[i][0]][kpts[i][1]].item<float>();
            keypoints_no_nms.push_back(cv::KeyPoint(kpts[i][1].item<float>(),kpts[i][0].item<float>(), 8, -1, response));
        }
        // 5.2 创建desc_no_nms
        cv::Mat desc_no_nms(cv::Size(desc_raw.size(1),desc_raw.size(0)), CV_32FC1, desc_raw.data<float>());     

        // 5.3 非最大值抑制处理
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;  
        cv::Mat desc;
        cv::Mat conf(keypoints_no_nms.size(), 1, CV_32F); // prob
        CNN_nms(keypoints_no_nms, conf, desc_no_nms, keypoints, desc, 8, 4, img_height, img_width, ratio_width, ratio_height);

        // 5.4 保存最终结果
        int nkeypoints = keypoints.size();
        if(nkeypoints == 0)
            _descriptors.release();
        else{
            // 修改为8U正常
            _descriptors.create(nkeypoints, 256, CV_32F);           
            descriptors = _descriptors.getMat(); // 公用一块内存
        }

        _keypoints = vector<cv::KeyPoint>(nkeypoints);

        //Modified for speeding up stereo fisheye matching
        int monoIndex = 0, stereoIndex = nkeypoints-1;
        int i = 0;
        for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                        keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint){
            if(keypoint->pt.x >= vLappingArea[0] && keypoint->pt.x <= vLappingArea[1]){
                _keypoints.at(stereoIndex) = (*keypoint);
                desc.row(i).copyTo(descriptors.row(stereoIndex));
                stereoIndex--;
            }
            else{
                _keypoints.at(monoIndex) = (*keypoint);
                desc.row(i).copyTo(descriptors.row(monoIndex));
                monoIndex++;
            }
            i++;
        }

        return monoIndex;
    }
} // namespace CNN_SLAM3(OG:ORB_SLAM3)

