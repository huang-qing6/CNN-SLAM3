#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>

#include "NewExtractor.h"

#include <chrono>

using namespace cv;
using namespace std;
namespace F = torch::nn::functional;

namespace ORB_SLAM3{

    const int PATCH_SIZE = 31;
    const int HALF_PATCH_SIZE = 15;
    const int EDGE_THRESHOLD = 19;

    /**原superpoint的shuffle
     * Input:
     *  keypoint: 特征点,大小为[1, 64, img_height/8, img_width/8]
     * Output:
     *  处理后的keypoint [1, 1, img_height, img_width]
     */
    torch::Tensor CNN_shuffle(torch::Tensor &tensor){
        const int64_t scale_factor = 8;
        int64_t num, ch, height, width;
        num = tensor.size(0);
        ch = tensor.size(1);
        height = tensor.size(2);
        width = tensor.size(3);

        // assert ch % (scale_factor * scale_factor) == 1? 64/(8*8)确实等于1
        tensor = tensor.reshape({num, ch / (scale_factor * scale_factor), scale_factor, scale_factor, height, width});
        tensor = tensor.permute({0, 1, 4, 2, 5, 3});
        tensor = tensor.reshape({num, ch / (scale_factor * scale_factor), scale_factor * height, scale_factor * width});

        return tensor;
    }

    /**非极大值抑制,
     * Input:
     *  keypoint [W H]
     *  desc [256 W*H]
     *  keypoint_res 特征点结果
     *  descriptors 描述子结果
     */ 
    void CNN_nms(cv::Mat keypoint, cv::Mat desc, std::vector<cv::KeyPoint>& keypoint_res, cv::Mat& descriptors,
    int border, int dist_thresh, int img_width, int img_height, float ratio_width, float ratio_height){
        std::vector<cv::Point2f> pts_raw;
        cv::Size size = keypoint.size();
        int w, h;
        // pts_raw 转储 w*h 的所有点 二维拉成一维 现在假设pts_raw的w*h点坐标与desc中[256 w*h]对应
        for(int i = 0; i < size.width; i++){
            for(int j = 0; j < size.height; j++){
                w = i;
                h = j;
                pts_raw.push_back(cv::Point2f(h, w));
            }
        }

        // 设置宽/列 width，高/行 height
        cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
        cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);
        // 初始化 grid inds
        grid.setTo(0);
        inds.setTo(0);

        for (int i = 0; i < pts_raw.size(); i++)
        {   
            int w = (int) pts_raw[i].x;
            int h = (int) pts_raw[i].y;

            grid.at<char>(h, w) = 1;
            inds.at<unsigned short>(h, w) = i;
        }

        // 扩展grid 上下左右延展 dist_thresh 大小
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
                    // 还原至原图像大小
                    keypoint_res.push_back(cv::KeyPoint(pts_raw[select_ind].x * ratio_width, pts_raw[select_ind].y * ratio_height, 1.0f));

                    select_indice.push_back(select_ind);
                    valid_cnt++;
                }
            }
        }
        
        descriptors.create(select_indice.size(), 256, CV_8U);

        for (int i=0; i<select_indice.size(); i++)
        {
            for (int j=0; j<256; j++)
            {
                // 保存的特征点对应的描述子信息，两个向量位置一致
                descriptors.at<unsigned char>(i, j) = desc.at<unsigned char>(select_indice[i], j);
            }
        }       

    }

    CNNextractor::CNNextractor(int _nfeatures, float _scaleFactor, int _nlevels,
                               int _iniThFAST, int _minThFAST):
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

        /*const int npoints = 512;
        const Point* pattern0 = (const Point*)bit_pattern_31_;
        std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));*/

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
        /*const char *net_fn = getenv("CNN_PATH");
        net_fn = (net_fn == nullptr) ? "Netsuper_ptq.pt" : net_fn; 
        module = torch::jit::load(net_fn);*/
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

        // 转换输入模型的图像大小360*240
        int img_width = 360;
        int img_height = 240;
        Mat image = _image.getMat();
        assert(image.type() == CV_8UC1);
        Mat img;
        image.convertTo(img, CV_32FC1, 1.f/*/ 255.f*/, 0);
        // 记录尺度转换比例
        float ratio_width = float(img.cols) / float(img_width);
        float ratio_height = float(img.rows) / float(img_height);
        cv::resize(img, img, cv::Size(img_width, img_height));

    // test用，操作模型
    std::vector<int64_t> dims = {1, img_height, img_width, 1}; // 根据实际输入形状调整 
    auto img_var = torch::from_blob(img.data, dims, torch::kFloat32); // 将输入数据移动到 CUDA 设备 
    img_var = img_var.to(torch::kCPU); // 调整输入的维度顺序（NCHW -> NHWC） 
    img_var = img_var.permute({0, 3, 1, 2}); // 转置
    std::vector<torch::jit::IValue> inputs; 

    inputs.push_back(img_var); // 执行推理
    auto output = module.forward(inputs).toTuple(); // 导入模型，进行推理 fp32用
    // auto output = module.quantize_inference(inputs).toTuple();

    // 保存结果
    auto keypoint_raw  = output->elements()[0].toTensor().to(torch::kCPU);
    auto desc_raw = output->elements()[1].toTensor().to(torch::kCPU);

    // 1.图像传至PL
        
    // 2.从PL获取特征点和描述子的raw_data    

    // 3.处理raw_keypoint 现在假设输出的是keypoints_test //[1*65*H/8*W/8] => [1*1*H*W]
        keypoint_raw = torch::softmax(keypoint_raw,1); // softmax,[1, 65, H/8, W/8]
        // 还差一个65通道减少为64的操作 [1, 64, H/8, W/8]
        keypoint_raw = torch::slice(keypoint_raw,1,0,64);
        // torch::reshape,sp里有实现 pixel_shuffle
        keypoint_raw = CNN_shuffle(keypoint_raw); // [1, 1, H, W]

    // 4.处理raw_desc 现在假设输出的是desc_test [1*256*H/8*W/8] => [1*256*H*W]
        vector<double> grid_size {8, 8}; // 上采样高,宽 [1*256*H/8*W/8]
        desc_raw = F::interpolate(desc_raw, 
        F::InterpolateFuncOptions().mode(torch::kBilinear).align_corners(false).scale_factor(grid_size));
        desc_raw = F::normalize(desc_raw, F::NormalizeFuncOptions().p(2).dim(1));

    // 5.压缩维度 
        keypoint_raw = keypoint_raw.squeeze(); // [B, 1, H, W] >> [B * H * W] 保留0通道,B = 1
        desc_raw = desc_raw.squeeze(); // [1*256*H*W] >> [256 * H * W]

    // 6.nms处理 CV_位数 类型(F C) 通道数，如CV_32FC1 需要考虑W H的输出，所以首先进行置换
        // 6.1 整理keypoint输出 创建pts_mat作为nms输入
        keypoint_raw = keypoint_raw.permute({1, 0}); // [H W] >> [W H]
        cv::Mat pts_mat(cv::Size(keypoint_raw.size(0), keypoint_raw.size(1)), CV_32FC1, keypoint_raw.data<float>());
        cv::Size size = pts_mat.size();
        // cout << pts_mat.rows << " " << pts_mat.cols << " " << pts_mat.channels() << endl;

        // 6.2 整理desc输出 创建desc_mat作为desc_mat输入
        desc_raw = desc_raw.permute({0, 2, 1}); // [256 H W] >> [256 W H]
        desc_raw = desc_raw.reshape({256, -1}); // 将W H压缩为一个维度
        cv::Mat desc_mat(cv::Size(256, img_height * img_width), CV_32FC1, desc_raw.data<float>()); //CV_8UC1 CV_32FC1
        // cout << desc_mat.rows << " " << desc_mat.cols << " " << desc_mat.channels() << endl;

        // 6.3 非最大值抑制处理
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;    
        CNN_nms(pts_mat, desc_mat, keypoints, descriptors, 8, 4, img_height, img_width, ratio_width, ratio_height);

        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());      
        int nkeypoints = keypoints.size();
        // cout << nkeypoints << endl; 目前需要考虑再约束一下特征点个数？
        _descriptors.create(nkeypoints, 256, CV_32F); //CV_8U
        size = _descriptors.size();
        size = descriptors.size();
        descriptors.copyTo(_descriptors.getMat());
        _keypoints = vector<cv::KeyPoint>(nkeypoints);  
        cout << "Frame extrator finished!" << endl;

        /*int offset = 0;
        //Modified for speeding up stereo fisheye matching
        int monoIndex = 0, stereoIndex = nkeypoints-1;
        for (int level = 0; level < nlevels; ++level)
        {
            vector<KeyPoint>& keypoints = allKeypoints[level];
            int nkeypointsLevel = (int)keypoints.size();

            if(nkeypointsLevel==0)
                continue;

            // preprocess the resized image
            Mat workingMat = mvImagePyramid[level].clone();
            GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

            // Compute the descriptors
            //Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
            Mat desc = cv::Mat(nkeypointsLevel, 32, CV_8U);
            computeDescriptors(workingMat, keypoints, desc, pattern);

            offset += nkeypointsLevel;

            float scale = mvScaleFactor[level]; //getScale(level, firstLevel, scaleFactor);
            int i = 0;
            for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                         keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint){

                // Scale keypoint coordinates
                if (level != 0){
                    keypoint->pt *= scale;
                }

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
        }*/
        // 暂时没想好这里怎么改
        int monoIndex = 0;
        return monoIndex;
    }
} // namespace CNN_SLAM3(OG:ORB_SLAM3)

