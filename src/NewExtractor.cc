#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>

#include "NewExtractor.h"

using namespace cv;
using namespace std;

namespace ORB_SLAM3{

    const int PATCH_SIZE = 31;
    const int HALF_PATCH_SIZE = 15;
    const int EDGE_THRESHOLD = 19;


//  非极大值抑制
    void CNN_nms(cv::Mat det, cv::Mat desc, std::vector<cv::KeyPoint>& pts, cv::Mat& descriptors,
        int border, int dist_thresh, int img_width, int img_height, float ratio_width, float ratio_height){
        /* 暂时不知道是写个啥子，对标GCN_SLAMv2的nms函数 */
        std::vector<cv::Point2f> pts_raw;

        for (int i = 0; i < det.rows; i++){

            int u = (int) det.at<float>(i, 0);
            int v = (int) det.at<float>(i, 1);

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
                    pts.push_back(cv::KeyPoint(pts_raw[select_ind].x * ratio_width, pts_raw[select_ind].y * ratio_height, 1.0f));

                    select_indice.push_back(select_ind);
                    valid_cnt++;
                }
            }
        }
        
        descriptors.create(select_indice.size(), 32, CV_8U);

        for (int i=0; i<select_indice.size(); i++)
        {
            for (int j=0; j<32; j++)
            {
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

        // 初始化CNN
        const char *net_fn = getenv("CNN_PATH");
        net_fn = (net_fn == nullptr) ? "cnn.onnx" : net_fn; 
        module = /*make_shared<torch::jit::Module>*/torch::jit::load(net_fn);
    }

    int CNNextractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                                  OutputArray _descriptors, std::vector<int> &vLappingArea)
    {
        
        torch::DeviceType device_type;
        device_type = torch::kFPGA; // 设备设置和kCUDA不同,不知道修改过后会发生什么
        torch::Device device(device_type);

        if(_image.empty())
            return -1;

        Mat image = _image.getMat();
        assert(image.type() == CV_8UC1 );

        Mat img;
        // 为什么要转大小？
        image.convertTo(img, CV_32FC1, 1.f / 255.f , 0);
        // 大小360*240， border和dist设置不确定
        int img_width = 360;
        int img_height = 240;
        int border = 8;
        int dist_thresh = 4;        

        float ratio_width = float(img.cols) / float(img_width);
        float ratio_height = float(img.rows) / float(img_height);
        
        cv::resize(img, img, cv::Size(img_width, img_height));
        std::vector<int64_t> dims = {1, img_height, img_width, 1};
        // 新接口写法
        auto img_var = torch::from_blob(img.data, dims, torch::kFloat32).to(device);
        img_var = img_var.permute({0,3,1,2});

/**  整体构思： 
 *  1.传入img开始提取特征点
 *  2.axi总线通知pl端传入ddr的数据
 *  3.axi总线收到信号，接收数据
 *  4.将数据传入ps端部署的cnn后半部分
 */
        /*** axi ddr part ***/

        /*** end ***/

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(img_var);
        // auto output = module->forward(inputs).toTuple();
        auto output = module.forward(inputs).toTuple();

        auto pts  = output->elements()[0].toTensor().to(torch::kCPU).squeeze();
        auto desc = output->elements()[1].toTensor().to(torch::kCPU).squeeze();

        cv::Mat pts_mat(cv::Size(3, pts.size(0)), CV_32FC1, pts.data<float>());
        cv::Mat desc_mat(cv::Size(32, pts.size(0)), CV_8UC1, desc.data<unsigned char>());

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        // nms(pts_mat, desc_mat, keypoints, descriptors, border, dist_thresh, img_width, img_height, ratio_width, ratio_height);
        CNN_nms(pts_mat, desc_mat, keypoints, descriptors, border, dist_thresh, img_width, img_height, ratio_width, ratio_height);    

        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
        
        int nkeypoints = keypoints.size();
        _descriptors.create(nkeypoints, 32, CV_8U);
        descriptors.copyTo(_descriptors.getMat());

        
        _keypoints = vector<cv::KeyPoint>(nkeypoints);
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