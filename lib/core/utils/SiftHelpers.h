#pragma once

#include "ppbafloc-core_export.h"
#include <string>
#include <opencv2/core.hpp>
#include <QString>

/**
 * @brief The SiftHelpers class: Some usefull functions when using SIFT.
 */
class PPBAFLOC_CORE_EXPORT SiftHelpers {
    public:
    //For exctracting Sift Features & Descriptors
    static int extractSiftFeatures(const std::string &imagename, cv::Mat &descriptor, std::vector<cv::KeyPoint> &keypoints);
    static int extractSiftFeatures(const cv::Mat& img, cv::Mat &descriptor, std::vector<cv::KeyPoint> &keypoints);
    static void extractSiftFeaturesDir( const std::string &dirname, std::vector<cv::Mat> &features, int maxImages);
    static void extractSiftFeaturesImgList(const std::vector<std::string> &imageNames, std::vector<cv::Mat> &descriptors, int maxImages);
    //Depricated, not used
    static void siftMatching(const cv::Mat &descriptor1, const cv::Mat &descriptor2, std::vector<cv::DMatch> &goodMatches);
};

