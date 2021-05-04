
#include "SiftHelpers.h"
#include "iohelpers.h"
#include <QDirIterator>
#include <iostream>
#if CV_MINOR_VERSION >= 5 && CV_MAJOR_VERSION == 4
#include <opencv2/features2d.hpp>
#define Sift cv::SIFT
#else
#include <opencv2/xfeatures2d.hpp>
#define Sift cv::xfeatures2d::SIFT
#endif
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

void SiftHelpers::extractSiftFeaturesImgList(const std::vector<std::string> &imageNames, std::vector<cv::Mat> &descriptors,
                                      int maxImages) {

    cv::Mat featOneImg;
    std::vector<cv::KeyPoint> keypoints;
    for (const auto &img: imageNames) {
        extractSiftFeatures(img, featOneImg, keypoints);
        descriptors.push_back(featOneImg);
    }
}

int SiftHelpers::extractSiftFeatures(const std::string &imagename, cv::Mat &descriptor, std::vector<cv::KeyPoint> &keypoints) {
    if(!IOHelpers::existsFile(QString::fromStdString(imagename))) {
        std::cout << "Image file " << imagename << " does not exist." << std::endl;
        return -1;
    }
    cv::Mat input = cv::imread(imagename, cv::IMREAD_GRAYSCALE);
    std::shared_ptr<Sift> sift_detector = Sift::create();
    keypoints.clear();
    sift_detector->detect(input, keypoints);
    sift_detector->compute(input, keypoints, descriptor);

    input = cv::Mat();
    return 0;
}

int SiftHelpers::extractSiftFeatures(const cv::Mat &img, cv::Mat &descriptor, std::vector<cv::KeyPoint> &keypoints)
{
    std::shared_ptr<Sift> sift_detector = Sift::create();

    keypoints.clear();
    sift_detector->detect(img, keypoints);
    sift_detector->compute(img, keypoints, descriptor);
    return 0;
}

void SiftHelpers::extractSiftFeaturesDir(const std::string &dirname, std::vector<cv::Mat> &features, int maxImages) {
    //making sure only images are processed
    QStringList filter;
    filter << "*.jpg" << "*.png" << "*.jpeg";
    QDirIterator it(QString::fromStdString(dirname), filter, QDir::Files, QDirIterator::Subdirectories);
    features.clear();
    cv::Mat featOneImg;
    std::vector<cv::KeyPoint> keypoints;

    int i = 0;
    while(it.hasNext()) {
        std::string path = it.next().toStdString();

        extractSiftFeatures(path, featOneImg, keypoints);
        features.push_back(featOneImg);

        std::cout << i++ << ":" << featOneImg.rows << ":" << path << std::endl;
        if (i >= maxImages && maxImages > 0)
            break;
    }
}

void SiftHelpers::siftMatching(const cv::Mat &descriptor1, const cv::Mat &descriptor2, std::vector<cv::DMatch> &goodMatches) {
    std::shared_ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(descriptor1, descriptor2, knnMatches, 2);

    const float ratio_thresh = 0.7f;
    for (auto & knnMatche : knnMatches) {
        if (knnMatche[0].distance < ratio_thresh * knnMatche[1].distance) {
            goodMatches.push_back(knnMatche[0]);
        }
    }
}
