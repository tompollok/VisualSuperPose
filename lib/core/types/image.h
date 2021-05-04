#ifndef IMAGE_H
#define IMAGE_H

#include <memory>

#include <opencv2/opencv.hpp>

#include <QByteArray>

#include "extrinsics.h"
#include "intrinsics.h"

struct PPBAFLOC_CORE_EXPORT BoW
{
    virtual QByteArray toByteArray() const { return QByteArray(); }
    virtual void fromByteArray(QByteArray data) {  }
};
/**
 * @brief The CSVRow struct for evaluation. Stores the evaluation results of image retrieval and registration.
 */
struct CSVRow
{
    std::string queryName = "";
    int gallerySize = -1;

    /// for registration
    double distanceX = std::numeric_limits<double>::max();
    double distanceY = std::numeric_limits<double>::max();
    double distanceZ = std::numeric_limits<double>::max();
    double distance = std::numeric_limits<double>::max();
    double roll = std::numeric_limits<double>::max();
    double pitch = std::numeric_limits<double>::max();
    double yaw = std::numeric_limits<double>::max();
    double angle = std::numeric_limits<double>::max();

    /// for retrieval
    int maxRightResults = -1;
    double AP10 = std::numeric_limits<double>::max();
    int rightAP10 = -1;
    double AP25 = std::numeric_limits<double>::max();
    int rightAP25 = -1;
    double AP100 = std::numeric_limits<double>::max();
    int rightAP100 = -1;
    std::vector<std::pair<std::string, bool>> retrievelImages;
};
//for cornell evaluation: 2D-3D correspondences
struct ColMapImagePoint;
struct PPBAFLOC_CORE_EXPORT ColMapWorldPoint {
    cv::Point3f pos;
    std::vector<std::shared_ptr<ColMapImagePoint>> imagepoints;
};

struct Image;
struct PPBAFLOC_CORE_EXPORT ColMapImagePoint {
    cv::Point2f pos;
    std::shared_ptr<Image> image;
    std::shared_ptr<ColMapWorldPoint> worldpoint;
};
//for cornell evaluation END

/**
 * @brief The Image struct represents every query and galleryimage that is used by this program.
 */
struct PPBAFLOC_CORE_EXPORT Image
{
    int id = -1; //DB primary key for the image
    std::string path; //of the image
    Extrinsics extrinsics; //position & rotation
    Intrinsics intrinsics; //camera parameters
    std::vector<cv::KeyPoint> siftKeypoints; //!TEMP! keypoints of image. !USE FORGET WHEN YOU DON'T NEED THEM ANYMORE!
    cv::Mat siftDescriptors; //!TEMP! descriptors of image. !USE FORGET WHEN YOU DON'T NEED THEM ANYMORE!
    std::shared_ptr<BoW> bow = nullptr; //!TEMP! fbow-map of image. !USE FORGET WHEN YOU DON'T NEED THEM ANYMORE!
    cv::Mat grayscaleImage; //!TEMP! the image in grayscale. !USE FORGET WHEN YOU DON'T NEED THEM ANYMORE!
    cv::Mat colorImage; //!TEMP! the image in rgb. !USE FORGET WHEN YOU DON'T NEED THEM ANYMORE!
    std::shared_ptr<CSVRow> csvrow = nullptr; //For Queryimages while evaluating

    void loadImageGrayscale() { grayscaleImage = cv::imread(path,cv::IMREAD_GRAYSCALE); }
    void loadImageColor() { colorImage = cv::imread(path,cv::IMREAD_COLOR); }
    //ALWAYS REMEMBER to forget those after usage!
    void forgetColorImage() { colorImage = cv::Mat(); }
    void forgetGrayscaleImage() { grayscaleImage = cv::Mat(); }
    void forgetImages();
    void forgetSiftKeypoints() {siftKeypoints.clear();}
    void forgetSiftDescriptors() { siftDescriptors = cv::Mat();}
    void forgetFbow() {bow = nullptr;}
    void forgetAll();

    //2D-3D Correspondences for evaluation
    std::vector<std::shared_ptr<ColMapImagePoint>> imagepoints;

};

//Checks if there are more than threshold correnspondences between image.imagepoints (for evaluation).
inline bool PPBAFLOC_CORE_EXPORT IsGoodReferenceFrame(
        std::shared_ptr<Image> imageA,
        std::shared_ptr<Image> imageB,
        int threshold) {
    int count = 0;
    for (auto &pntA : imageA->imagepoints) {
        for (auto &pnt2d : pntA->worldpoint->imagepoints) {
            if (pnt2d->image == imageB) {
                count++;
                break;
            }
        }
        if (count >= threshold)
            return true;
    }
    return false;
}




#endif // IMAGE_H
