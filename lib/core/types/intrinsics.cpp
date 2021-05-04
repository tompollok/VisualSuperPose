#include "intrinsics.h"
#include <opencv2/calib3d.hpp>

Intrinsics::Intrinsics()
{
    mImageSize = cv::Size(0,0);
    mFocalLength = cv::Point2d(0,0);
    mPrincipalPoint = cv::Point2d(0,0);
    mDistorionCoefficients = cv::Mat();
}

Intrinsics::Intrinsics(const cv::Size &imageSize, const cv::Point2d &focalLength, const cv::Point2d &principalPoint, const cv::Mat &distorionCoefficients)
{
    mImageSize = imageSize;
    mFocalLength = focalLength;
    mPrincipalPoint = principalPoint;
    mDistorionCoefficients = distorionCoefficients;
}

Intrinsics::Intrinsics(const Intrinsics &intrinsics)
{
    mImageSize = intrinsics.mImageSize;
    mFocalLength = intrinsics.mFocalLength;
    mPrincipalPoint = intrinsics.mPrincipalPoint;
    mDistorionCoefficients = intrinsics.mDistorionCoefficients;
}

Intrinsics &Intrinsics::operator=(const Intrinsics &intrinsics)
{
    if (this != &intrinsics) {
        mImageSize = intrinsics.mImageSize;
        mFocalLength = intrinsics.mFocalLength;
        mPrincipalPoint = intrinsics.mPrincipalPoint;
        mDistorionCoefficients = intrinsics.mDistorionCoefficients;
    }
    return *this;
}

std::vector<cv::Point2d> Intrinsics::undistortPoints(const std::vector<cv::Point2d> &points) const
{
    std::vector<cv::Point2d> dest;
    cv::undistortPoints(points, dest, getK3x3(), distorionCoefficients(), cv::noArray(), getK3x3());
    return dest;
}

std::vector<cv::Point2d> Intrinsics::distortPoints(const std::vector<cv::Point2d> &pointsUndistorted) const
{
    cv::Mat mapX, mapY;
    cv::initUndistortRectifyMap(getK3x3(), distorionCoefficients(), cv::noArray(), getK3x3(),
                                imageSize(), CV_32FC1, mapX, mapY);
    std::vector<cv::Point2d> out(pointsUndistorted.size());
    for (size_t i = 0; i < pointsUndistorted.size(); ++i)
    {
        int x = static_cast<int>(pointsUndistorted[i].x);
        int y = static_cast<int>(pointsUndistorted[i].y);
        out[i].x = mapX.at<float>(y, x);
        out[i].y = mapY.at<float>(y, x);
    }
    return out;
}

cv::Size Intrinsics::imageSize() const
{
    return mImageSize;
}

int Intrinsics::width() const
{
    return mImageSize.width;
}

int Intrinsics::height() const
{
    return mImageSize.height;
}

void Intrinsics::setImageSize(const cv::Size &imageSize)
{
    mImageSize = imageSize;
}

void Intrinsics::setImageSize(const int &width, const int &height)
{
    mImageSize.width = width;
    mImageSize.height = height;
}

void Intrinsics::setWidth(const int &width)
{
    mImageSize.width = width;
}

void Intrinsics::setHeight(const int &height)
{
    mImageSize.height = height;
}

cv::Point2d Intrinsics::focalLength() const
{
    return mFocalLength;
}

double Intrinsics::Fx() const
{
    return mFocalLength.x;
}

double Intrinsics::Fy() const
{
    return mFocalLength.y;
}

void Intrinsics::setFocalLength(const cv::Point2d &focalLength)
{
    mFocalLength = focalLength;
}

void Intrinsics::setFocalLength(const double &Fx, const double &Fy)
{
    mFocalLength.x = Fx;
    mFocalLength.y = Fy;
}

void Intrinsics::setFx(const double &Fx)
{
    mFocalLength.x = Fx;
}

void Intrinsics::setFy(const double &Fy)
{
    mFocalLength.y = Fy;
}

cv::Point2d Intrinsics::principalPoint() const
{
    return mPrincipalPoint;
}

double Intrinsics::Cx() const
{
    return mPrincipalPoint.x;
}

double Intrinsics::Cy() const
{
    return mPrincipalPoint.y;
}

void Intrinsics::setPrincipalPoint(const cv::Point2d &principalPoint)
{
    mPrincipalPoint = principalPoint;
}

void Intrinsics::setPrincipalPnt(const double &Cx, const double &Cy)
{
    mPrincipalPoint.x = Cx;
    mPrincipalPoint.y = Cy;
}

void Intrinsics::setCx(const double &Cx)
{
    mPrincipalPoint.x = Cx;
}

void Intrinsics::setCy(const double &Cy)
{
    mPrincipalPoint.y = Cy;
}

cv::Matx33d Intrinsics::getK3x3() const
{
    return cv::Matx33d(mFocalLength.x, 0,              mPrincipalPoint.x,
                       0,              mFocalLength.y, mPrincipalPoint.y,
                       0,              0,              1);
}

std::vector<double> Intrinsics::distorionCoefficients() const
{
    return mDistorionCoefficients;
}

void Intrinsics::setDistorionCoefficients(const std::vector<double> &distorionCoefficients)
{
    if (distorionCoefficients.size() != 8) {
        std::cout << "Warning: distortion coefficients had wrong format! Input ignored";
        return;
    }
    mDistorionCoefficients = distorionCoefficients;
}

