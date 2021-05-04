 #ifndef INTRINSICS_H
#define INTRINSICS_H

#include "ppbafloc-core_export.h"
#include <opencv2/core.hpp>
#include <iostream>

class PPBAFLOC_CORE_EXPORT Intrinsics
{
public:
    /**
     * @brief Default constructor, everything will be zero.
     */
    Intrinsics();
    /**
     * @brief Please use this Constructor
     */
    Intrinsics(const cv::Size &imageSize,
               const cv::Point2d &focalLength,
               const cv::Point2d &principalPoint,
               const cv::Mat &distorionCoefficients);
    /**
     * @brief Copy constructur
     */
    Intrinsics(const Intrinsics &intrinsics);
    /**
     * @brief Copy operator.
     */
    Intrinsics &operator= (const Intrinsics &intrinsics);

    /**
     * @brief points without distortion
     */
    std::vector<cv::Point2d> undistortPoints(const std::vector<cv::Point2d> &points) const;

    /**
     * @brief re-apply distortion to points
     */
    std::vector<cv::Point2d> distortPoints(const std::vector<cv::Point2d> &pointsUndistorted) const;

    /**
     * @brief imageSize: (width, height)
     */
    cv::Size imageSize() const;
    int width() const;
    int height() const;
    void setImageSize(const cv::Size &imageSize);
    void setImageSize(const int &width, const int &height);
    void setWidth(const int &width);
    void setHeight(const int &height);

    /**
     * @brief focalLength: (fx, fy)
     */
    cv::Point2d focalLength() const;
    double Fx() const;
    double Fy() const;
    void setFocalLength(const cv::Point2d &focalLength);
    void setFocalLength(const double &Fx, const double &Fy);
    void setFx(const double &Fx);
    void setFy(const double &Fy);

    /**
     * @brief principalPoint: (cx, cy)
     */
    cv::Point2d principalPoint() const;
    double Cx() const;
    double Cy() const;
    void setPrincipalPoint(const cv::Point2d &principalPoint);
    void setPrincipalPnt(const double &Cx, const double &Cy);
    void setCx(const double &Cx);
    void setCy(const double &Cy);

    /**
     * @brief Kameramatrix
     *  (fx 0  cx)
     *  (0  fy cy)
     *  (0  0  1 )
     */
    cv::Matx33d getK3x3() const;

    /**
     * @brief distorionCoefficients 1x8 Matrix (k1, k2, p1, p2, k3, k4, k5, k6)
     */
    std::vector<double> distorionCoefficients() const;
    void setDistorionCoefficients(const std::vector<double> &distorionCoefficients);

private:
    cv::Size mImageSize; //width, height
    cv::Point2d mFocalLength; //fx, fy
    cv::Point2d mPrincipalPoint; // cx, cy
    std::vector<double> mDistorionCoefficients; // k1, k2, p1, p2, k3, k4, k5, k6
};

#endif // INTRINSICS_H
