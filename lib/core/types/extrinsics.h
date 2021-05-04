#ifndef EXTRINSICS_H
#define EXTRINSICS_H

#include "ppbafloc-core_export.h"
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
/**
 * @brief The Extrinsics class represents the camera transformation
 * in homogenous coordinates
 */
class PPBAFLOC_CORE_EXPORT Extrinsics
{
public:
    enum TransformationDirection {
        Ref2Local = 0, //Reference Coordinate System to Camera Coordinate System
        Local2Ref = 1  //Camera Coordinate System to Reference Coordinate System
    };

    /**
     * @brief Default constructor, everything will be zero.
     */
    Extrinsics();
    /**
     * @brief Constructor, use one of these.
     */
    Extrinsics(
            const cv::Matx44d &RTMat,
            const TransformationDirection &direction);
    Extrinsics(
            const cv::Matx33d &rotation,
            const cv::Vec3d &translation,
            const TransformationDirection &direction);
    Extrinsics(
            const cv::Vec3d &rodrigues,
            const cv::Vec3d &translation,
            const TransformationDirection &direction);
    Extrinsics(
            const cv::Vec4d &quaternion,
            const cv::Vec3d &translation,
            const TransformationDirection &direction);

    /**
     @brief Copy constructor
     */
    Extrinsics(const Extrinsics& extrinsics);

    /**
     @brief Move constructor.
     */
    Extrinsics(Extrinsics&& extrinsics);

    /**
     @brief Copy operator.
     */
    Extrinsics& operator= (const Extrinsics& extrinsics);

    /**
     @brief Move operator.
     */
    Extrinsics& operator= (Extrinsics&& extrinsics);

    /**
     @brief Comparison operator.
     */
    bool operator==(const Extrinsics& extrinsics) const;
    bool operator!=(const Extrinsics& extrinsics) const;

    /**
     * @brief Rotation getter & setter
     */
    cv::Matx33d getRotationMatrix(
            const TransformationDirection &direction) const;
    cv::Vec4d getRotationQuaternion(
            const TransformationDirection &direction) const;
    cv::Vec3d getRotationRodrigues(
            const TransformationDirection &direction) const;
    void setRotation(
            const cv::Matx33d &rotation,
            const TransformationDirection &direction);
    void setRotation(
            const cv::Vec3d &rodrigues,
            const TransformationDirection &direction);
    void setRotation(
            const cv::Vec4d &quaternion,
            const TransformationDirection &direction);

    /**
     * @brief Translation getter & setter
     */
    cv::Vec3d getTranslation(
            const TransformationDirection &direction) const;

    void setTranslationVec(
            const cv::Vec3d& translation,
            const TransformationDirection &direction);
    void setTranslationVec(
            const double& x, const double& y, const double& z,
            const TransformationDirection &direction);

    /**
     * @brief Transformation getter & setter
     */
    cv::Matx44d getRTMatrix(
            const TransformationDirection &direction) const;
    void setRTMatrix(
            const cv::Matx44d &RTMat,
            const TransformationDirection &direction);
    void setRTMatrix(
            const cv::Matx33d &rotation,
            const cv::Vec3d &translation,
            const TransformationDirection &direction);
    void setRTMatrix(
            const cv::Vec3d &rodrigues,
            const cv::Vec3d &translation,
            const TransformationDirection &direction);
    void setRTMatrix(
            const cv::Vec4d &quaternion,
            const cv::Vec3d &translation,
            const TransformationDirection &direction);

    /**
     * @brief Transformation to Translation & Rotation
     */
    static void decomposeRTMatrix(
            const cv::Matx44d &RTMat,
            cv::Matx33d &rotation,
            cv::Vec3d &translation);
    /**
     * @brief Rotation & Translation to Transformation
     */
    static void composeRTMatrix(
            const cv::Matx33d &rotation,
            const cv::Vec3d &translation,
            cv::Matx44d &RTMat);
    //some conversions
    static cv::Matx33d quaternionToRotation(
            const cv::Vec4d &quaternion);
    static cv::Vec3d quaternionToRodrigues(
            const cv::Vec4d &quaternion);
    static cv::Vec4d rotationToQuaternion(
            const cv::Matx33d &rotation);
    static cv::Vec4d rodriguesToQuaternion(
            const cv::Vec3d rodrigues);

private:
    cv::Matx44d inverseRT() const;

private:
    TransformationDirection mDirection = Ref2Local;
    cv::Matx33d mRotation = cv::Matx33d::eye(); //always in Ref2Local
    cv::Vec3d mTranslation = cv::Vec3d(0,0,0); //always in Ref2Local
};

#endif // EXTRINSICS_H
