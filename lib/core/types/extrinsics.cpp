#include "extrinsics.h"

Extrinsics::Extrinsics()
{

}

Extrinsics::Extrinsics(
        const cv::Matx44d &RTMat,
        const Extrinsics::TransformationDirection &direction)
{
    setRTMatrix(RTMat, direction);
}

Extrinsics::Extrinsics(
        const cv::Matx33d &rotation,
        const cv::Vec3d &translation,
        const Extrinsics::TransformationDirection &direction)
{
    setRTMatrix(rotation, translation, direction);
}

Extrinsics::Extrinsics(
        const cv::Vec3d &rodrigues,
        const cv::Vec3d &translation,
        const Extrinsics::TransformationDirection &direction)
{
    setRTMatrix(rodrigues, translation, direction);
}

Extrinsics::Extrinsics(
        const cv::Vec4d &quaternion,
        const cv::Vec3d &translation,
        const Extrinsics::TransformationDirection &direction)
{
    setRTMatrix(quaternion, translation, direction);
}

Extrinsics::Extrinsics(const Extrinsics &extrinsics)
    : mDirection(extrinsics.mDirection),
      mRotation(extrinsics.mRotation),
      mTranslation(extrinsics.mTranslation) {}

Extrinsics::Extrinsics(Extrinsics &&extrinsics)
    : mDirection(extrinsics.mDirection),
      mRotation(extrinsics.mRotation),
      mTranslation(extrinsics.mTranslation)
{
    extrinsics.mTranslation = cv::Vec3d(0,0,0);
    extrinsics.mRotation = cv::Matx33d::eye();
}

Extrinsics &Extrinsics::operator=(const Extrinsics &extrinsics)
{
    mRotation = extrinsics.mRotation;
    mTranslation = extrinsics.mTranslation;
    mDirection = extrinsics.mDirection;
    return *this;
}

Extrinsics &Extrinsics::operator=(Extrinsics &&extrinsics)
{
    if(this != &extrinsics) {
        mTranslation = extrinsics.mTranslation;
        extrinsics.mTranslation = cv::Vec3d(0,0,0);

        mRotation = extrinsics.mRotation;
        extrinsics.mRotation = cv::Matx33d::eye();

        mDirection = extrinsics.mDirection;
        extrinsics.mDirection = Ref2Local;
    }
    return *this;
}

bool Extrinsics::operator==(const Extrinsics &extrinsics) const
{
    return (mRotation == extrinsics.mRotation &&
            mTranslation == extrinsics.mTranslation &&
            mDirection == extrinsics.mDirection);
}

bool Extrinsics::operator!=(const Extrinsics &extrinsics) const
{
    return !(*this == extrinsics);
}

cv::Matx33d Extrinsics::getRotationMatrix(const Extrinsics::TransformationDirection &direction) const
{
    return mDirection == direction ? mRotation : mRotation.t();
}

cv::Vec4d Extrinsics::getRotationQuaternion(const Extrinsics::TransformationDirection &direction) const
{
    return rotationToQuaternion(mDirection == direction ? mRotation : mRotation.t());
}

cv::Vec3d Extrinsics::getRotationRodrigues(const Extrinsics::TransformationDirection &direction) const
{
    cv::Vec3d rodrigues;
    cv::Rodrigues(mDirection == direction ? mRotation : mRotation.t(), rodrigues);
    return rodrigues;
}

void Extrinsics::setRotation(
        const cv::Matx33d &rotation,
        const Extrinsics::TransformationDirection &direction)
{
    if (direction == mDirection) {
        mRotation = rotation;
    } else {
        // 1. invert
        //   r' = r.t()
        //   t' = -r' * t
        cv::Vec3d tInv = -mRotation.t() * mTranslation;
        // 2. set
        //   r' = rotation
        mRotation = rotation;
        // 3. invert again
        //   r'' = r'.t() = rotation.t()
        //   t'' = -r'' * t'
        mRotation = mRotation.t();
        mTranslation = -mRotation * tInv;
    }
}

void Extrinsics::setRotation(const cv::Vec3d &rodrigues,
        const Extrinsics::TransformationDirection &direction)
{
    cv::Matx33d rotation;
    cv::Rodrigues(rodrigues, rotation);
    setRotation(rotation, direction);
}

void Extrinsics::setRotation(const cv::Vec4d &quaternion, const Extrinsics::TransformationDirection &direction)
{
    setRotation(quaternionToRotation(quaternion), direction);
}

cv::Vec3d Extrinsics::getTranslation(const TransformationDirection &direction) const
{
    if (mDirection != direction)
    {
        return -mRotation.t() * mTranslation;
    }
    return mTranslation;
}

void Extrinsics::setTranslationVec(
        const cv::Vec3d &translation,
        const Extrinsics::TransformationDirection &direction)
{
    if (direction == mDirection) {
        mTranslation = translation;
    } else {
        mTranslation = -mRotation * translation;
    }
}

void Extrinsics::setTranslationVec(
        const double &x, const double &y, const double &z,
        const Extrinsics::TransformationDirection &direction)
{
    cv::Vec3d translation(x,y,z);
    setTranslationVec(translation,direction);
}

cv::Matx44d Extrinsics::getRTMatrix(
        const Extrinsics::TransformationDirection &direction) const
{
    if (mDirection != direction)
    {
        return inverseRT();
    }
    cv::Matx44d RTMat;
    composeRTMatrix(mRotation, mTranslation, RTMat);
    return RTMat;
}

void Extrinsics::setRTMatrix(
        const cv::Matx44d &RTMat,
        const Extrinsics::TransformationDirection &direction)
{
    if (direction == mDirection) {
        decomposeRTMatrix(RTMat, mRotation, mTranslation);
    } else {
        decomposeRTMatrix(RTMat.inv(), mRotation, mTranslation);
    }
    mDirection = direction;
}

void Extrinsics::setRTMatrix(
        const cv::Matx33d &rotation,
        const cv::Vec3d &translation,
        const TransformationDirection &direction)
{
    mDirection = direction;
    mRotation = rotation;
    mTranslation = translation;
}

void Extrinsics::setRTMatrix(
        const cv::Vec3d &rodrigues,
        const cv::Vec3d &translation,
        const Extrinsics::TransformationDirection &direction)
{
    cv::Matx33d rotation;
    cv::Rodrigues(rodrigues, rotation);
    setRTMatrix(rotation, translation, direction);
}

void Extrinsics::setRTMatrix(
        const cv::Vec4d &quaternion,
        const cv::Vec3d &translation,
        const Extrinsics::TransformationDirection &direction)
{
    setRTMatrix(quaternionToRotation(quaternion), translation, direction);
}

void Extrinsics::decomposeRTMatrix(
        const cv::Matx44d &RTMat,
        cv::Matx33d &rotation,
        cv::Vec3d &translation)
{
    rotation(0,0) = RTMat(0,0); rotation(0,1) = RTMat(0,1); rotation(0,2) = RTMat(0,2);
    rotation(1,0) = RTMat(1,0); rotation(1,1) = RTMat(1,1); rotation(1,2) = RTMat(1,2);
    rotation(2,0) = RTMat(2,0); rotation(2,1) = RTMat(2,1); rotation(2,2) = RTMat(2,2);

    translation(0) = RTMat(0,3);
    translation(1) = RTMat(1,3);
    translation(2) = RTMat(2,3);
}

void Extrinsics::composeRTMatrix(const cv::Matx33d &rotation, const cv::Vec3d &translation, cv::Matx44d &RTMat)
{
    RTMat(0,0) = rotation(0,0); RTMat(0,1) = rotation(0,1); RTMat(0,2) = rotation(0,2); RTMat(0,3) = translation(0);
    RTMat(1,0) = rotation(1,0); RTMat(1,1) = rotation(1,1); RTMat(1,2) = rotation(1,2); RTMat(1,3) = translation(1);
    RTMat(2,0) = rotation(2,0); RTMat(2,1) = rotation(2,1); RTMat(2,2) = rotation(2,2); RTMat(2,3) = translation(2);
    RTMat(3,0) = 0;             RTMat(3,1) = 0;             RTMat(3,2) = 0;             RTMat(3,3) = 1;
}

cv::Matx33d Extrinsics::quaternionToRotation(const cv::Vec4d &quaternion)
{
    //https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    cv::Matx33d rotation;
    double qw = quaternion(0);
    double qx = quaternion(1);
    double qy = quaternion(2);
    double qz = quaternion(3);

    rotation(0,0) = 1 - 2*qy*qy - 2*qz*qz;
    rotation(0,1) = 2*qx*qy - 2*qz*qw;
    rotation(0,2) = 2*qx*qz + 2*qy*qw;
    rotation(1,0) = 2*qx*qy + 2*qz*qw;
    rotation(1,1) = 1 - 2*qx*qx - 2*qz*qz;
    rotation(1,2) = 2*qy*qz - 2*qx*qw;
    rotation(2,0) = 2*qx*qz - 2*qy*qw;
    rotation(2,1) = 2*qy*qz + 2*qx*qw;
    rotation(2,2) = 1 - 2*qx*qx - 2*qy*qy;

    return rotation;
}

cv::Vec3d Extrinsics::quaternionToRodrigues(const cv::Vec4d &quaternion)
{
    cv::Vec3d out;
    cv::Rodrigues(quaternionToRotation(quaternion), out);
    return out;
}

cv::Vec4d Extrinsics::rotationToQuaternion(const cv::Matx33d &rotation)
{
    cv::Vec3d rodrigues;
    cv::Rodrigues(rotation, rodrigues);
    return rodriguesToQuaternion(rodrigues);
}

cv::Vec4d Extrinsics::rodriguesToQuaternion(const cv::Vec3d rodrigues)
{
    double angle = norm(rodrigues);
    cv::Vec3d axis(rodrigues(0) / angle, rodrigues(1) / angle, rodrigues(2) / angle);
    double angle_2 = angle / 2;
    //qw, qx, qy, qz
    cv::Vec4d quaternion(cos(angle_2), axis(0) * sin(angle_2), axis(1) * sin(angle_2),
                         axis(2) * sin(angle_2));
    return quaternion;

}

cv::Matx44d Extrinsics::inverseRT() const
{
    cv::Mat m = cv::Mat::eye(4,4,CV_64F);
    cv::Matx33d rTransposed = mRotation.t();
    cv::Mat(rTransposed).copyTo(m.rowRange(0, 3).colRange(0, 3));
    cv::Mat(- rTransposed * mTranslation).copyTo(m.rowRange(0, 3). colRange(3, 4));
    return m;
}
