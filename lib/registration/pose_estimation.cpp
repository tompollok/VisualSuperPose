#include "pose_estimation.h"

#include <math.h>

#include <cassert>
#include <cmath>
#include <iostream>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define EPSILON 1.0e-8

cv::Vec3d calculateRPY(const cv::Mat& rot) {
  double pitch_rad = asin(-rot.at<float>(2, 0));
  double yaw_rad = std::numeric_limits<double>::max(), roll_rad = yaw_rad;
  if (cos(pitch_rad) > EPSILON) {
    yaw_rad = asin(rot.at<float>(1, 0) / cos(pitch_rad));
    roll_rad = asin(rot.at<float>(2, 1) / cos(pitch_rad));
  } else {
    yaw_rad = 0.0;
    roll_rad = acos(rot.at<float>(1, 1));
  }
  return cv::Vec3d(roll_rad, pitch_rad, yaw_rad);
}

cv::Vec3d radToDegree(const cv::Vec3d& radians) {
  cv::Vec3d ret;
  for (size_t i = 0; i < 3; ++i) {
    ret[i] = (180.0 / M_PI) * radians[i];
  }
  return ret;
}

void printRPY(const cv::Vec3d& rpy) {
  std::cout << "[Pose Evaluation] Rotation Difference Roll: " << rpy[0]
            << " deg Pitch: " << rpy[1] << " deg Yaw: " << rpy[2] << std::endl;
}

PoseEstimation::PoseEstimation() {}

Extrinsics PoseEstimation::estimatePose(
    const bool evaluation, const std::vector<cv::Point3f>& points3d,
    const std::vector<cv::Point2f>& points2d, const Image& img) {
  auto start = std::chrono::high_resolution_clock::now();

  if (points2d.size() != points3d.size()) {
    std::cout << "[Pose Estimation] numer of 2D points and numer of 3D points "
                 "need to be equal."
              << std::endl;
  }
  if (points2d.size() < 7) {
    std::cout << "[Pose Estimation] min 6 Points are required for the pose "
                 "estimation. Pose could not be calculated."
              << std::endl;
    Extrinsics e;
    return e;
  }

  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
  cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
  int iterationsCount = 700;
  float reprojectionError = 4.0;
  float confidence = 0.97;
  bool useExtrinsicGuess = false;

  cv::Mat K;
  std::vector<int> inlier;
  cv::Mat(img.intrinsics.getK3x3()).convertTo(K, CV_32F);

  bool success;

  try {
    success = cv::solvePnPRansac(points3d, points2d, K, cv::noArray(), rvec,
                                 tvec, useExtrinsicGuess, iterationsCount,
                                 reprojectionError, confidence, inlier,
                                 cv::SOLVEPNP_EPNP);
  } catch (...) {
    std::cout << "[PoseEstimation] Exception catched. Pose not calculatable."
              << std::endl;
    Extrinsics e;
    return e;
  }
  std::cout << "Inliers: " << inlier.size() << std::endl;

  Extrinsics::TransformationDirection direction =
      Extrinsics::TransformationDirection::Ref2Local;

  std::cout << "pose estimation " << (success ? "successfull" : "failed")
            << std::endl;
  if (!success) {
    Extrinsics e;
    return e;
  }

  cv::Mat rotationMatrix = cv::Mat::zeros(3, 3, CV_64F);
  Rodrigues(rvec, rotationMatrix);

  Extrinsics e(cv::Vec3d(rvec), cv::Vec3d(tvec), direction);

  auto finish = std::chrono::high_resolution_clock::now();

  std::cout << "Calculated Rotation: " << std::endl;
  std::cout << rotationMatrix << std::endl;

  std::cout << "Calculated Translation: " << std::endl;
  std::cout << tvec << std::endl;

  std::cout << "Real Rotation: " << std::endl;
  std::cout << img.extrinsics.getRotationMatrix(direction) << std::endl;

  std::cout << "Real Translation: " << std::endl;
  std::cout << img.extrinsics.getTranslation(direction) << std::endl;

  if (evaluation) {
    evaluatePose(img, e, rotationMatrix);
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "[Elapsed time] Pose estimation: " << elapsed.count()
              << " s\n";
  }

  return e;
}

void PoseEstimation::evaluatePose(const Image& img,
                                  const Extrinsics& extrinsics,
                                  cv::Mat& rotationMat) {
  Extrinsics::TransformationDirection direction =
      Extrinsics::TransformationDirection::Ref2Local;
  Extrinsics::TransformationDirection inverseDirection =
      Extrinsics::TransformationDirection::Local2Ref;

  cv::Vec3d translationDifference = img.extrinsics.getTranslation(direction) -
                                    extrinsics.getTranslation(direction);
  std::cout << "[Pose Evaluation] Translation Difference total: "
            << cv::norm(translationDifference) << std::endl;
  std::cout << "[Pose Evaluation] Translation Difference X: "
            << translationDifference[0] << std::endl;
  std::cout << "[Pose Evaluation] Translation Difference Y: "
            << translationDifference[1] << std::endl;
  std::cout << "[Pose Evaluation] Translation Difference Z: "
            << translationDifference[2] << std::endl;

  cv::Vec4d inversePoseQI =
      img.extrinsics.getRotationQuaternion(inverseDirection);
  cv::Vec4d calculatedPoseQI = extrinsics.getRotationQuaternion(direction);
  cv::Vec4d resultQI = multiplyQuaternions(inversePoseQI, calculatedPoseQI);
  double angle_rad = 2 * acos(resultQI[0]);
  double angle_deg = (180 / M_PI) * angle_rad;
  std::cout << "[Pose Evaluation] Rotation Difference total: " << angle_deg
            << " degrees" << std::endl;

  cv::Matx33d gtRot = img.extrinsics.getRotationMatrix(direction);
  cv::Mat gtRotM = cv::Mat(gtRot);

  cv::Mat diffRotMat = rotationMat.t() * gtRotM;
  std::cout << "[Pose Evaluation] Rotation Difference Matrix:\n"
            << diffRotMat << std::endl;
  cv::Mat diffFloat;
  diffRotMat.convertTo(diffFloat, CV_32F);
  cv::Vec3d rpy = calculateRPY(diffFloat);
  rpy = radToDegree(rpy);
  printRPY(rpy);

  img.csvrow->distanceX = translationDifference[0];
  img.csvrow->distanceY = translationDifference[1];
  img.csvrow->distanceZ = translationDifference[2];
  img.csvrow->distance = cv::norm(translationDifference);
  img.csvrow->roll = rpy[0];
  img.csvrow->pitch = rpy[1];
  img.csvrow->yaw = rpy[2];
  img.csvrow->angle = angle_deg;
}

cv::Vec4d PoseEstimation::multiplyQuaternions(cv::Vec4d a, cv::Vec4d b) {
  cv::Vec4d result;
  //  w:0  x:1  y:2  z:3
  result[1] = a[1] * b[0] + a[2] * b[3] - a[3] * b[2] + a[0] * b[1];
  result[2] = -a[1] * b[3] + a[2] * b[0] + a[3] * b[1] + a[0] * b[2];
  result[3] = a[1] * b[2] - a[2] * b[1] + a[3] * b[0] + a[0] * b[3];
  result[0] = -a[1] * b[1] - a[2] * b[2] - a[3] * b[3] + a[0] * b[0];
  return result;
}
