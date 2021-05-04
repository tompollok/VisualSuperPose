#ifndef POSE_ESTIMATION_H
#define POSE_ESTIMATION_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../core/types/image.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "ppbafloc-registration_export.h"

class PPBAFLOC_REGISTRATION_EXPORT PoseEstimation {
 public:
  PoseEstimation();

  /**
   * @brief PoseEstimation::estimatePose
   * @param evaluation: if true, evauluation outputs are printed, otherwise not.
   * @param points3d: triangulated 3d points.
   * @param points2d: correspondenting 2d points to the 3d points.
   * @param img: passed query image.
   * @return calculated 6DoF-camera pose.
   */
  Extrinsics estimatePose(const bool evaluation,
                          const std::vector<cv::Point3f>&,
                          const std::vector<cv::Point2f>&, const Image&);

 private:
  /**
   * @brief PoseEstimation::evaluatePose
   * Calculates all evaluation values.
   * @param img: passed query image.
   * @param extrinsics: Calculated 6DoF-camera pose.
   * @param rotationMat: Calculated rotation represented as rotation mat.
   */
  void evaluatePose(const Image&, const Extrinsics&, cv::Mat&);

  /**
   * @brief PoseEstimation::multiplyQuaternions
   * This method multiplies two passed quaternions.
   * @param a: First passed quaternion.
   * @param b: Second passed quaternion.
   * @return multiplied quaternions.
   */
  cv::Vec4d multiplyQuaternions(cv::Vec4d, cv::Vec4d);
};

#endif  // POSE_ESTIMATION_H
