#pragma once
#ifndef REGISTRATION_H
#define REGISTRATION_H

#include <types/image.h>

#include <memory>
#include <opencv2/opencv.hpp>

#include "correspondence_solver.h"
#include "ppbafloc-registration_export.h"

class PPBAFLOC_REGISTRATION_EXPORT Registration {
 public:
  Registration() {}

  /**
   * @brief Registration::applyClassicPoseEstimation
   * Calculates the pose of a passed queryImage by using the passed retrieval
   * images. Matching is performed with SIFT
   * @param evaluation: if true, evauluation outputs are printed, otherwise not.
   * @param queryImage: pointer to query image.
   * @param retrievalImages: vector of retrieval images.
   * @param outResult: calculated camera pose.
   * @param triangulatedPoints: the triangulated 3D points are saved into this
   * vector
   * @return true if the camera pose was successfully calculated, false
   * otherwise.
   */
  bool applyClassicPoseEstimation(
      bool evaluation, std::shared_ptr<Image>& queryImage,
      std::vector<std::shared_ptr<Image>>& retrievalImages,
      Extrinsics& outResult, std::vector<cv::Point3f>& triangulatedPoints);

  /**
   * @brief Registration::applyDeepLearningBasedPoseEstimation
   * Calculates the pose of a passed queryImage by using the passed retrieval
   * images. Matching is performed with SuperPoint and SuperGlue
   * @param evaluation: if true, evauluation outputs are printed, otherwise not.
   * @param queryImage: pointer to query image.
   * @param retrievalImages: vector of retrieval images.
   * @param outResult: calculated camera pose.
   * @param triangulatedPoints: the triangulated 3D points are saved into this
   * vector
   * @return true if the camera pose was successfully calculated, false
   * otherwise.
   */
  bool applyDeepLearningBasedPoseEstimation(
      bool evaluation, std::shared_ptr<Image>& queryImage,
      std::vector<std::shared_ptr<Image>>& retrievalImages,
      Extrinsics& outResult, std::vector<cv::Point3f>& triangulatedPoints);

  bool setupDeepLearningBasedPoseEstimation(const std::string& superpointModel,
                                            const std::string& superglueModel,
                                            int resize_width = -1);

 private:
  std::unique_ptr<CorrespondenceSolverBase> mDLMatching = nullptr;
};

#endif  // REGISTRATION_H
