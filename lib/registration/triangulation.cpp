#include "triangulation.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../core/types/extrinsics.h"
#include "opencv2/imgproc/imgproc.hpp"

/**
 * @brief Triangulation::Triangulation
 * Constructor
 */
Triangulation::Triangulation() {}

bool Triangulation::triangulateSeveralImages(
    const bool evaluation, std::vector<std::shared_ptr<Image>>& images,
    std::vector<std::vector<std::tuple<int, cv::Point2f>>>& correspondences,
    std::vector<cv::Point2f>& points2d, std::vector<cv::Point3f>& points3d)

{
  auto start = std::chrono::high_resolution_clock::now();

  points2d.clear();
  points3d.clear();
  std::vector<TriangulationParams> triangulationParams;
  for (auto i = 1; i < images.size() - 1; i++) {
    for (auto j = i + 1; j < images.size(); j++) {
      TriangulationParams newTriangulationParams;
      newTriangulationParams.imageIndexA = i;
      newTriangulationParams.imageIndexB = j;
      for (const auto& correspondence : correspondences) {
        std::tuple<int, cv::Point2f> tupleQuery;
        std::tuple<int, cv::Point2f> tupleA;
        std::tuple<int, cv::Point2f> tupleB;
        bool initializedTupelA = false;
        bool initializedTupelB = false;
        for (const auto& tuple : correspondence) {
          if (std::get<0>(tuple) == 0) {
            tupleQuery = tuple;
          } else if (std::get<0>(tuple) == i) {
            tupleA = tuple;
            initializedTupelA = true;
          } else if (std::get<0>(tuple) == j) {
            tupleB = tuple;
            initializedTupelB = true;
          }
        }
        if (initializedTupelA && initializedTupelB) {
          newTriangulationParams.pointsA.push_back(std::get<1>(tupleA));
          newTriangulationParams.pointsB.push_back(std::get<1>(tupleB));
          newTriangulationParams.correspondingPoints.push_back(
              std::get<1>(tupleQuery));
        }
      }
      if (newTriangulationParams.correspondingPoints.size() >= 5) {
        triangulationParams.push_back(newTriangulationParams);
      }
    }
  }

  if (triangulationParams.size() == 0) {
    std::cout << "[Triangulation] Error: Triangulation params size is zero."
              << std::endl;
    return false;
  }

  for (auto& triangulationParam : triangulationParams) {
    std::vector<cv::Point3f> resultPoints;
    triangulate(*images[triangulationParam.imageIndexA],
                *images[triangulationParam.imageIndexB],
                triangulationParam.pointsA, triangulationParam.pointsB,
                resultPoints);
    points2d.insert(points2d.end(),
                    triangulationParam.correspondingPoints.begin(),
                    triangulationParam.correspondingPoints.end());
    points3d.insert(points3d.end(), resultPoints.begin(), resultPoints.end());
  }

  auto finish = std::chrono::high_resolution_clock::now();

  if (evaluation) {
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "[Elapsed time] Triangulation: " << elapsed.count() << " s\n";
  }
  return true;
}

void Triangulation::triangulate(
    Image& A, Image& B, std::vector<cv::Point2f>& firstKeypointCoordinates,
    std::vector<cv::Point2f>& secondKeypointCoordinates,
    std::vector<cv::Point3f>& result) {
  cv::Mat transformationMatA(cv::Size(4, 3), CV_32F, cv::Scalar(0));
  cv::Mat transformationMatB(cv::Size(4, 3), CV_32F, cv::Scalar(0));
  getTransformationMat(A, transformationMatA);
  getTransformationMat(B, transformationMatB);

  std::vector<cv::Point2f> firstUndistortedPoints, secondUndistortedPoints;
  cv::undistortPoints(firstKeypointCoordinates, firstUndistortedPoints,
                      A.intrinsics.getK3x3(),
                      A.intrinsics.distorionCoefficients(), cv::noArray(),
                      A.intrinsics.getK3x3());
  cv::undistortPoints(secondKeypointCoordinates, secondUndistortedPoints,
                      B.intrinsics.getK3x3(),
                      B.intrinsics.distorionCoefficients(), cv::noArray(),
                      B.intrinsics.getK3x3());

  cv::Mat pnts3D(1, firstKeypointCoordinates.size(), CV_32FC4);
  cv::triangulatePoints(transformationMatA, transformationMatB,
                        firstUndistortedPoints, secondUndistortedPoints,
                        pnts3D);
  convert3DPoints(pnts3D, result);
}

void Triangulation::getTransformationMat(Image& img,
                                         cv::Mat& transformationMat) {
  Extrinsics::TransformationDirection direction =
      Extrinsics::TransformationDirection::Ref2Local;

  cv::Mat extrinsicCalibration(cv::Size(4, 4), CV_32F, cv::Scalar(0));
  cv::Mat intrinsicCalibration(cv::Size(4, 4), CV_32F, cv::Scalar(0));
  extrinsicCalibration.at<float>(3, 3) = 1;
  intrinsicCalibration.at<float>(3, 3) = 1;

  setTranslationVector(img.extrinsics.getTranslation(direction),
                       extrinsicCalibration);
  setMatrix(img.extrinsics.getRotationMatrix(direction), extrinsicCalibration);
  setMatrix(img.intrinsics.getK3x3(), intrinsicCalibration);

  cv::Mat transformation = intrinsicCalibration * extrinsicCalibration;

  cv::Rect rect(0, 0, 4, 3);
  transformation(rect).copyTo(transformationMat(rect));
}

void Triangulation::setTranslationVector(const cv::Vec3d& translationVec,
                                         cv::Mat& homogeneousMat) {
  assert(homogeneousMat.cols == 4);
  assert(homogeneousMat.rows == 4);

  homogeneousMat.at<float>(0, 3) = translationVec[0];
  homogeneousMat.at<float>(1, 3) = translationVec[1];
  homogeneousMat.at<float>(2, 3) = translationVec[2];
}

void Triangulation::setMatrix(const cv::Matx33d& rotationMat,
                              cv::Mat& homogeneousMat) {
  assert(rotationMat.cols == 3);
  assert(rotationMat.rows == 3);
  assert(homogeneousMat.cols == 4);
  assert(homogeneousMat.rows == 4);

  homogeneousMat.at<float>(0, 0) = rotationMat(0, 0);
  homogeneousMat.at<float>(0, 1) = rotationMat(0, 1);
  homogeneousMat.at<float>(0, 2) = rotationMat(0, 2);

  homogeneousMat.at<float>(1, 0) = rotationMat(1, 0);
  homogeneousMat.at<float>(1, 1) = rotationMat(1, 1);
  homogeneousMat.at<float>(1, 2) = rotationMat(1, 2);

  homogeneousMat.at<float>(2, 0) = rotationMat(2, 0);
  homogeneousMat.at<float>(2, 1) = rotationMat(2, 1);
  homogeneousMat.at<float>(2, 2) = rotationMat(2, 2);
}

void Triangulation::convert3DPoints(const cv::Mat& triangulationResult,
                                    std::vector<cv::Point3f>& points3f) {
  for (int i = 0; i < triangulationResult.cols; i++) {
    cv::Point3d newPoint;
    newPoint.x = triangulationResult.at<float>(0, i) /
                 triangulationResult.at<float>(3, i);
    newPoint.y = triangulationResult.at<float>(1, i) /
                 triangulationResult.at<float>(3, i);
    newPoint.z = triangulationResult.at<float>(2, i) /
                 triangulationResult.at<float>(3, i);
    points3f.push_back(newPoint);
  }
}
