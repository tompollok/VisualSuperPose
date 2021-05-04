#include "registration.h"

#include "SuperGlueMatcher.h"
#include "pose_estimation.h"
#include "triangulation.h"

bool poseEstimation(bool evaluation, CorrespondenceSolverBase& solver,
                    std::shared_ptr<Image>& queryImage,
                    std::vector<std::shared_ptr<Image>>& retrievalImages,
                    Extrinsics& result,
                    std::vector<cv::Point3f>& triangulatedPoints) {
  auto start = std::chrono::high_resolution_clock::now();
  std::cout << "Number retrieved images: " << retrievalImages.size()
            << std::endl;

  std::vector<std::shared_ptr<Image>> images;
  images.push_back(queryImage);
  images.insert(images.end(), retrievalImages.begin(), retrievalImages.end());
  std::cout << "Number images: " << images.size() << std::endl;

  std::vector<std::vector<std::tuple<int, cv::Point2f>>> correspondences;
  auto t1 = std::chrono::high_resolution_clock::now();
  bool matchingSuccess = solver.matchFeatures(images, correspondences);
  auto t2 = std::chrono::high_resolution_clock::now();
  if (evaluation) {
    std::cout << "[Elapsed time] Correspondence Solver: "
              << std::chrono::duration<double>(t2 - t1).count() << " s"
              << std::endl;
  }

  if (!matchingSuccess) {
    std::cout << "Too few matches!" << std::endl;
    return false;
  }

  Triangulation triangulation;
  std::vector<cv::Point2f> points2f;
  bool triangulationSuccessfull = triangulation.triangulateSeveralImages(
      evaluation, images, correspondences, points2f, triangulatedPoints);
  if (!triangulationSuccessfull) {
    return false;
  }

  std::vector<cv::Point2f> undistortedPoints;
  cv::undistortPoints(points2f, undistortedPoints,
                      queryImage->intrinsics.getK3x3(),
                      queryImage->intrinsics.distorionCoefficients(),
                      cv::noArray(), queryImage->intrinsics.getK3x3());

  PoseEstimation poseEstimation;
  result = poseEstimation.estimatePose(evaluation, triangulatedPoints,
                                       undistortedPoints, *queryImage);

  auto finish = std::chrono::high_resolution_clock::now();
  if (evaluation) {
    std::cout << "Number of triangulated points: " << triangulatedPoints.size()
              << std::endl;
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "[Elapsed time] Complete registration: " << elapsed.count()
              << " s" << std::endl;
  }
  return true;
}

bool Registration::applyClassicPoseEstimation(
    bool evaluation, std::shared_ptr<Image>& queryImage,
    std::vector<std::shared_ptr<Image>>& retrievalImages, Extrinsics& result,
    std::vector<cv::Point3f>& triangulatedPoints) {
  CorrespondenceSolver solver;
  return poseEstimation(evaluation, solver, queryImage, retrievalImages, result,
                        triangulatedPoints);
}

bool Registration::applyDeepLearningBasedPoseEstimation(
    bool evaluation, std::shared_ptr<Image>& queryImage,
    std::vector<std::shared_ptr<Image>>& retrievalImages, Extrinsics& outResult,
    std::vector<cv::Point3f>& triangulatedPoints) {
  if (mDLMatching == nullptr) {
    throw std::runtime_error("DeepLearning Matching not initialized");
  }

  return poseEstimation(evaluation, *mDLMatching, queryImage, retrievalImages,
                        outResult, triangulatedPoints);
}

bool Registration::setupDeepLearningBasedPoseEstimation(
    const std::string& superpointModel, const std::string& superglueModel,
    int resize_width) {
  try {
    mDLMatching = std::make_unique<SuperGlueMatcher>(
        superpointModel, superglueModel, resize_width);
    return true;
  } catch (...) {
    mDLMatching = nullptr;
    return false;
  }
}
