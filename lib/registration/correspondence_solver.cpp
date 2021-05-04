#include "correspondence_solver.h"

#include <types/image.h>

#include <cassert>
#include <opencv2/opencv.hpp>

CorrespondenceSolverBase::~CorrespondenceSolverBase() {}

bool compareMatchesQueryIdxSmaller(const cv::DMatch& i, const cv::DMatch& j) {
  return (i.queryIdx < j.queryIdx);
}

CorrespondenceSolver::CorrespondenceSolver() {}

bool CorrespondenceSolver::matchFeatures(
    std::vector<std::shared_ptr<Image>>& images,
    std::vector<std::vector<std::tuple<int, cv::Point2f>>>& correspondences) {
  std::vector<std::vector<cv::KeyPoint>> keypointVector;
  std::vector<cv::Mat> descriptorVector;
  for (const auto& image : images) {
    keypointVector.push_back(image->siftKeypoints);
    descriptorVector.push_back(image->siftDescriptors);
  }
  bool returnValue =
      matchFeatures(keypointVector, descriptorVector, images, correspondences);
  return returnValue;
}

bool CorrespondenceSolver::matchFeatures(
    std::vector<std::vector<cv::KeyPoint>>& keypointVector,
    std::vector<cv::Mat>& descriptorVector,
    std::vector<std::shared_ptr<Image>>& images,
    std::vector<std::vector<std::tuple<int, cv::Point2f>>>& correspondences) {
  if (keypointVector.size() != descriptorVector.size()) {
    std::cout << "[CorrespondenceSolver] Error, keypointVector and "
                 "descriptorVector must have the same size."
              << std::endl;
    return false;
  }
  if (keypointVector.size() <= 2) {
    std::cout << "[CorrespondenceSolver] to few passed images." << std::endl;
    return false;
  }

  // Calculate for every image the matches with the query image
  std::vector<std::vector<cv::DMatch>> matchesWithQueryImage;
  for (uint i = 1; i < keypointVector.size(); i++) {
    std::vector<cv::DMatch> matches;
    if (matchFeaturesForTwoImages(keypointVector.at(0), descriptorVector.at(0),
                                  keypointVector.at(i), descriptorVector.at(i),
                                  matches)) {
      matchesWithQueryImage.push_back(matches);
    } else {
      keypointVector.erase(keypointVector.begin() + i);
      descriptorVector.erase(descriptorVector.begin() + i);
      images.erase(images.begin() + 1);
      i--;
    }
  }
  if (matchesWithQueryImage.size() < 2) {
    std::cout << "[CorrespondenceSolver] too few matching images." << std::endl;
    return false;
  }

  // Sort the matches
  for (auto& match : matchesWithQueryImage) {
    std::sort(match.begin(), match.end(), compareMatchesQueryIdxSmaller);
  }

  // Merge the query image matches
  correspondences.clear();
  for (auto i = 0; i < keypointVector[0].size(); i++) {
    std::vector<std::tuple<int, cv::Point2f>> newVec;
    newVec.push_back(std::tuple<int, cv::Point2f>(0, keypointVector[0][i].pt));
    correspondences.push_back(newVec);
  }
  for (auto i = 0; i < matchesWithQueryImage.size(); i++) {
    insertCorrespondences(correspondences, matchesWithQueryImage[i],
                          keypointVector[i + 1], i + 1);
  }
  for (auto i = 0; i < correspondences.size(); i++) {
    if (correspondences[i].size() < 3) {
      correspondences.erase(correspondences.begin() + i);
      i--;
    }
  }

  if (correspondences.size() < 10) {
    std::cout << "correspondences.size() = " << correspondences.size()
              << std::endl;
    return false;
  }

  return true;
}

bool CorrespondenceSolver::matchFeaturesForTwoImages(
    const std::vector<cv::KeyPoint>& keypointsA, const cv::Mat& descriptorsA,
    const std::vector<cv::KeyPoint>& keypointsB, const cv::Mat& descriptorsB,
    std::vector<cv::DMatch>& matches) {
  if (keypointsA.size() < 10 || keypointsB.size() < 10) {
    std::cout << "[Corresponcence Solver] wrong kp size." << std::endl;
    return false;
  }
  if (keypointsA.size() != descriptorsA.rows ||
      keypointsB.size() != descriptorsB.rows) {
    std::cout << "[Corresponcence Solver] wrong descriptor size: "
              << keypointsA.size() << "," << descriptorsA.rows << " "
              << keypointsB.size() << "," << descriptorsB.rows << std::endl;
    return false;
  }
  if (descriptorsA.type() != descriptorsB.type()) {
    std::cout << "wrong descriptor type" << std::endl;
    return false;
  }
  std::vector<std::vector<cv::DMatch>> tempMatches;
  cv::Ptr<cv::DescriptorMatcher> MATCHER =
      cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  MATCHER->knnMatch(descriptorsA, descriptorsB, tempMatches, 2);
  // https://github.com/834810071/OpenCV_SFM/blob/master/OpenCV_SFM/MonocularReconstruction.cpp

  std::vector<cv::Vec3b> c1, c2;

  cv::Mat mask;
  float min_dist = FLT_MAX;
  for (int r = 0; r < tempMatches.size(); ++r) {
    // Rotio Test
    if (tempMatches[r][0].distance > 0.8 * tempMatches[r][1].distance) continue;
    float dist = tempMatches[r][0].distance;
    if (dist < min_dist) min_dist = dist;
  }
  matches.clear();
  for (size_t r = 0; r < tempMatches.size(); ++r) {
    // Ratio Test
    if (tempMatches[r][0].distance > 0.6 * tempMatches[r][1].distance ||
        tempMatches[r][0].distance > 5 * std::max(min_dist, 10.0f))
      continue;
    matches.push_back(tempMatches[r][0]);
  }
  if (matches.size() == 0) {
    std::cout << "no matches found!" << std::endl;
    return false;
  }
  // keypoints to points
  std::vector<cv::Point2f> pA, pB;
  pA.clear();
  pB.clear();
  for (cv::DMatch match : matches) {
    pA.push_back(keypointsA.at(match.queryIdx).pt);
    pB.push_back(keypointsB.at(match.trainIdx).pt);
  }
  // fundamental mat to filter matches
  cv::Mat F = cv::findFundamentalMat(pA, pB, mask, cv::RANSAC);
  // filter matches
  std::vector<cv::DMatch> goodMatches;
  for (int row = 0; row < mask.rows; row++) {
    if (mask.at<uchar>(cv::Point(0, row)) == 1.0) {
      goodMatches.push_back(matches.at(row));
    }
  }
  matches = goodMatches;
  return true;
}

void CorrespondenceSolver::getFeaturePoints(
    const std::vector<cv::KeyPoint>& firstKeypoints,
    const std::vector<cv::KeyPoint>& secondKeypoints,
    const std::vector<cv::DMatch>& goodMatches,
    std::vector<cv::Point2f>& firstCoordinates,
    std::vector<cv::Point2f>& secondCoordinates) {
  firstCoordinates.clear();
  secondCoordinates.clear();

  for (int i = 0; i < goodMatches.size(); i++) {
    firstCoordinates.push_back(
        firstKeypoints.at(goodMatches.at(i).queryIdx).pt);
    secondCoordinates.push_back(
        secondKeypoints.at(goodMatches.at(i).trainIdx).pt);
  }
}

void CorrespondenceSolver::getFeaturePoints(
    const std::vector<cv::KeyPoint>& firstKeypoints,
    const std::vector<cv::KeyPoint>& secondKeypoints,
    const std::vector<cv::DMatch>& goodMatches, IdxsPtsTupel& firstTupel,
    IdxsPtsTupel& secondTupel) {
  firstTupel.indexes.clear();
  firstTupel.points.clear();
  secondTupel.indexes.clear();
  secondTupel.points.clear();

  for (int i = 0; i < goodMatches.size(); i++) {
    firstTupel.indexes.push_back(goodMatches.at(i).queryIdx);
    firstTupel.points.push_back(
        firstKeypoints.at(goodMatches.at(i).queryIdx).pt);
    secondTupel.indexes.push_back(goodMatches.at(i).trainIdx);
    secondTupel.points.push_back(
        secondKeypoints.at(goodMatches.at(i).trainIdx).pt);
  }
}

void CorrespondenceSolver::insertCorrespondences(
    std::vector<std::vector<std::tuple<int, cv::Point2f>>>& correspondences,
    const std::vector<cv::DMatch>& matches,
    const std::vector<cv::KeyPoint>& keypoints, const int imageIndex) {
  for (int i = 0; i < matches.size(); i++) {
    correspondences[matches[i].queryIdx].push_back(std::tuple<int, cv::Point2f>(
        imageIndex, keypoints.at(matches[i].trainIdx).pt));
  }
}

void CorrespondenceSolver::drawCorrespondences(
    const std::vector<cv::KeyPoint>& keypointsA, const cv::Mat& descriptorsA,
    const cv::Mat& imgA, const std::vector<cv::KeyPoint>& keypointsB,
    const cv::Mat& descriptorsB, const cv::Mat& imgB) {
  std::vector<cv::DMatch> match;
  matchFeaturesForTwoImages(keypointsA, descriptorsA, keypointsB, descriptorsB,
                            match);
  cv::Mat output;
  cv::drawMatches(imgA, keypointsA, imgB, keypointsB, match, output);
  std::stringstream path;
  path << "/home/anne/Praktikum/ergebnisbilder/bildCorrespondences.png";
  cv::imwrite(path.str(), output);
  cv::imshow("Test", output);
  cv::waitKey(0);
}
