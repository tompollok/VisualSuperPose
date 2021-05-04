#ifndef CORRESPONDENCE_SOLVER_H
#define CORRESPONDENCE_SOLVER_H

#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>

#include "ppbafloc-registration_export.h"

class Image;

class PPBAFLOC_REGISTRATION_EXPORT CorrespondenceSolverBase {
 public:
  virtual ~CorrespondenceSolverBase();

  /**
   * @brief CorrespondenceSolverBase::matchFeatures
   * Calculates the matches between the query image (first passed image) and the
   * reference images.
   * @param images: Vector of query image (first index) followed by reference
   * images to match
   * @param correspondences: resulting correspondences.
   * @return true if correspondence search was successfull, false otherwise.
   */
  virtual bool matchFeatures(
      std::vector<std::shared_ptr<Image>>& images,
      std::vector<std::vector<std::tuple<int, cv::Point2f>>>&
          correspondences) = 0;
};

struct IdxsPtsTupel {
  std::vector<int> indexes;
  std::vector<cv::Point2f> points;
};

class PPBAFLOC_REGISTRATION_EXPORT CorrespondenceSolver
    : public CorrespondenceSolverBase {
 public:
  /**
   * @brief CorrespondenceSolver::CorrespondenceSolver
   * Constructor
   */
  CorrespondenceSolver();

  /**
   * @brief CorrespondenceSolver::matchFeatures
   * Calculates the matches between the query image (first passed image) and the
   * reference images.
   * @param images: passed images whose matches should be calculated.
   * @param correspondences: resulting correspondences.
   * @return true if correspondence search was successfull, false otherwise.
   */
  bool matchFeatures(
      std::vector<std::shared_ptr<Image>>&,
      std::vector<std::vector<std::tuple<int, cv::Point2f>>>&) override;

  /**
   * @brief CorrespondenceSolver::matchFeatures
   * Calculates the matches between the query image (first passed image) and the
   * reference images.
   * @param keypointVector: Passed keypoint vectors.
   * @param descriptorVector: Passed descriptor vectors.
   * @param images: Passed images. The first passed image should be the query
   * image.
   * @param correspondences: Calculated correspondences.
   * @return true if correspondence search was successfull, false otherwise.
   */
  bool matchFeatures(std::vector<std::vector<cv::KeyPoint>>&,
                     std::vector<cv::Mat>&,
                     std::vector<std::shared_ptr<Image>>&,
                     std::vector<std::vector<std::tuple<int, cv::Point2f>>>&);

  /**
   * @brief CorrespondenceSolver::drawCorrespondences
   * This method visualizes the calculated correspondences between two images.
   * @param keypointsA: calculated keypoints of the first image.
   * @param descriptorsA: calculated descriptors of the first image.
   * @param imgA: first passed image.
   * @param keypointsB: calculated keypoints of the second image.
   * @param descriptorsB: calculated descriptors of the second image.
   * @param imgB: second passed image.
   */
  void drawCorrespondences(const std::vector<cv::KeyPoint>&, const cv::Mat&,
                           const cv::Mat&, const std::vector<cv::KeyPoint>&,
                           const cv::Mat&, const cv::Mat&);

 private:
  /**
   * @brief CorrespondenceSolver::matchFeaturesForTwoImages
   * Calculates the matches for two images and their SIFT keypoints and
   * descriptors.
   * @param keypointsA: passed keypoints for the first image.
   * @param descriptorsA: passed descriptors for the first image.
   * @param keypointsB: passed keypoints for the second image.
   * @param descriptorsB: passed descriptors for the second image.
   * @param matches: calculated matches
   * @return true if the features could be matched
   */
  bool matchFeaturesForTwoImages(const std::vector<cv::KeyPoint>&,
                                 const cv::Mat&,
                                 const std::vector<cv::KeyPoint>&,
                                 const cv::Mat&, std::vector<cv::DMatch>&);

  /**
   * @brief CorrespondenceSolver::getFeaturePoints
   * Calculates the feature point coordinates of two images.
   * @param firstKeypoints: keypoints of the first image.
   * @param secondKeypoints: keypoints of the second image.
   * @param goodMatches: matches of the images.
   * @param firstCoordinates: feature coordinates of the first image.
   * @param secondCoordinates: feature coordinates of the second image.
   */
  void getFeaturePoints(const std::vector<cv::KeyPoint>&,
                        const std::vector<cv::KeyPoint>&,
                        const std::vector<cv::DMatch>&,
                        std::vector<cv::Point2f>&, std::vector<cv::Point2f>&);

  /**
   * @brief CorrespondenceSolver::getFeaturePoints
   * @param firstKeypoints: keypoints of the first image.
   * @param secondKeypoints: keypoints of the second image.
   * @param goodMatches: matches of the images.
   * @param firstTupel: resulting index points tupel.
   * @param secondTupel: resulting index points tupel.
   */
  void getFeaturePoints(const std::vector<cv::KeyPoint>&,
                        const std::vector<cv::KeyPoint>&,
                        const std::vector<cv::DMatch>&, IdxsPtsTupel&,
                        IdxsPtsTupel&);

  /**
   * @brief CorrespondenceSolver::insertCorrespondences
   * Initializes the correspondences.
   * @param correspondences: vector of correspondences that gets filled by this
   * method.
   * @param matches: passed matches.
   * @param keypoints: passed keypoints.
   * @param imageIndex: image index.
   */
  void insertCorrespondences(
      std::vector<std::vector<std::tuple<int, cv::Point2f>>>&,
      const std::vector<cv::DMatch>&, const std::vector<cv::KeyPoint>&,
      const int);
};

#endif  // CORRESPONDENCE_SOLVER_H
