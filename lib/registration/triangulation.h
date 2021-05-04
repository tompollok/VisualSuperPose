#ifndef TRIANGULATION_H
#define TRIANGULATION_H

#include "../core/types/image.h"
#include "ppbafloc-registration_export.h"

struct TriangulationParams {
  int imageIndexA;
  int imageIndexB;
  std::vector<cv::Point2f> pointsA;
  std::vector<cv::Point2f> pointsB;
  std::vector<cv::Point2f> correspondingPoints;
};

class PPBAFLOC_REGISTRATION_EXPORT Triangulation {
 public:
  /**
   * @brief Triangulation::Triangulation
   * Constructor
   */
  Triangulation();

  /**
   * @brief Triangulation::triangulateSeveralImages
   * Method for triangulating more than two images.
   * @param evaluation: true if evaluation values should be printed, false
   * otherwise.
   * @param images: passed images that should be triangulated.
   * @param correspondences: Calculated correspondences from the correspondence
   * solver.
   * @param points2d: correspondenting 2d points in the query image to the
   * resulting 3d points.
   * @param points3d: resulting triangulated points
   */
  bool triangulateSeveralImages(
      const bool, std::vector<std::shared_ptr<Image>>&,
      std::vector<std::vector<std::tuple<int, cv::Point2f>>>&,
      std::vector<cv::Point2f>&, std::vector<cv::Point3f>&);

  /**
   * @brief Triangulation::triangulate
   * Triangulates keypoints of two passed images.
   * @param A: first image
   * @param B: second image
   * @param firstKeypointCoordinates: keypoint coordinates of the first image
   * @param secondKeypointCoordinates: keypoint coordinates of the second image
   * @param result: resulting triangulated points
   */
  void triangulate(Image&, Image&, std::vector<cv::Point2f>&,
                   std::vector<cv::Point2f>&, std::vector<cv::Point3f>&);

 private:
  /**
   * @brief Triangulation::getTransformationMat
   * Returns a transformation mat as homogenous mat of a passed image.
   * @param img: input image
   * @param transformationMat: result homogenous transformation mat
   */
  void getTransformationMat(Image&, cv::Mat&);

  /**
   * @brief Triangulation::setTranslationVector
   * Sets a translation vector to a homogeneous mat.
   * @param translationVec: 3 dimensional translation vector
   * @param homogeneousMat: resulting homogeneous mat
   */
  void setTranslationVector(const cv::Vec3d&, cv::Mat&);

  /**
   * @brief Triangulation::setMatrix
   * Converts a rotation mat to a homogeneous mat
   * @param rotationMat: rotation mat to be converted
   * @param homogeneousMat: result homogeneous mat
   */
  void setMatrix(const cv::Matx33d&, cv::Mat&);

  /**
   * @brief Triangulation::convert3DPoints
   * Converts 3 dimensional points from the representation as cv::Mat to
   * std::vector<cv::Point3f>
   * @param triangulationResult: representation of points as cv::Mat
   * @param points3f: resulting representation of points as
   * std::vector<cv::Point3f>
   */
  void convert3DPoints(const cv::Mat&, std::vector<cv::Point3f>&);
};

#endif  // TRIANGULATION_H
