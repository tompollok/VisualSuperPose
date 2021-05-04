#pragma once

#include <types/image.h>

#include <opencv2/core/mat.hpp>
#include <string>

#undef slots
#include <torch/script.h>
#include <torch/torch.h>
#define slots Q_SLOTS

#include "correspondence_solver.h"
#include "ppbafloc-registration_export.h"

/**
 * @brief Deep Learning based implementation of the Correspondence Solver with
 * SuperPoint and SuperGlue using C++ binding of PyTorch
 */
class PPBAFLOC_REGISTRATION_EXPORT SuperGlueMatcher
    : public CorrespondenceSolverBase {
 public:
  /**
   * @param superPointModel Path to the SuperPoint model traced for the
   * TorchScript JIT compiler
   * @param superGlueModel Path to the SuperGlue model traced for the
   * TorchScript JIT compiler
   * @param targetWidth width to rescale the input images to (height is
   * calculated accordingly to keep aspect ratio). Keep at -1 for no rescale
   */
  SuperGlueMatcher(const std::string& superPointModel,
                   const std::string& superGlueModel, int targetWidth = -1);

  /**
   * @brief matchFeatures: run correspondence matching on image list
   * @param images: Vector of query image (first index) followed by reference
   * images to match
   * @param correspondences: resulting correspondences.
   * @return true if correspondence search was successfull, false otherwise.
   */
  bool matchFeatures(std::vector<std::shared_ptr<Image>>& images,
                     std::vector<std::vector<std::tuple<int, cv::Point2f>>>&
                         outCorrespondences) override;

  /**
   * @brief verbose console output
   */
  void setVerbose(bool active) { mVerbose = active; }

 private:
  struct SGMImage {
    torch::Tensor image;
    cv::Point2f scale;
    torch::Tensor keypoints, scores, descriptors;
  };

  void loadAndDetect(SGMImage& img, std::string path);
  void match(SGMImage& query, SGMImage& train,
             std::vector<std::vector<std::tuple<int, cv::Point2f>>>&
                 outCorrespondences,
             int trainImageIndex);

 private:
  std::string mSuperPointModelPath;
  std::string mSuperGlueModelPath;

  torch::jit::script::Module mSuperPointModel;
  torch::jit::script::Module mSuperGlueModel;
  torch::Device mDevice;

  int mTargetWidth = -1;
  bool mVerbose = false;
};
