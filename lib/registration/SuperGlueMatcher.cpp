#include "SuperGlueMatcher.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

SuperGlueMatcher::SuperGlueMatcher(const std::string &superPointModel,
                                   const std::string &superGlueModel,
                                   int targetWidth)
    : mDevice(torch::kCPU), mTargetWidth(targetWidth) {
  this->mSuperPointModelPath = superPointModel;
  this->mSuperGlueModelPath = superGlueModel;

  torch::manual_seed(1);
  torch::autograd::GradMode::set_enabled(false);

  mSuperPointModel = torch::jit::load(this->mSuperPointModelPath);
  mSuperGlueModel = torch::jit::load(this->mSuperGlueModelPath);

  if (torch::cuda::is_available()) {
    std::cout << "SG Inference on GPU" << std::endl;
    mDevice = torch::Device(torch::kCUDA);
  } else {
    std::cout << "SG Inference on CPU" << std::endl;
  }

  mSuperPointModel.eval();
  mSuperPointModel.to(mDevice);

  mSuperGlueModel.eval();
  mSuperGlueModel.to(mDevice);
}

void SuperGlueMatcher::loadAndDetect(SGMImage &img, std::string path) {
  cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
  image.convertTo(image, CV_32F, 1.0f / 255.0f);

  if (mTargetWidth >= 0) {
    img.scale.x = (float)mTargetWidth / image.cols;
    img.scale.y = img.scale.x;
    int target_height = std::lround(img.scale.x * image.rows);
    cv::resize(image, image, cv::Size(mTargetWidth, target_height));
  } else {
    img.scale = cv::Point2f(1.f, 1.f);
  }

  img.image = torch::from_blob(image.data, {1, 1, image.rows, image.cols},
                               torch::TensorOptions().dtype(torch::kFloat32))
                  .clone()
                  .to(mDevice);
  img.scale.x = 1.f / img.scale.x;
  img.scale.y = 1.f / img.scale.y;

  auto result = mSuperPointModel.forward({img.image}).toGenericDict();
  img.keypoints = result.at("keypoints").toTensorVector()[0];
  img.scores = result.at("scores").toTensorVector()[0];
  img.descriptors = result.at("descriptors").toTensorVector()[0];
}

void SuperGlueMatcher::match(
    SGMImage &query, SGMImage &train,
    std::vector<std::vector<std::tuple<int, cv::Point2f>>> &outCorrespondences,
    int trainImageIndex) {
  torch::Dict<std::string, torch::Tensor> input;
  input.insert("image0", query.image);
  input.insert("image1", train.image);
  input.insert("keypoints0", query.keypoints.unsqueeze(0));
  input.insert("keypoints1", train.keypoints.unsqueeze(0));
  input.insert("scores0", query.scores.unsqueeze(0));
  input.insert("scores1", train.scores.unsqueeze(0));
  input.insert("descriptors0", query.descriptors.unsqueeze(0));
  input.insert("descriptors1", train.descriptors.unsqueeze(0));

  torch::Dict<std::string, torch::Tensor> pred =
      c10::impl::toTypedDict<std::string, torch::Tensor>(
          mSuperGlueModel.forward({input}).toGenericDict());

  auto matches = pred.at("matches0")[0];

  size_t numValid = 0;
  for (size_t queryIdx = 0; queryIdx < static_cast<size_t>(matches.size(0));
       ++queryIdx) {
    int trainIdx = matches[queryIdx].item<int>();
    if (trainIdx > -1) {
      numValid++;
      cv::Point2f pt;
      pt.x = train.keypoints[trainIdx][0].item<float>() * train.scale.x;
      pt.y = train.keypoints[trainIdx][1].item<float>() * train.scale.y;
      outCorrespondences[queryIdx].push_back(
          std::tuple<int, cv::Point2f>(trainImageIndex, pt));
    }
  }

  if (mVerbose) {
    std::cout << trainImageIndex << " - #matches: " << numValid << std::endl;
  }
}

bool SuperGlueMatcher::matchFeatures(
    std::vector<std::shared_ptr<Image>> &images,
    std::vector<std::vector<std::tuple<int, cv::Point2f>>>
        &outCorrespondences) {
  SGMImage query;
  loadAndDetect(query, images[0]->path);
  if (mVerbose) {
    std::cout << images[0]->path << " - #kp: " << query.keypoints.size(0)
              << std::endl;
  }

  outCorrespondences.clear();
  outCorrespondences.resize(query.keypoints.size(0));
  for (size_t i = 0; i < static_cast<size_t>(query.keypoints.size(0)); ++i) {
    auto pt = query.keypoints[i];
    cv::Point2f pt2;
    pt2.x = pt[0].item<float>() * query.scale.x;
    pt2.y = pt[1].item<float>() * query.scale.y;
    outCorrespondences[i].push_back(std::tuple<int, cv::Point2f>(0, pt2));
  }

  for (size_t i = 1; i < images.size(); ++i) {
    SGMImage img;

    loadAndDetect(img, images[i]->path);
    if (mVerbose) {
      std::cout << images[i]->path << " - #kp: " << img.keypoints.size(0)
                << std::endl;
    }

    match(query, img, outCorrespondences, i);
  }

  // Remove points with only one or zero matches
  outCorrespondences.erase(
      std::remove_if(outCorrespondences.begin(), outCorrespondences.end(),
                     [](auto &c) { return c.size() < 3; }),
      outCorrespondences.end());

  std::cout << "kp with corr: " << outCorrespondences.size() << std::endl;

  return outCorrespondences.size() >= 10;
}
