#include "torchreidretriever.h"

#include <core.h>
#include <search.h>

#include <QtCore/QDataStream>
#include <QtCore/QDirIterator>
#include <QtCore/QStringList>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>

namespace {
bool comp(std::pair<double, std::shared_ptr<Image>> &a,
          std::pair<double, std::shared_ptr<Image>> &b) {
  return a.first < b.first;
}
bool compPath(std::pair<double, std::string> &a,
              std::pair<double, std::string> &b) {
  return a.first < b.first;
}

bool compId(std::pair<int, double> &a, std::pair<int, double> &b) {
  return a.second < b.second;
}

void calcScoreMultipleWithDB(
    Database &db, std::vector<std::vector<std::pair<int, double>>> &scores,
    const std::vector<cv::Mat> &queryHashes) {
  auto callback = [&](int id, QByteArray &hash) -> bool {
    QDataStream stream(&hash, QIODevice::ReadOnly);
    int matType, rows, cols;
    stream >> matType >> rows >> cols;
    QByteArray hashByte;
    stream >> hashByte;
    cv::Mat hashMat =
        cv::Mat(rows, cols, matType, (void *)hashByte.data()).clone();

    for (size_t i = 0; i < queryHashes.size(); ++i) {
      double dist = 0.;
      try {
        dist = cv::norm(queryHashes[i], hashMat, cv::NORM_L2);
      } catch (const std::exception &e) {
        std::cout << id << ":" << e.what() << std::endl;
      }

      scores[i].push_back({id, dist});
    }

    return true;
  };
  db.getHashAll(callback);
}

}  // namespace
TorchreidRetriever::TorchreidRetriever(){};

TorchreidRetriever::TorchreidRetriever(const QString &modelpath,
                                       const cv::Size &inputformat)
    : mInputFormat(inputformat) {
  mModel = cv::dnn::readNetFromONNX(modelpath.toStdString());
  mModel.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_CUDA);
  mModel.setPreferableTarget(cv::dnn::Target::DNN_TARGET_CUDA);
}

TorchreidRetriever::TorchreidRetriever(const QString &modelpath,
                                       const cv::Size &inputformat,
                                       const QString &galleryDirPath)
    : TorchreidRetriever(modelpath, inputformat) {
  mGalleryDirPath = galleryDirPath;
}

TorchreidRetriever::TorchreidRetriever(const QString &modelpath,
                                       const cv::Size &inputformat,
                                       Database *db)
    : TorchreidRetriever(modelpath, inputformat) {
  mDB = db;
}

void TorchreidRetriever::fillDatabaseHashes(int batchSize) {
  if (mDB == nullptr) {
    throw std::runtime_error(
        "TorchreidRetriever::fillDatabaseHashes no DB given!");
  }

  if (batchSize < 1) {
    batchSize = 1;
  }

  std::vector<std::pair<int, std::string>> paths;
  mDB->getPathList(paths);
  size_t n = paths.size();

  size_t batches = std::ceil(n / static_cast<float>(batchSize));

  for (size_t b = 0; b < batches; ++b) {
    auto t0 = std::chrono::high_resolution_clock::now();

    size_t offset = b * batchSize;
    size_t bsize = batchSize;
    if (b == batches - 1) {
      bsize = n - offset;
    }

    std::vector<std::string> imgPathList(bsize);
    for (size_t i = 0; i < bsize; ++i) {
      imgPathList[i] = paths[offset + i].second;
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<cv::Mat> results = applyModel(imgPathList);

    auto t2 = std::chrono::high_resolution_clock::now();

    mDB->transaction();
    for (size_t i = 0; i < bsize; ++i) {
      mDB->addHashVector(paths[offset + i].first, results[i]);
    }
    mDB->commit();

    auto t3 = std::chrono::high_resolution_clock::now();
    std::cout << "Batch " << b + 1 << "/" << batches << " "
              << std::chrono::duration<double>(t1 - t0).count() << " - "
              << std::chrono::duration<double>(t2 - t1).count() << " - "
              << std::chrono::duration<double>(t3 - t2).count() << std::endl;
  }
}

std::vector<std::vector<std::shared_ptr<Image>>>
TorchreidRetriever::findReferenceImagesMultipleQueries(
    const std::vector<std::shared_ptr<Image>> &queryImages,
    std::vector<std::shared_ptr<Image>> &galleryImgs,
    const uint64 maxReferenceCount, const int maxGalleryCount,
    const uint64 numThreads, const uint64 batchSize, bool useDatabase) {
  std::vector<cv::Mat> queryHashes;
  cv::Mat queryHash;

  for (const auto &query : queryImages) {
    queryHash = applyModel(query).clone();
    queryHashes.push_back(queryHash);
  }
  int endIndex = 0;

  std::vector<std::vector<std::pair<int, double>>> scores(queryHashes.size());
  std::vector<std::vector<std::pair<double, std::string>>> scoresWithPaths(
      queryHashes.size());
  std::vector<std::string> files;
  // std::cout << "number of images found " << n << std::endl;

  auto tStart = std::chrono::high_resolution_clock::now();
  if (useDatabase) {
    auto t0 = std::chrono::high_resolution_clock::now();
    calcScoreMultipleWithDB(*mDB, scores, queryHashes);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<double, std::string>> scoresOneQueryImage;
    for (size_t i = 0; i < queryHashes.size(); i++) {
      sort(scores[i].begin(), scores[i].end(), compId);
      size_t limit = maxReferenceCount + 1;
      scores[i].resize(std::min(scores[i].size(), limit));

      for (size_t j = 0; j < scores[i].size(); j++) {
        scoresOneQueryImage.push_back(
            {scores[i][j].second, this->mDB->getPath(scores[i][j].first)});
      }
      scoresWithPaths[i].insert(std::end(scoresWithPaths[i]),
                                std::begin(scoresOneQueryImage),
                                std::end(scoresOneQueryImage));
      scoresOneQueryImage.clear();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Retrieved CNN DB - "
              << std::chrono::duration<double>(t1 - t0).count() << " s - "
              << std::chrono::duration<double>(t2 - t1).count() << " s"
              << std::endl;
  } else {
    if (galleryImgs.empty()) {
      QStringList filter;
      filter << "*.jpg"
             << "*.png"
             << "*.jpeg";
      QDirIterator it(mGalleryDirPath, filter, QDir::Files,
                      QDirIterator::Subdirectories);

      int counter = 0;
      for (int i = 0; it.hasNext(); ++i) {
        files.push_back(it.next().toStdString());
        counter++;

        if (counter == maxGalleryCount - 1 && maxGalleryCount > 0) break;
      }
    } else {
      int limit = galleryImgs.size();
      if (maxGalleryCount > 0) {
        limit = maxGalleryCount;
      }
      for (int i = 0; i < limit; i++) {
        files.push_back(galleryImgs[i]->path);
      }
    }

    int n = files.size();
    for (const auto &queryImage : queryImages) {
      if (!queryImage->csvrow) {
        queryImage->csvrow = std::shared_ptr<CSVRow>(new CSVRow);
      }
      queryImage->csvrow->gallerySize = n;
    }

    if (numThreads > 1) {
      std::cout << "WARNING MULTIPLE THREADS FOR CNN RETRIEVAL WITHOUT DB NOT "
                   "IMPLEMENTED"
                << std::endl;
    }

    int numBatches = n / batchSize;
    std::vector<std::shared_ptr<Image>> batch;
    std::vector<cv::Mat> results;
    std::vector<std::pair<double, std::string>> oneQueryImageDists;
    for (int i = 0; i < numBatches + 1; i++) {
      if ((i + 1) * batchSize > n) {
        endIndex = numBatches * batchSize + (n % batchSize);
      } else {
        endIndex = (i + 1) * batchSize;
      }

      auto t0 = std::chrono::high_resolution_clock::now();
      results = applyModel(std::vector<std::string>(
          files.begin() + i * batchSize, files.begin() + endIndex));
      auto t1 = std::chrono::high_resolution_clock::now();

      for (size_t k = 0; k < queryImages.size(); k++) {
        // std::cout << "Calculating retrieval distances for query image " <<
        // queryImages[k]->path << std::endl;
        for (int j = 0; j < results.size(); j++) {
          double dist = cv::norm(queryHashes[k], results[j], cv::NORM_L2);
          oneQueryImageDists.push_back({dist, files.at(i * batchSize + j)});
        }
        scoresWithPaths[k].insert(std::end(scoresWithPaths[k]),
                                  std::begin(oneQueryImageDists),
                                  std::end(oneQueryImageDists));
        oneQueryImageDists.clear();
      }
      results.clear();

      auto t2 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> dtScore = t2 - t1;
      std::cout << "Batch " << i + 1 << "/" << numBatches + 1
                << ": Apply Model: "
                << std::chrono::duration<double>(t1 - t0).count()
                << " s - Score Calc: " << dtScore.count() << " ms" << std::endl;
    }
    std::cout << "Finished calculating distances" << std::endl;
    for (size_t i = 0; i < queryImages.size(); i++) {
      sort(scoresWithPaths[i].begin(), scoresWithPaths[i].end(), compPath);
      // if (scoresWithPaths.size() > static_cast<size_t>(maxReferenceCount))
      scoresWithPaths[i].resize(maxReferenceCount + 1);
    }
  }

  auto tEnd = std::chrono::high_resolution_clock::now();
  std::cout << "score calculation time: "
            << std::chrono::duration<double>(tEnd - tStart).count() << " s"
            << std::endl;

  std::vector<std::vector<std::shared_ptr<Image>>> retrievalImages(
      queryHashes.size());
  for (size_t i = 0; i < scoresWithPaths.size(); i++) {
    for (const auto &score : scoresWithPaths[i]) {
      if (score.second == queryImages[i]->path) {
        std::cout << "Removing query images from retrieved images" << std::endl;
        continue;
      }
      // std::cout << score.second << "distance is " << score.first <<
      // std::endl;
      std::shared_ptr<Image> t = std::shared_ptr<Image>(new Image);
      t->path = score.second;
      retrievalImages[i].push_back(t);
    }
  }
  return retrievalImages;
}

cv::Mat TorchreidRetriever::applyModel(const std::shared_ptr<Image> &image) {
  return applyModel(image->path);
}

cv::Mat TorchreidRetriever::applyModel(const std::string &imagePath) {
  cv::Mat imgc = cv::imread(imagePath, cv::IMREAD_COLOR);
  return applyModel(imgc);
}

cv::Mat TorchreidRetriever::applyModel(const cv::Mat &image) {
  cv::Mat detections = cv::Mat::zeros(1, 512, CV_32F);
  cv::Mat img = image;
  if (img.empty() || img.cols == 0 || img.rows == 0) {
    std::cout << "Empty image " << std::endl;
    return detections;
  }
  float rMean = 0.485, gMean = 0.456, bMean = 0.406;
  float rStd = 0.229, gStd = 0.224, bStd = 0.225;
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  img.convertTo(img, CV_32F, 1.0 / 255.0);
  cv::resize(img, img, mInputFormat);
  cv::Mat srcN = img.clone();
  srcN.forEach<cv::Vec3f>([rMean, gMean, bMean, rStd, gStd, bStd](
                              cv::Vec3f &pixel, const int *position) -> void {
    pixel[0] = (pixel[0] - rMean) / rStd;
    pixel[1] = (pixel[1] - gMean) / gStd;
    pixel[2] = (pixel[2] - bMean) / bStd;
  });
  cv::Mat blob = cv::dnn::blobFromImage(srcN, 1.0, mInputFormat,
                                        cv::Scalar(0, 0, 0), false);

  mModel.setInput(blob);

  auto start = std::chrono::high_resolution_clock::now();
  mModel.forward(detections);
  std::chrono::duration<double, std::milli> dt =
      std::chrono::high_resolution_clock::now() - start;
  std::cout << " Time taken for one forward pass " << dt.count() << " ms"
            << std::endl;
  cv::normalize(detections, detections);
  return detections.clone();
}

std::vector<cv::Mat> TorchreidRetriever::applyModel(
    const std::vector<std::string> &imagePaths) {
  std::vector<cv::Mat> img_list;
  cv::Mat img;
  float rMean = 0.485, gMean = 0.456, bMean = 0.406;
  float rStd = 0.229, gStd = 0.224, bStd = 0.225;
  cv::Mat srcN;

  double dtImread = 0.0, dtNormalize = 0.0;
  for (const auto &i : imagePaths) {
    if (i.empty()) {
      std::cout << "Path could not be read" << std::endl;
      continue;
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    img = cv::imread(i, cv::IMREAD_COLOR);
    auto t1 = std::chrono::high_resolution_clock::now();
    dtImread += std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (img.empty() || img.cols == 0 || img.rows == 0) {
      std::cout << "Empty image " << i << std::endl;
      continue;
    }
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32F, 1.0 / 255.0);
    cv::resize(img, img, mInputFormat);

    srcN = img.clone();

    auto t2 = std::chrono::high_resolution_clock::now();

    // next few lines based on:
    // https://stackoverflow.com/questions/47632756/the-fast-efficient-way-to-normalize-each-channel-of-an-image-with-different-valu
    // post by user Ja_cpp
    srcN.forEach<cv::Vec3f>([rMean, gMean, bMean, rStd, gStd, bStd](
                                cv::Vec3f &pixel, const int *position) -> void {
      pixel[0] = (pixel[0] - rMean) / rStd;
      pixel[1] = (pixel[1] - gMean) / gStd;
      pixel[2] = (pixel[2] - bMean) / bStd;
    });
    auto t3 = std::chrono::high_resolution_clock::now();
    dtNormalize += std::chrono::duration<double, std::milli>(t3 - t2).count();

    img_list.push_back(srcN);
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  std::vector<cv::Mat> results;
  results = applyModel(img_list);
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "ApplyModel(List): Imread: " << dtImread
            << " ms - Normalize: " << dtNormalize
            << " ms - ApplyModel(single): "
            << std::chrono::duration<double, std::milli>(t1 - t0).count()
            << " ms" << std::endl;

  return results;
}

std::vector<cv::Mat> TorchreidRetriever::applyModel(
    const std::vector<cv::Mat> &images) {
  cv::Mat total_blob = cv::dnn::blobFromImages(images, 1.0, mInputFormat,
                                               cv::Scalar(0, 0, 0), false);

  mModel.setInput(total_blob);
  cv::Mat detections;

  auto start = std::chrono::high_resolution_clock::now();
  mModel.forward(detections);
  std::chrono::duration<double, std::milli> dt =
      std::chrono::high_resolution_clock::now() - start;
  std::cout << " Time taken for one forward pass " << images.size()
            << " images - " << dt.count() << " ms" << std::endl;

  std::vector<cv::Mat> detectionList;
  cv::Mat slice;
  cv::Range ranges[2];
  ranges[0] = cv::Range::all();
  int numDimensions = detections.cols / images.size();

  for (int i = 0; i < images.size() * numDimensions; i = i + numDimensions) {
    ranges[1] = cv::Range(i, i + numDimensions);
    slice = detections(ranges);
    detections(ranges).copyTo(slice);
    cv::normalize(slice, slice);
    detectionList.push_back(cv::Mat(1, 512, CV_32F, slice.data).clone());
  }
  return detectionList;
}

std::vector<cv::Mat> TorchreidRetriever::applyModel(
    const std::vector<std::shared_ptr<Image>> &images) {
  std::vector<std::string> pathList;
  for (auto &i : images) {
    pathList.push_back(i->path);
  }
  return applyModel(pathList);
}
