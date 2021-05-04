#ifndef TORCHREIDRETRIEVER_H
#define TORCHREIDRETRIEVER_H

#include <database/database.h>

#include <QString>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include "ppbafloc-retrieval_export.h"
#include "types/image.h"

class PPBAFLOC_RETRIEVAL_EXPORT TorchreidRetriever {
 public:
  // Constructors
  /**
   * @brief This constructor shouldn't be used
   */
  TorchreidRetriever();
  /**
   * @brief This constructor should be used normally
   * @param modelpath path to *.onnx CNN model
   * @param inputformat size that images will be rescaled to; usually 224*224
   */
  TorchreidRetriever(const QString &modelpath,      // where to find the NN
                     const cv::Size &inputformat);  // of the NN
  /**
   * @brief Use this constructor if your are using a DB
   * @param modelpath path to *.onnx CNN model
   * @param inputformat size that images will be rescaled to; usually 224*224
   * @param db reference to database to be used
   */
  TorchreidRetriever(const QString &modelpath, const cv::Size &inputformat,
                     Database *db);
  /**
   * @brief Use this constructor if you are not using a DB and galleryImgs will
   * be empty in findReferenceImagesMultipleQueries
   * @param modelpath path to *.onnx CNN model
   * @param inputformat size that images will be rescaled to; usually 224*224
   * @param galleryDirPath path to directory containing gallery images
   */
  TorchreidRetriever(const QString &modelpath, const cv::Size &inputformat,
                     const QString &galleryDirPath);  // mutual parent dir for
                                                      // your gallery images

  /**
   * @brief findReferenceImagesMultipleQueries Reference Image Retrieval with
   * CNN
   * @param maxReferenceCount max output size of reference images
   * @param maxGalleryCount max amount of used gallery images
   * @return reference images
   */
  std::vector<std::vector<std::shared_ptr<Image>>>
  findReferenceImagesMultipleQueries(
      const std::vector<std::shared_ptr<Image>> &queryImages,
      std::vector<std::shared_ptr<Image>> &galleryImgs,
      const uint64 maxReferenceCount, const int maxGalleryCount,
      const uint64 numThreads, const uint64 batchSize, bool useDatabase);
  /**
   * @brief fillDatabaseHashes pre calculation when using database. Should be
   * used before findReferenceImagesMultipleQueries is called for queryimages
   */
  void fillDatabaseHashes(int batchSize = 1);

 private:
  cv::dnn::Net mModel;
  cv::Size mInputFormat;
  Database *mDB = nullptr;
  QString mGalleryDirPath;

  // these functions call the model with different parameters
  /**
   *@brief applies model to input
   * @param image input matrix
   * @return output matrix of CNN model
   */
  cv::Mat applyModel(const std::shared_ptr<Image> &image);

  /**
   * @brief applies model to input
   * @param image input matrix
   * @return output matrix of CNN model
   */
  cv::Mat applyModel(const cv::Mat &image);

  /**
   * @brief applies model to input
   * @param imagePath path to input image to feed to CNN
   * @return output matrix of CNN model
   */
  cv::Mat applyModel(const std::string &imagePath);

  /**
   * @brief applies model to input
   * @param images list of image matrices to feed to CNN
   * @return list of output matrices of CNN model
   */
  std::vector<cv::Mat> applyModel(const std::vector<cv::Mat> &images);

  /**
   * @brief applies model to input
   * @param list of image paths to feed to CNN
   * @return list of output matrices of CNN model
   */
  std::vector<cv::Mat> applyModel(const std::vector<std::string> &imagePaths);

  /**
   * @brief applies model to input
   * @param list of images to feed to CNN
   * @return list of output matrices of CNN model
   */
  std::vector<cv::Mat> applyModel(
      const std::vector<std::shared_ptr<Image>> &images);
};

#endif  // TORCHREIDRETRIEVER_H
