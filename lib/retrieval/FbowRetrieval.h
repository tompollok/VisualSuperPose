
#ifndef PPBAFLOC_FBOWRETRIEVAL_H
#define PPBAFLOC_FBOWRETRIEVAL_H

#include <database/DBHelper.h>
#include <database/database.h>
#include <types/image.h>

#include <opencv2/core.hpp>
#include <string>

#include "ppbafloc-retrieval_export.h"

class PPBAFLOC_RETRIEVAL_EXPORT FbowRetrieval {
 public:
  // Constructors
  FbowRetrieval() = default;
  /**
   * @brief FbowRetrieval when doing Retrieval
   * @param vocabPath String for path do vocabulary that is going to be used
   */
  explicit FbowRetrieval(const std::string &vocabPath) {
    setVocabPath(vocabPath);
  }
  /**
   * @brief FbowRetrieval when doing Training
   * @param vocabPath String for path do vocabulary that is going to be used
   * @param trainingDirPath Path to directory containing images used for
   * training of vocabulary
   * @param galleryDirPath Path to directory containing images used searching
   * for reference images
   */
  FbowRetrieval(const std::string &vocabPath,
                const std::string &trainingDirPath,
                const std::string &galleryDirPath);
  /**
   * @brief FbowRetrieval when doing Training with DB or Retrieval with DB
   * @param vocabPath String for path do vocabulary that is going to be used
   * @param trainingDirPath Path to directory containing images used for
   * training of vocabulary
   * @param galleryDirPath Path to directory containing images used searching
   * for reference images
   * @param db reference to database to read from
   */
  FbowRetrieval(const std::string &vocabPath,
                const std::string &trainingDirPath,
                const std::string &galleryDirPath, Database *db);

  /**
   * Set path to csv file to filter training images (!ONLY FOR GOOGLE LANDMARKS)
   * @param filterFile path to csv file
   */
  void setVocabCreationFilterFile(const std::string &filterFile);
  void setVocabPath(const std::string &vocabPath);

  /**
   * @brief K-Means "Training" and Vocabulary creation.
   * Parameters need to be set hardcoded in this function.
   */
  void createFbowVocabulary();
  /**
   * @brief fillDBFbow Gallery Precalculation to fill DB before Retrieval
   */
  void fillDBFbow(int nThreads = 1);
  /**
   * @brief retrieveImagesDB Retrieval with DB-gallery.
   * @param queries list of pointers to Images representing query images
   * @param outRetrievedPerQuery list to write result reference images to
   * @param numRetrieved number of images to retrieve for each query image
   */
  void retrieveImagesDB(
      const std::vector<std::shared_ptr<Image>> &queries,
      std::vector<std::vector<std::shared_ptr<Image>>> &outRetrievedPerQuery,
      int numRetrieved);
  /**
   * @brief retrieveImagesGalleryDir Retrieval with loose images.
   * @param queries list of pointers to Images representing query images
   * @param outRetrievedPerQuery to write result reference images to
   * @param maxNumberGalleryImages how many gallery images to compare (max)
   * @param numRetrieved number of images to retrieve for each query image
   * @param numThreads number of threads to use
   */
  void retrieveImagesGalleryDir(
      const std::vector<std::shared_ptr<Image>> &queries,
      std::vector<std::vector<std::shared_ptr<Image>>> &outRetrievedPerQuery,
      int maxNumberGalleryImages, int numRetrieved,
      unsigned int numThreads = 1);
  /**
   * @brief retrieveImagesGalleryDir Retrieval with gallery images.
   * @param queries list of pointers to Images representing query images
   * @param outRetrievedPerQuery to write result reference images to
   * @param galleryImgs list of images to search in
   * @param maxNumberGalleryImages how many gallery images to compare (max)
   * @param numRetrieved number of images to retrieve for each query image
   * @param numThreads number of threads to use
   */
  void retrieveImagesFromList(
      const std::vector<std::shared_ptr<Image>> &queries,
      const std::vector<std::shared_ptr<Image>> &galleryImgs,
      std::vector<std::vector<std::shared_ptr<Image>>> &outRetrievedPerQuery,
      int maxNumberGalleryImages, int numRetrieved,
      unsigned int numThreads = 1);

 private:
  std::string mVocabPath;        // FBoW
  std::string mGalleryDirPath;   // Possible Reference Images
  std::string mTrainingDirPath;  // Train Images
  std::string mCleanCSV;         // Google Landmarks train_clean.csv
  bool mVocabExists;
  Database *mDB = nullptr;

 private:
  /**
   * @brief retrieves the referenceframes for
   * @param queries images from
   * @param galleryImgs and returns the results in
   * @param outRetrievedPerQuery.
   * @param maxNumGalleryImgs max amount of galleryimages considered.
   * @param numRetrieved Size of outRetrievedPerQuery
   */
  void retrieve(
      const std::vector<std::shared_ptr<Image>> &queries,
      const std::vector<std::shared_ptr<Image>> &galleryImgs,
      std::vector<std::vector<std::shared_ptr<Image>>> &outRetrievedPerQuery,
      int maxNumGalleryImgs, int numRetrieved, unsigned int numThreads,
      bool useDB);
  /**
   * @brief isImageClean: Checks if img_path is mentioned in the google
   * landmarks clean csv. !ONLY FOR GOOGLE LANDMARKS V2!
   */
  bool isImageClean(const std::string img_path, const std::string &filterFile);
  /**
   * @brief getFilteredImages: Finds all the Google Landmarks Images in
   * mTrainingDirPath and returns only those mentioned in the train_clean.csv
   */
  void getFilteredImages(std::vector<std::string> &imageFiles,
                         const std::string &filterFile);
};

#endif  // PPBAFLOC_FBOWRETRIEVAL_H
