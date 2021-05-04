#include <FbowRetrieval.h>
#include <database/DBImporterMT.h>
#include <database/database.h>
#include <import/colmapimporter.h>
#include <registration.h>
#include <torchreidretriever.h>
#include <types/image.h>
#include <utils/Evaluator.h>
#include <utils/csvHelper.h>
#include <utils/iohelpers.h>

#include <QDir>
#include <QDirIterator>
#include <QFileInfo>
#include <QString>
#include <QTextStream>
#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

#include "utils/SiftHelpers.h"

/**
 *
 * README - Usage of parameters
 *
 * !All paths are used and expected as abolute paths!
 *
 * if use_single_dir:
 *          gallery_directory must point to images (e.g.
 *./../colmap/Alamo/images/, where images contains *.jpg)
 *          reconstruction_directory must point to text files for colmap (e.g.
 *./../colmap/Alamo/sparse or ./../colmap/Alamo/model or ....)
 *
 * if not use_single_dir:
 *          gallery_directory must contain the colmap subdirectories (doesn't
 *matter how many) then the structure has to be as follows (with Alamo as
 *example): gallery_directory/Alamo/images/*.jpg
 *              gallery_directory/Alamo/model/*.txt
 *
 * if fill_database:
 *          database is filled with images
 *          If use_single_dir --> only that one directory is added to db. It
 *will check whether the image path is already in the Database. If it is, the
 *path will not be added. Those checks don't exist for the calculation of the
 *SIFT features and Fbow maps! Those will be newly calculacted each time. So
 *make sure that they are not calculated unnecessarily.
 *
 *if use_database:
 *          the provided database file will be used as "gallery images". That
 *means the image search will include all (oand only) the images that are
 *currently stored in the database. Make sure that the SIFT features and FBOW
 *          Maps were calculated before!
 *
 *if not use_database:
 *          if use_single_dir
 *              only the images from that one specified dir will be imported and
 *used for retrieval and registration.
 *
 *          if not use_single_dir
 *              same as stated at the top with not use single dir; that
 *directory structure will be expected; all the images from all the Cornell
 *datasets will be imported and used for retrieval and registration
 *
 * if evaluation:
 *          Do evaluation and print evaluation results.
 *
 *
 * if evaluate_google_retrieval:
 *      for retrieval: use_database will be set to false automatically; the
 *gallery_dir will be recursively searched for image files and the retrieval
 *will be executed online on all of those image files. Fill_database will be set
 *to false, too. Also: the path to train_clean_csv *must be provided* for
 *groundtruth comparison.
 *
 * for CNN Retrieval usage:
 * set use_cnn_retrieval to true
 *      you must provide retrieval_net_path then!
 *
 *int parameter retrieval_net_batch:
 *      how many images to feed to the CNN at once. Adapt this to your RAM size
 *(GPU Size if opencv for CUDA is available)
 *
 * if evaluate_cnn_dir is set:
 *      set fill_database and use_database to false since we expect to evaluate
 *on Google Landmarks Dataset evaluate_google_retrieval and use_cnn_retrieval
 *will be set to true The directory provided here is expected to contain all the
 *.onnx model files that are to be evaluated registration will not be executed!
 *(Since we don't have any poses for the Google Landmarks Dataset) Right now,
 *only the models and their scores will be printed; *RESULT IMAGES ARE NOT SAVED
 *RIGHT NOW; THIS STILL NEEDS TO BE IMPLEMENTED*
 *
 * ATTENTION:
 *  for parameter combination:
 *      use_single_dir and not use_database:
 *          in lines 414-420 extrinsics and intrinsics are loaded. For
 *use_single_dir and not use_databse only the images from the specified
 *directory will be used for searching for extrinsics/intrinsics.
 *
 */
namespace {
/**
 * @brief showRetrievalResults displays retrieved images.
 * @param isGoodResult: vector of ref.size() that declares the reference images
 * as true positive or false positive.
 */
void showRetrievalResults(
    const std::shared_ptr<Image> &query,
    const std::vector<std::shared_ptr<Image>> &ref,
    const std::vector<bool> &isGoodResult = std::vector<bool>()) {
  cv::Mat img;
  cv::Mat imgc = cv::imread(query->path, cv::IMREAD_COLOR);
  double b = static_cast<double>(imgc.rows) / static_cast<double>(imgc.cols);
  bool useEvalResults = isGoodResult.size() == ref.size();
  cv::resize(imgc, imgc, cv::Size(300, static_cast<int>(b * 300)));
  cv::Mat temp = imgc;
  for (size_t i = 0; i < ref.size(); i++) {
    imgc = cv::imread(ref[i]->path, cv::IMREAD_COLOR);
    if ((i + 1) % 6 == 0) {
      if (i < 6) {
        img = temp;
      } else {
        b = static_cast<double>(img.cols) / static_cast<double>(temp.cols);
        cv::resize(temp, temp,
                   cv::Size(img.cols, static_cast<int>(temp.rows * b)));
        cv::vconcat(img, temp, img);
      }
      double b =
          static_cast<double>(imgc.rows) / static_cast<double>(imgc.cols);
      cv::resize(imgc, imgc, cv::Size(300, static_cast<int>(b * 300)));
      if (useEvalResults) {
        if (isGoodResult.at(i)) {
          cv::rectangle(imgc, cv::Size(0, 0), cv::Size(imgc.cols, imgc.rows),
                        cv::Scalar(0, 255, 0), 10);
        } else {
          cv::rectangle(imgc, cv::Size(0, 0), cv::Size(imgc.cols, imgc.rows),
                        cv::Scalar(0, 0, 255), 10);
        }
      }
      temp = imgc;
    } else {
      double a =
          static_cast<double>(temp.rows) / static_cast<double>(imgc.rows);
      cv::resize(imgc, imgc,
                 cv::Size(static_cast<int>(imgc.cols * a), temp.rows));
      if (useEvalResults) {
        if (isGoodResult.at(i)) {
          cv::rectangle(imgc, cv::Size(0, 0), cv::Size(imgc.cols, imgc.rows),
                        cv::Scalar(0, 255, 0), 10);
        } else {
          cv::rectangle(imgc, cv::Size(0, 0), cv::Size(imgc.cols, imgc.rows),
                        cv::Scalar(0, 0, 255), 10);
        }
      }
      cv::hconcat(temp, imgc, temp);
    }
  }
  if (img.empty())
    img = temp;
  else {
    double a = static_cast<double>(temp.rows) / static_cast<double>(imgc.rows);
    cv::resize(imgc, imgc,
               cv::Size(static_cast<int>(imgc.cols * a), temp.rows));
    cv::hconcat(temp, imgc, temp);
  }
  cv::imshow("Retrieval results", img);
  cv::waitKey();
}
}  // namespace
/**
 * @brief The AppSettings struct for loading the app settings from *.yml
 */
struct AppSettings {
  bool load(std::string configPath) {
    cv::FileStorage fs(configPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
      return false;
    }

    trainingDirPath = QString::fromStdString(fs["train_directory"]);
    galleryDirPath = QString::fromStdString(fs["gallery_directory"]);
    reconstructionDirPath =
        QString::fromStdString(fs["reconstruction_directory"]);
    queryImagePath = QString::fromStdString(fs["query_image"]);
    vocabFilePath = QString::fromStdString(fs["vocab_file"]);

    auto node = fs["query_image"];
    if (!node.isNone()) {
      queryImagePath = QString::fromStdString(node);
    }

    node = fs["query_image_list"];
    if (!node.isNone()) {
      queryImageList = QString::fromStdString(node);
    }

    node = fs["query_image_list_prefix"];
    if (!node.isNone()) {
      queryImageListPrefix = QString::fromStdString(node);
    }

    node = fs["result_dir"];
    if (!node.isNone()) {
      resultDirPath = QString::fromStdString(node);
    }

    node = fs["database_path"];
    if (!node.isNone()) {
      databasePath = QString::fromStdString(node);
    }

    node = fs["train_clean_csv"];
    if (!node.isNone()) {
      trainCleanCSV = QString::fromStdString(node);
    }
    node = fs["retrieval_net_path"];
    if (!node.isNone()) {
      retrievalNetPath = QString::fromStdString(node);
    }
    node = fs["retrieval_net_batch"];
    if (!node.isNone()) {
      retrievalNetBatch = node;
    }
    node = fs["retrieve_images_num"];
    if (!node.isNone()) {
      retrieveImages = node;
    }

    node = fs["max_num_gallery_images"];
    if (!node.isNone()) {
      maxNumGalleryImages = node;
    }

    node = fs["write_match_pairs_file"];
    if (!node.isNone()) {
      matchPairsFile = QString::fromStdString(node);
    }

    node = fs["num_threads"];
    if (!node.isNone()) {
      numThreads = node;
    }

    node = fs["display_images"];
    if (!node.isNone()) {
      displayImages = static_cast<int>(node);
    }

    node = fs["filter_images"];
    if (!node.isNone()) {
      /*
      QString filterImagesString = QString::fromStdString(node);
      if (filterImagesString.contains("true", Qt::CaseInsensitive)) {
          filterImages = true;
      } else {
          filterImages = false;
      }
       */
      filterImages = static_cast<int>(node);
    }
    node = fs["use_database"];
    if (!node.isNone()) {
      useDatabase = static_cast<int>(node);
    }
    node = fs["fill_database"];
    if (!node.isNone()) {
      fillDatabase = static_cast<int>(node);
    }
    node = fs["use_single_dir"];
    if (!node.isNone()) {
      useSingleDir = static_cast<int>(node);
    }
    node = fs["evaluation"];
    if (!node.isNone()) {
      evaluation = static_cast<int>(node);
    }
    node = fs["use_multiple_models"];
    if (!node.isNone()) {
      useMultipleModels = static_cast<int>(node);
    }
    node = fs["use_cnn_retrieval"];
    if (!node.isNone()) {
      useCNNRetrieval = static_cast<int>(node);
    }
    node = fs["do_registration"];
    if (!node.isNone()) {
      doRegistration = static_cast<int>(node);
    }

    node = fs["model_prefix"];
    if (!node.isNone()) {
      cnnModelPrefix = QString::fromStdString(node);
    }

    node = fs["use_superglue"];
    if (!node.isNone()) {
      useSuperglue = static_cast<int>(node);
    }
    node = fs["superpoint_model"];
    if (!node.isNone()) {
      superpointModel = QString::fromStdString(node);
    }
    node = fs["superglue_model"];
    if (!node.isNone()) {
      superglueModel = QString::fromStdString(node);
    }
    node = fs["superpoint_resize_width"];
    if (!node.isNone()) {
      superpoint_resize_width = static_cast<int>(node);
    }

    node = fs["evaluate_google_retrieval"];
    if (!node.isNone()) {
      evaluateGoogleRetrieval = static_cast<int>(node);
      if (evaluateGoogleRetrieval) {
        useDatabase = false;
        useSingleDir = false;
      }
    }

    if (databasePath.isEmpty()) {
      useDatabase = false;
      fillDatabase = false;
    }

    if (trainCleanCSV.isEmpty()) {
      filterImages = false;
    }

    if (resultDirPath.isEmpty()) {
      matchPairsFile = "";
    }

    node = fs["evaluate_cnn_dir"];
    if (!node.isNone()) {
      evaluateCNNDir = QString::fromStdString(node);
      /*
      fillDatabase = false;
      useDatabase = false;
      useCNNRetrieval = true;*/
      // evaluateGoogleRetrieval = true;
    }

    node = fs["evaluate_colmap_retrieval"];
    if (!node.isNone()) {
      colmapRetrievalEvaluation = static_cast<int>(node);
    }

    node = fs["save_csv_evaluation_dir"];
    if (!node.isNone()) {
      saveCsvEvaluationDir = QString::fromStdString(node);
    }

    node = fs["evaluate_both_registrations"];
    if (!node.isNone()) {
      evaluateBothRegistrations = static_cast<int>(node);
      if (evaluateBothRegistrations) {
        evaluation = true;
        useSuperglue = true;
      }
    }

    return true;
  }

  void printSettings() const {
    std::cout
        << "Settings:\n"
        << "    Training Dir: " << trainingDirPath.toStdString() << std::endl
        << "    Gallery Dir: " << galleryDirPath.toStdString() << std::endl
        << "    Reconstruction Dir: " << reconstructionDirPath.toStdString()
        << std::endl
        << "    Query Image: " << queryImagePath.toStdString() << std::endl
        << "    Query Image List: "
        << ((queryImageList.isEmpty()) ? "not set"
                                       : queryImageList.toStdString())
        << std::endl
        << "    Query Image List Prefix: "
        << ((queryImageListPrefix.isEmpty())
                ? "not set"
                : queryImageListPrefix.toStdString())
        << std::endl
        << "    Vocabulary File: " << vocabFilePath.toStdString() << std::endl
        << "    Retrieval CNN Model path: "
        << ((retrievalNetPath.isEmpty()) ? "not set"
                                         : retrievalNetPath.toStdString())
        << std::endl
        << "    Retrieval CNN Batch size: " << retrievalNetBatch << std::endl
        << "    CNN Model Directory for Google Landmarks Evaluation: "
        << ((evaluateCNNDir.isEmpty()) ? "not set"
                                       : evaluateCNNDir.toStdString())
        << std::endl
        << "    Result Dir: "
        << ((resultDirPath.isEmpty()) ? "not set" : resultDirPath.toStdString())
        << std::endl
        << "    Database path: "
        << ((databasePath.isEmpty()) ? "not set" : databasePath.toStdString())
        << std::endl
        << "    Use Database: " << (useDatabase ? "yes" : "no") << std::endl
        << "    Fill Database: " << (fillDatabase ? "yes" : "no") << std::endl
        << "    Use single dir for Database: " << (useSingleDir ? "yes" : "no")
        << std::endl
        << "    Train Clean csv: "
        << ((trainCleanCSV.isEmpty()) ? "not set" : trainCleanCSV.toStdString())
        << std::endl
        << "    Filter Images: " << (filterImages ? "yes" : "no") << std::endl
        << "    Gallery Images Cap: "
        << ((maxNumGalleryImages > 0) ? std::to_string(maxNumGalleryImages)
                                      : "None")
        << std::endl
        << "    Number of Images to retrieve: " << retrieveImages << std::endl
        << "    #Threads: " << numThreads << std::endl
        << "    Display Images: " << (displayImages ? "yes" : "no") << std::endl
        << "    Write Match Pairs file: "
        << (matchPairsFile.isEmpty() ? "no" : matchPairsFile.toStdString())
        << std::endl
        << "    Use SuperGlue Matching: " << (useSuperglue ? "yes" : "no")
        << std::endl
        << "    SuperPoint resize image width: "
        << (superpoint_resize_width > 0
                ? std::to_string(superpoint_resize_width)
                : "original")
        << std::endl
        << "    Evaluation: " << (evaluation ? "yes" : "no") << std::endl
        << "    Evaluate Google Retrieval: "
        << (evaluateGoogleRetrieval ? "yes" : "no") << std::endl
        << "    Evaluate Colmap Retrieval: "
        << (colmapRetrievalEvaluation ? "yes" : "no") << std::endl
        << "    Save Evaluation CSV Dir: "
        << ((saveCsvEvaluationDir.isEmpty())
                ? "not set"
                : saveCsvEvaluationDir.toStdString())
        << std::endl
        << "    Registration: "
        << (doRegistration ? (evaluateBothRegistrations
                                  ? "SuperGlue + Classic"
                                  : (useSuperglue ? "SuperGlue" : "Classic"))
                           : "none")
        << std::endl;
  }

  QString evaluationReportFileName() const {
    QString retrievalType, registrationtype, dataset;
    if (useCNNRetrieval) {
      QString epochs = "Epoch" + retrievalNetPath.split("/")
                                     .back()
                                     .split(".", QString::SkipEmptyParts)
                                     .first()
                                     .split("-")
                                     .last();
      QString prefix = cnnModelPrefix;
      QStringList l = retrievalNetPath.split("/");
      QString model = l.at(l.size() - 2);  // should be ResNet50 or ResNet18
      retrievalType = model + "_" + prefix + "_" + epochs;
    } else
      retrievalType = "FBoW";
    if (!doRegistration)
      registrationtype = "NaN";
    else if (useSuperglue)
      registrationtype = "Superglue";
    else
      registrationtype = "Classic";
    if (evaluateGoogleRetrieval)
      dataset = "GoogleLandmarks";
    else
      dataset = "Cornell";
    return IOHelpers::appendSlash(saveCsvEvaluationDir) + retrievalType + "_" +
           registrationtype + "_" + dataset + ".csv";
  }

  QString trainingDirPath;
  QString galleryDirPath;
  QString reconstructionDirPath;
  QString queryImagePath;
  QString queryImageList;
  QString queryImageListPrefix;
  QString vocabFilePath;
  QString resultDirPath;
  QString trainCleanCSV;
  QString databasePath;
  QString matchPairsFile;
  QString superpointModel = "SuperPoint.zip";
  QString superglueModel = "SuperGlue.zip";
  int superpoint_resize_width = -1;
  QString retrievalNetPath;
  QString evaluateCNNDir;
  QString saveCsvEvaluationDir;
  QString cnnModelPrefix =
      "unknown";  // should be "small" or "large" for traindata size
  int maxNumGalleryImages = -1;
  int numThreads = 1;
  int retrieveImages = 20;
  int retrievalNetBatch = 1;
  bool displayImages = false;
  bool filterImages = true;
  bool useDatabase = false;
  bool fillDatabase = false;
  bool useSingleDir = true;
  bool useSuperglue = false;
  bool evaluation = false;
  bool evaluateGoogleRetrieval = false;
  bool useMultipleModels = false;
  bool useCNNRetrieval = false;
  bool colmapRetrievalEvaluation = false;
  bool doRegistration = false;
  bool evaluateBothRegistrations = false;
};

/**
 * @brief fillDatabaseMain precalculations for gallery so the pipeline runs fast
 * without overfilling RAM
 */

void fillDatabaseMain(Database &db, const AppSettings &settings,
                      FbowRetrieval &fbowRetr, TorchreidRetriever &torchRetr) {
  auto start = std::chrono::high_resolution_clock::now();

  if (settings.numThreads > 1) {
    std::cout << "FillDatabase: Importing reconstruction and calculating SIFT"
              << std::endl;
    DBImporterMT dbimporter(db, settings.numThreads - 1);
    if (settings.useSingleDir) {
      dbimporter.importSingleReconstruction(settings.galleryDirPath,
                                            settings.reconstructionDirPath);
    } else {
      dbimporter.importRecursive(settings.galleryDirPath);
    }
    auto tSIFT = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed: "
              << std::chrono::duration<double>(tSIFT - start).count() << "s"
              << std::endl;
  } else {
    DBHelper dbhelper = DBHelper(db);
    std::cout << "FillDatabase: importing images" << std::endl;
    if (settings.useSingleDir) {
      dbhelper.fillDatabaseExtrIntr(
          settings.galleryDirPath.toStdString(),
          settings.reconstructionDirPath.toStdString());
    } else {
      dbhelper.fillDatabaseExtrIntr(settings.galleryDirPath.toStdString());
    }
    auto tImport = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed: "
              << std::chrono::duration<double>(tImport - start).count() << "s"
              << std::endl;
    std::cout << "FillDatabase: calculating sift" << std::endl;
    dbhelper.fillDatabaseSIFT();
    auto tSIFT = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed: "
              << std::chrono::duration<double>(tSIFT - tImport).count() << "s"
              << std::endl;
  }

  auto t1 = std::chrono::high_resolution_clock::now();

  if (settings.useCNNRetrieval) {
    std::cout << "Fill Database: calculating CNN hash" << std::endl;
    torchRetr.fillDatabaseHashes(settings.retrievalNetBatch);
  }
  auto tHash = std::chrono::high_resolution_clock::now();
  if (settings.useCNNRetrieval) {
    std::cout << "Elapsed Hashing: "
              << std::chrono::duration<double>(tHash - t1).count() << "s"
              << std::endl;
  }

  if (!settings.vocabFilePath.isEmpty()) {
    std::cout << "FillDatabase: calculating fbow" << std::endl;
    fbowRetr.fillDBFbow(settings.numThreads);
  }
  auto tBow = std::chrono::high_resolution_clock::now();
  std::cout << "Elapsed FBoW: "
            << std::chrono::duration<double>(tBow - tHash).count() << "s"
            << std::endl;
  std::cout
      << "Elapsed time filling database: "
      << std::chrono::duration_cast<std::chrono::minutes>(tBow - start).count()
      << " min\n";
}

/**
 * @brief showAndSaveRetrieved for visualizing the retrieval results
 */
void showAndSaveRetrieved(
    const AppSettings &settings,
    const std::vector<std::shared_ptr<Image>> &retrievedImages,
    std::shared_ptr<Image> &queryImage, bool createSubfolder,
    std::vector<bool> isGoodRefFrame = std::vector<bool>()) {
  std::cout << __FUNCTION__ << std::endl;
  if (settings.displayImages) {
    showRetrievalResults(queryImage, retrievedImages, isGoodRefFrame);
  }
  std::cout << __FUNCTION__ << "1" << std::endl;
  if (!settings.resultDirPath.isEmpty()) {
    QString saveDir = IOHelpers::appendSlash(settings.resultDirPath);
    if (createSubfolder) {
      saveDir +=
          IOHelpers::filenameWithoutExtension(IOHelpers::getFilenameFromPath(
              QString::fromStdString(queryImage->path)));
    }

    QDir d(saveDir);
    d.mkpath(saveDir);

    cv::Mat qImg = imread(queryImage->path, cv::IMREAD_COLOR);
    cv::imwrite(
        QString(IOHelpers::appendSlash(saveDir) + "query.jpg").toStdString(),
        qImg);

    for (size_t i = 0; i < retrievedImages.size(); i++) {
      cv::Mat img = imread(retrievedImages[i]->path, cv::IMREAD_COLOR);
      cv::resize(img, img, cv::Size(img.cols * 0.4, img.rows * 0.4), 0, 0);
      std::string imgName = "retrieved_image_" + std::to_string(i) + ".jpg";
      cv::imwrite(IOHelpers::appendSlash(saveDir).toStdString() + imgName, img);
    }

    if (!settings.matchPairsFile.isEmpty()) {
      QFile imgListFile(IOHelpers::appendSlash(saveDir) +
                        settings.matchPairsFile);
      imgListFile.open(QFile::WriteOnly);
      QTextStream ts(&imgListFile);

      for (size_t i = 0; i < retrievedImages.size(); i++) {
        if (retrievedImages[i]->path.compare(queryImage->path) != 0) {
          ts << "query.jpg"
             << " " << QString("retrieved_image_%1.jpg").arg(i) << endl;
        }
      }
    }

    // Save images.txt (for visualizer)
    {
      QFile imgListFile(IOHelpers::appendSlash(saveDir) + "images.txt");
      imgListFile.open(QFile::WriteOnly);
      QTextStream ts(&imgListFile);

      ts << QString::fromStdString(queryImage->path) << endl;

      for (size_t i = 0; i < retrievedImages.size(); i++) {
        if (retrievedImages[i]->path.compare(queryImage->path) != 0) {
          ts << QString::fromStdString(retrievedImages[i]->path) << endl;
        }
      }
    }
  }
  std::cout << __FUNCTION__ << "2" << std::endl;
}
/**
 * @brief retrieveMultipleCNNModels CNN retrieval with more than one Model. (For
 * evaluation) (should not be used this way right now! Only one model please!
 * The function is all right, but how it is used is not.)
 */
void retrieveMultipleCNNModels(
    const AppSettings &settings, std::vector<QString> &modelPaths,
    std::vector<std::shared_ptr<Image>> &galleryImages,
    std::vector<std::shared_ptr<Image>> &queryImages,
    std::vector<std::vector<std::shared_ptr<Image>>> &outRetrievedImages) {
  std::vector<std::vector<std::shared_ptr<Image>>> retrievedImages;
  TorchreidRetriever retriever;
  std::vector<
      std::pair<std::string, std::vector<std::vector<std::shared_ptr<Image>>>>>
      modelImagePairs;

  if (modelPaths.empty()) {
    QStringList filter;
    filter << "*.onnx";
    QDirIterator it(settings.evaluateCNNDir, filter, QDir::Files,
                    QDirIterator::Subdirectories);
    for (int i = 0; it.hasNext(); ++i) {
      QString modelPath = it.next();
      modelPaths.push_back(modelPath);
    }
  }

  for (const auto &path : modelPaths) {
    std::cout << "model is " << path.toStdString() << std::endl;
    retriever =
        TorchreidRetriever(path, cv::Size(224, 224), settings.galleryDirPath);
    retrievedImages = retriever.findReferenceImagesMultipleQueries(
        queryImages, galleryImages, settings.retrieveImages,
        settings.maxNumGalleryImages, settings.numThreads,
        settings.retrievalNetBatch, settings.useDatabase);
    outRetrievedImages = retrievedImages;
    modelImagePairs.push_back({path.toStdString(), retrievedImages});
  }
}

void calculateSIFT(std::shared_ptr<Image> &img) {
  SiftHelpers::extractSiftFeatures(img->path, img->siftDescriptors,
                                   img->siftKeypoints);
}
/**
 * @brief loadSIFT from DB or calculate SIFT if it is not possible to load.
 */
void loadSIFT(const AppSettings &settings, std::unique_ptr<Database> &db,
              std::vector<std::shared_ptr<Image>> &images) {
  if (settings.useDatabase) {
    DBHelper dbhelper = DBHelper(*db.get());
    for (auto &img : images) {
      if (!dbhelper.getImageByPath(img->path, img)) {
        calculateSIFT(img);
      }
    }
  } else {
    for (auto &img : images) {
      calculateSIFT(img);
    }
  }
}
/**
 * @brief loadExtrinsicsFromList: function to fix wrong usage of Image in
 * combination with DB.
 */
bool loadExtrinsicsFromList(std::shared_ptr<Image> &target,
                            const std::vector<std::shared_ptr<Image>> &source) {
  for (auto &s : source) {
    if (target->path.compare(s->path) == 0) {
      target->extrinsics = s->extrinsics;
      target->intrinsics = s->intrinsics;
      return true;
    }
  }
  return false;
}

bool loadQueryImageList(AppSettings &settings,
                        std::vector<std::shared_ptr<Image>> &queryImages) {
  QFile f(settings.queryImageList);
  if (!f.open(QFile::ReadOnly | QFile::Text)) {
    return false;
  }

  QTextStream ts(&f);
  while (!ts.atEnd()) {
    QString line = ts.readLine().trimmed();
    if (line.isEmpty()) {
      continue;
    }

    QString queryFile = settings.queryImageListPrefix + line;
    if (!QFileInfo::exists(queryFile)) {
      std::cout << "Query Image does not exist! Skipping. \""
                << queryFile.toStdString() << "\"" << std::endl;
      continue;
    }

    auto queryImage = std::make_shared<Image>();
    queryImage->path = queryFile.toStdString();
    queryImages.push_back(queryImage);
  }
  return true;
}
/**
 * @brief runPipeline this function runs the whole pipeline with or without
 * evaluation.
 */
void runPipeline(AppSettings &settings, std::unique_ptr<Database> &db,
                 std::vector<std::shared_ptr<Image>> queryImages,
                 std::vector<std::shared_ptr<Image>> galleryImages) {
  auto start = std::chrono::high_resolution_clock::now();
  //
  // =======================================
  // LOAD Retrieval
  //
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "      LOAD RETRIEVAL \n";
  std::cout << "--------------------------------------------" << std::endl;

  TorchreidRetriever cnnRetriever;
  FbowRetrieval fbowInstance;
  if (settings.useCNNRetrieval) {
    if (settings.useDatabase) {
      cnnRetriever = TorchreidRetriever(settings.retrievalNetPath,
                                        cv::Size(224, 224), db.get());
    } else {
      if (settings.retrievalNetPath.isEmpty()) {
        QStringList filter;
        filter << "*.onnx";
        QDirIterator it(settings.evaluateCNNDir, filter, QDir::Files,
                        QDirIterator::Subdirectories);
        QString modelPath = it.next();
        cnnRetriever = TorchreidRetriever(modelPath, cv::Size(224, 224),
                                          settings.galleryDirPath);
      } else {
        cnnRetriever =
            TorchreidRetriever(settings.retrievalNetPath, cv::Size(224, 224),
                               settings.galleryDirPath);
      }
    }
  }

  if (!settings.useCNNRetrieval ||
      (settings.fillDatabase && !settings.vocabFilePath.isEmpty())) {
    if (settings.useDatabase) {
      fbowInstance =
          FbowRetrieval(settings.vocabFilePath.toStdString(),
                        settings.trainingDirPath.toStdString(),
                        settings.galleryDirPath.toStdString(), db.get());
    } else {
      fbowInstance = FbowRetrieval(settings.vocabFilePath.toStdString(),
                                   settings.trainingDirPath.toStdString(),
                                   settings.galleryDirPath.toStdString());
    }

    if (settings.filterImages) {
      fbowInstance.setVocabCreationFilterFile(
          settings.trainCleanCSV.toStdString());
    }

    if (!QFileInfo::exists(settings.vocabFilePath)) {
      std::cout << "creating vocabulary" << std::endl;
      fbowInstance.createFbowVocabulary();
    }
  }

  if (settings.fillDatabase && settings.useDatabase) {
    std::cout << "Filling Database " << std::endl;
    fillDatabaseMain(*db, settings, fbowInstance, cnnRetriever);
  }

  //
  // =======================================
  // Retrieval
  //
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "      DO RETRIEVAL \n";
  std::cout << "--------------------------------------------" << std::endl;
  std::vector<std::vector<std::shared_ptr<Image>>> retrievedImages(
      queryImages.size());
  std::vector<double> meanAPs;
  Evaluator eval = Evaluator();

  auto tRet0 = std::chrono::high_resolution_clock::now();
  if (settings.useCNNRetrieval) {
    if (settings.useDatabase) {
      retrievedImages = cnnRetriever.findReferenceImagesMultipleQueries(
          queryImages, galleryImages, settings.retrieveImages,
          settings.maxNumGalleryImages, settings.numThreads,
          settings.retrievalNetBatch, settings.useDatabase);

    } else {
      if (!settings.evaluateCNNDir.isEmpty() &&
          (settings.evaluateGoogleRetrieval ||
           settings.colmapRetrievalEvaluation)) {
        std::vector<QString> modelPathsMock(0);
        retrieveMultipleCNNModels(settings, modelPathsMock, galleryImages,
                                  queryImages, retrievedImages);
      } else {
        std::vector<QString> modelPaths = {settings.retrievalNetPath};
        retrieveMultipleCNNModels(settings, modelPaths, galleryImages,
                                  queryImages, retrievedImages);
      }
    }
  }

  else {
    std::cout << "------------------------------------------------------------"
              << std::endl;

    if (settings.useDatabase) {
      fbowInstance.retrieveImagesDB(queryImages, retrievedImages,
                                    settings.retrieveImages);
    } else if (settings.evaluateGoogleRetrieval) {
      fbowInstance.retrieveImagesGalleryDir(
          queryImages, retrievedImages, settings.maxNumGalleryImages,
          settings.retrieveImages, settings.numThreads);
    } else {
      fbowInstance.retrieveImagesFromList(
          queryImages, galleryImages, retrievedImages,
          settings.maxNumGalleryImages, settings.retrieveImages,
          settings.numThreads);
    }
  }
  auto tRet1 = std::chrono::high_resolution_clock::now();
  std::cout << "Elapsed Time Retrieval "
            << std::chrono::duration<double>(tRet1 - tRet0).count() << " s"
            << std::endl;

  //
  // =======================================
  // FIX IMAGE CREATION
  //
  if (!settings.evaluateGoogleRetrieval) {
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "      STUPID FIX FOR STUPID IMAGE CREATION \n";
    std::cout << "--------------------------------------------" << std::endl;

    QHash<QString, std::shared_ptr<Image>> hash;
    for (auto &img : galleryImages) {
      hash.insert(QString::fromStdString(img->path), img);
    }
    std::vector<std::future<void>> fts;
    for (auto &ret : retrievedImages) {
      fts.push_back(std::async(
          std::launch::async,
          [](std::vector<std::shared_ptr<Image>> &ret,
             QHash<QString, std::shared_ptr<Image>> &hash) {
            for (auto &img : ret) {
              if (hash.contains(QString::fromStdString(img->path)))
                img = hash.find(QString::fromStdString(img->path)).value();
            }
          },
          std::ref(ret), std::ref(hash)));
    }
    for (auto &ft : fts) {
      ft.get();
    }
  }

  //
  // =======================================
  // Evaluate Retrieval
  //
  std::unique_ptr<CSVHelper> mainEvalReport = nullptr;
  std::unique_ptr<CSVHelper> extraEvalReport = nullptr;

  if (settings.evaluation) {
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "      DO RETRIEVAL EVALUATION \n";
    std::cout << "--------------------------------------------" << std::endl;

    mainEvalReport = std::make_unique<CSVHelper>(
        settings.evaluationReportFileName().toStdString());

    if (settings.evaluateBothRegistrations) {
      QString filename =
          settings.evaluationReportFileName().replace("Superglue", "Classic");
      extraEvalReport = std::make_unique<CSVHelper>(filename.toStdString());
    }

    for (size_t i = 0; i < retrievedImages.size(); i++) {
      std::shared_ptr<Image> &queryImage = queryImages.at(i);
      if (settings.evaluateGoogleRetrieval) {
        std::cout << "eval google" << std::endl;
        std::vector<bool> isGoodReferenceFrame =
            eval.evaluateRetrievalLandmarks(
                settings.trainCleanCSV.toStdString(), retrievedImages.at(i),
                queryImage);
        mainEvalReport->saveRetrievalAndRegistration(queryImage);
        if (extraEvalReport) {
          extraEvalReport->saveRetrievalAndRegistration(queryImage);
        }
        showAndSaveRetrieved(settings, retrievedImages.at(i), queryImage,
                             queryImages.size() > 1, isGoodReferenceFrame);
      } else {  // cornell
        std::cout << "eval cornell" << std::endl;
        std::vector<bool> isGoodReferenceFrame =
            eval.evaluateRetrievalCornell(retrievedImages.at(i), queryImage);
        mainEvalReport->saveRetrievalAndRegistration(queryImage);
        if (extraEvalReport) {
          extraEvalReport->saveRetrievalAndRegistration(queryImage);
        }
        showAndSaveRetrieved(settings, retrievedImages.at(i), queryImage,
                             queryImages.size() > 1, isGoodReferenceFrame);
      }
    }
  }

  if (settings.doRegistration) {
    auto tRegStart = std::chrono::high_resolution_clock::now();

    //
    // =======================================
    // Registration
    //
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "      DO REGISTRATION \n";
    std::cout << "--------------------------------------------" << std::endl;

    Registration registration;
    if (settings.useSuperglue) {
      if (!registration.setupDeepLearningBasedPoseEstimation(
              settings.superpointModel.toStdString(),
              settings.superglueModel.toStdString(),
              settings.superpoint_resize_width)) {
        std::cout << "SuperGlue/SuperPoint setup failed!\nFalling back to "
                     "classic SIFT matching."
                  << std::endl;
        settings.useSuperglue = false;
        settings.evaluateBothRegistrations = false;
      }
    }

    for (size_t queryIndex = 0; queryIndex < queryImages.size(); ++queryIndex) {
      auto tq0 = std::chrono::high_resolution_clock::now();
      auto &queryImage = queryImages[queryIndex];
      std::cout
          << "============================================================"
          << std::endl;
      std::cout << "Registration of \"" << queryImage->path << "\""
                << std::endl;

      std::vector<std::shared_ptr<Image>> retrievedImagesWithExtrinsics;
      for (auto &retrieved : retrievedImages[queryIndex]) {
        if (retrieved->path.compare(queryImage->path) == 0) {
          std::cout << "Excluding query image from retrieved" << std::endl;
          continue;
        }

        if (loadExtrinsicsFromList(retrieved, galleryImages)) {
          retrievedImagesWithExtrinsics.push_back(retrieved);
        }
      }
      std::cout << "Reference Images: " << retrievedImagesWithExtrinsics.size()
                << std::endl;

      if (retrievedImagesWithExtrinsics.size() >= 2) {
        std::vector<std::shared_ptr<Image>> v = {queryImage};

        if (settings.evaluateBothRegistrations || !settings.useSuperglue) {
          loadSIFT(settings, db, v);
          loadSIFT(settings, db, retrievedImagesWithExtrinsics);
        }
        Extrinsics resultPose;

        if (settings.evaluateBothRegistrations) {
          std::cout << "----------------------------------------------"
                       "\nSuperGlue Registration"
                    << std::endl;
          std::vector<cv::Point3f> triangulated1;
          if (registration.applyDeepLearningBasedPoseEstimation(
                  settings.evaluation, queryImage,
                  retrievedImagesWithExtrinsics, resultPose, triangulated1))
            mainEvalReport->saveRegistration(queryImage);

          std::cout << "----------------------------------------------"
                       "\nClassic Registration"
                    << std::endl;
          std::vector<cv::Point3f> triangulated2;
          if (registration.applyClassicPoseEstimation(
                  settings.evaluation, queryImage,
                  retrievedImagesWithExtrinsics, resultPose, triangulated2))
            extraEvalReport->saveRegistration(queryImage);
        } else {
          std::vector<cv::Point3f> triangulated;
          bool success = false;
          if (settings.useSuperglue) {
            success = registration.applyDeepLearningBasedPoseEstimation(
                settings.evaluation, queryImage, retrievedImagesWithExtrinsics,
                resultPose, triangulated);
          } else {
            success = registration.applyClassicPoseEstimation(
                settings.evaluation, queryImage, retrievedImagesWithExtrinsics,
                resultPose, triangulated);
          }

          if (settings.evaluation && success) {
            mainEvalReport->saveRegistration(queryImage);
          }
        }
      } else {
        std::cout << "Too few reference images, Skipping." << std::endl;
      }
      for (auto &img : retrievedImagesWithExtrinsics) {
        img->forgetAll();
      }
      queryImage->forgetAll();

      auto tq1 = std::chrono::high_resolution_clock::now();
      std::cout << "Elapsed time Registration for Query: "
                << std::chrono::duration<double>(tq1 - tq0).count() << " s"
                << std::endl;
    }

    auto tRegEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = tRegEnd - tRegStart;
    std::cout << "Elapsed time Registration: " << elapsed.count() << " s"
              << std::endl;
  }

  if (settings.evaluation) {
    std::cout << "Saving Evaluation Report" << std::endl;
    mainEvalReport = nullptr;
    extraEvalReport = nullptr;
  }

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "============================================================"
            << std::endl;
  std::cout << "Elapsed time total process: " << elapsed.count() << " s"
            << std::endl;
}

int main(int argc, char *argv[]) {
  auto start = std::chrono::high_resolution_clock::now();
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "      DO PRE-CALCULATIONS & LOADING \n";
  std::cout << "--------------------------------------------" << std::endl;
  std::string configpath = "config.yml";

  if (argc > 1) {
    configpath = argv[1];
  }

  AppSettings settings;
  try {
    if (!settings.load(configpath)) {
      throw std::exception();
    }
  } catch (...) {
    std::cout << "Error parsing config file. Exiting" << std::endl;
    return 1;
  }

  settings.printSettings();

  std::unique_ptr<Database> db = nullptr;
  size_t imageNumDB = 0;
  if (settings.useDatabase) {
    auto t0 = std::chrono::high_resolution_clock::now();
    db = std::make_unique<Database>();
    db->createConnection(settings.databasePath);

    imageNumDB = db->getNumImages();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto d = std::chrono::duration<double>(t1 - t0);
    std::cout
        << "DB Loaded. Images: " << imageNumDB << " Time: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(d).count()
        << " ms" << std::endl;
  }

  std::vector<std::shared_ptr<Image>> queryImages;
  if (!settings.queryImageList.isEmpty()) {
    if (!loadQueryImageList(settings, queryImages)) {
      std::cout << "Error parsing query image list file" << std::endl;
    }
  } else if (!settings.queryImagePath.isEmpty()) {
    if (!QFileInfo::exists(settings.queryImagePath)) {
      std::cout << "Could not open file \""
                << settings.queryImagePath.toStdString() << "\"" << std::endl;
    } else {
      auto img = std::make_shared<Image>();
      img->path = settings.queryImagePath.toStdString();
      queryImages.push_back(img);
    }
  }

  if (queryImages.empty()) {
    std::cout << "No query images to work with. Extiting" << std::endl;
    return 1;
  } else {
    std::cout << queryImages.size() << " query images" << std::endl;
  }

  std::vector<std::shared_ptr<Image>> galleryImages;

  if (!settings.evaluateGoogleRetrieval &&
      !(settings.useDatabase && !settings.evaluation &&
        !settings.doRegistration)) {
    int64 t1 = cv::getTickCount();
    ColmapImporter importer;
    std::cout << "Importing image reconstruction(s)" << std::endl;
    if (settings.useSingleDir) {
      importer.importImages(settings.reconstructionDirPath,
                            settings.galleryDirPath, galleryImages);
    } else {
      importer.importImagesFromSubdirs(settings.galleryDirPath, galleryImages);
    }
    int64 t2 = cv::getTickCount();
    std::cout << "Import Finished. Found " << galleryImages.size()
              << " images in "
              << static_cast<double>(t2 - t1) / cv::getTickFrequency()
              << "seconds." << std::endl;

    for (auto it = queryImages.begin(); it != queryImages.end(); ++it) {
      bool found = false;
      for (auto &img : galleryImages) {
        if (img->path.compare((*it)->path) == 0) {
          *it = img;
          found = true;
          break;
        }
      }

      if (!found) {
        std::cout
            << "Query Image \"" << (*it)->path
            << "\" not found in reconstruction: No ground truth available."
            << std::endl;
      }
    }

    if (settings.evaluation) {
      int64 t3 = cv::getTickCount();
      importer.loadEvalForImages(settings.galleryDirPath, galleryImages);
      int64 t4 = cv::getTickCount();
      std::cout << "Loading eval points took: "
                << static_cast<double>(t4 - t3) / cv::getTickFrequency()
                << " seconds\n";
    }

    std::chrono::duration<double> dt =
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::high_resolution_clock::now() - start);
    std::cout << "Finished reconstruction import. Time: " << dt.count() << "s"
              << std::endl;
  }

  for (auto &img : queryImages) {
    if (!img->csvrow) {
      img->csvrow = std::make_shared<CSVRow>();
    }

    if (settings.useDatabase) {
      img->csvrow->gallerySize = imageNumDB;
    }
  }

  if (settings.useMultipleModels) {
    std::cout << settings.evaluateCNNDir.toStdString() << std::endl;
    QDirIterator it(settings.evaluateCNNDir, QStringList() << "*.onnx",
                    QDir::Files);
    settings.useMultipleModels = false;
    settings.evaluateCNNDir = QString();
    while (it.hasNext()) {
      QString modelPath = it.next();
      std::cout << "--------------------------------------------" << std::endl;
      std::cout << "Do Multi-Model-Query: " << modelPath.toStdString()
                << std::endl;
      std::cout << "--------------------------------------------" << std::endl;
      settings.retrievalNetPath = modelPath;
      runPipeline(settings, db, queryImages, galleryImages);
    }
  } else {
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "Do Single-Model-Query: "
              << settings.retrievalNetPath.toStdString() << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
    runPipeline(settings, db, queryImages, galleryImages);
  }

  return 0;
}
