#include "FbowRetrieval.h"

#include <core.h>
#include <database/DBImporterMT.h>
#include <utils/SiftHelpers.h>
#include <utils/iohelpers.h>

#include <QDebug>
#include <QFile>
#include <QtCore/QBuffer>
#include <QtCore/QDataStream>
#include <QtCore/QDirIterator>
#include <chrono>
#include <fstream>
#include <thread>

#include "Fbow.h"

void calcScoreImage(std::vector<std::string> inputFiles,
                    const std::string& vocabPath,
                    std::vector<fbow::fBow>& queryBows,
                    std::vector<std::vector<std::pair<int, double>>>& scores,
                    size_t scoreVecOffset);
void calcScoreMultipleWithDB(
    Database& db, std::vector<std::vector<std::pair<int, double>>>& scores,
    const std::vector<fbow::fBow>& queryFbows);

FbowRetrieval::FbowRetrieval(const std::string& vocabPath,
                             const std::string& trainingDirPath,
                             const std::string& galleryDirPath) {
  this->setVocabPath(vocabPath);
  this->mGalleryDirPath = galleryDirPath;
  this->mTrainingDirPath = trainingDirPath;
};

FbowRetrieval::FbowRetrieval(const std::string& vocabPath,
                             const std::string& trainingDirPath,
                             const std::string& galleryDirPath, Database* db)
    : FbowRetrieval(vocabPath, trainingDirPath, galleryDirPath) {
  mDB = db;
}

void runCalcBow(std::vector<std::shared_ptr<Image>>& images, int offset,
                int size, fbow::Vocabulary& voc) {
  size_t end = offset + size;
  for (size_t i = offset; i < end; ++i) {
    auto bow = voc.transform(images[i]->siftDescriptors);
    setBoW(images[i].get(), bow);
  }
}

void FbowRetrieval::fillDBFbow(int nThreads) {
  if (mDB == nullptr) {
    throw std::runtime_error("FbowRetrieval::fillDBFBow(): No Database given");
  }

  if (nThreads <= 0) {
    nThreads = 1;
  }

  std::cout << "saving fbow into database" << std::endl;
  if (!this->mVocabExists) {
    std::cout << "Vocabulary has not been found or has not been created yet. "
                 "It will be created now. "
              << std::endl;
    this->createFbowVocabulary();
  }
  fbow::Vocabulary voc;
  voc.readFromFile(this->mVocabPath);

  std::vector<int> idList = mDB->getIDList();

  const size_t batchSize = 500;  // Limit Memory usage

  std::vector<std::shared_ptr<Image>> temp(batchSize);
  for (size_t i = 0; i < temp.size(); ++i) {
    temp[i] = std::make_shared<Image>();
  }

  auto t0 = std::chrono::high_resolution_clock::now();

  size_t batches = std::ceil(idList.size() / static_cast<double>(batchSize));
  for (size_t b = 0; b < batches; ++b) {
    std::cout << "\rSaving FBoW Vectors. Batch " << b + 1 << "/" << batches
              << std::flush;
    std::cout << std::endl;

    auto t01 = std::chrono::high_resolution_clock::now();

    size_t bsize = batchSize;
    size_t offset = b * batchSize;
    if (b == batches - 1) {
      bsize = idList.size() - offset;
    }

    for (size_t i = 0; i < bsize; ++i) {
      temp[i]->siftDescriptors = mDB->getSift(idList[offset + i]);
    }

    auto t02 = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    for (size_t i = 0; i < static_cast<size_t>(nThreads); ++i) {
      size_t threadBatch = bsize / nThreads;
      size_t threadOffset = threadBatch * i;
      if (i == static_cast<size_t>(nThreads) - 1) {
        threadBatch = bsize - threadOffset;
      }
      threads.push_back(std::thread(runCalcBow, std::ref(temp), threadOffset,
                                    threadBatch, std::ref(voc)));
    }

    for (auto& t : threads) {
      t.join();
    }

    auto t03 = std::chrono::high_resolution_clock::now();

    mDB->transaction();
    for (size_t i = 0; i < bsize; ++i) {
      mDB->addFbow(idList[offset + i], temp[i]->bow->toByteArray());
    }
    mDB->commit();

    auto t04 = std::chrono::high_resolution_clock::now();
    std::cout << "load: " << std::chrono::duration<double>(t02 - t01).count()
              << " bow: " << std::chrono::duration<double>(t03 - t02).count()
              << " save: " << std::chrono::duration<double>(t04 - t03).count()
              << std::endl;
  }
  std::cout << std::endl;

  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "Elapsed: " << std::chrono::duration<double>(t1 - t0).count()
            << std::endl;
}

void FbowRetrieval::setVocabPath(const std::string& vocabPath) {
  if (!IOHelpers::existsFile(QString::fromStdString(vocabPath))) {
    // std::cout << "Image file " << vocabPath << " does not exist." <<
    // std::endl; std::cout << "Consider creating the vocabulary first." <<
    // std::endl;
    this->mVocabExists = false;
  } else {
    this->mVocabExists = true;
  }
  this->mVocabPath = vocabPath;
};

void FbowRetrieval::setVocabCreationFilterFile(const std::string& filterFile) {
  mCleanCSV = filterFile;
}

void FbowRetrieval::getFilteredImages(std::vector<std::string>& imageFiles,
                                      const std::string& filterFile) {
  QStringList filter;
  filter << "*.jpg"
         << "*.png"
         << "*.jpeg";
  QDirIterator it(QString::fromStdString(this->mTrainingDirPath), filter,
                  QDir::Files, QDirIterator::Subdirectories);

  imageFiles.clear();
  for (int i = 0; it.hasNext(); ++i) {
    std::string img_path = it.next().toStdString();
    if (this->isImageClean(img_path, filterFile)) {
      imageFiles.push_back(img_path);
    }
  }
}

void FbowRetrieval::createFbowVocabulary() {
  std::vector<cv::Mat> features;

  if (!mCleanCSV.empty()) {
    std::vector<std::string> imgList;
    getFilteredImages(imgList, mCleanCSV);
    SiftHelpers::extractSiftFeaturesImgList(imgList, features, -1);
  } else {
    SiftHelpers::extractSiftFeaturesDir(this->mTrainingDirPath, features, -1);
  }

  fbow::Vocabulary voc;
  fbow::VocabularyCreator creator;
  // fbow::VocabularyCreator::Params params =
  // fbow::VocabularyCreator::Params(10, 3, 7, 11);
  fbow::VocabularyCreator::Params params =
      fbow::VocabularyCreator::Params(10, 6, 7, 15);

  params.verbose = true;
  creator.create(voc, features, "SIFT", params);

  std::cout << "Saving vocabulary ..." << std::endl;
  voc.saveToFile(this->mVocabPath);
  this->mVocabExists = true;
};

bool FbowRetrieval::isImageClean(const std::string img_path,
                                 const std::string& cleanCSV) {
  QFileInfo fi(QString::fromStdString(img_path));
  QString imageName = fi.fileName();
  imageName = imageName.split(".", QString::SkipEmptyParts).at(0);
  std::cout << "isImageclean" << imageName.toStdString() << std::endl;
  QFile file(QString::fromStdString(cleanCSV));
  file.open(QIODevice::ReadOnly);
  while (!file.atEnd()) {
    QString line = file.readLine();
    if (line.contains(imageName, Qt::CaseSensitive)) {
      file.close();
      std::cout << "File clean is " << imageName.toStdString() << std::endl;
      return true;
    }
  }
  file.close();
  return false;
}

void FbowRetrieval::retrieveImagesDB(
    const std::vector<std::shared_ptr<Image>>& queries,
    std::vector<std::vector<std::shared_ptr<Image>>>& outRetrievedPerQuery,
    int numRetrieved) {
  retrieve(queries, {}, outRetrievedPerQuery, -1, numRetrieved, 1, true);
}

void FbowRetrieval::retrieveImagesGalleryDir(
    const std::vector<std::shared_ptr<Image>>& queries,
    std::vector<std::vector<std::shared_ptr<Image>>>& outRetrievedPerQuery,
    int maxNumberGalleryImages, int numRetrieved, unsigned int numThreads) {
  retrieve(queries, {}, outRetrievedPerQuery, maxNumberGalleryImages,
           numRetrieved, numThreads, false);
}

void FbowRetrieval::retrieveImagesFromList(
    const std::vector<std::shared_ptr<Image>>& queries,
    const std::vector<std::shared_ptr<Image>>& galleryImgs,
    std::vector<std::vector<std::shared_ptr<Image>>>& outRetrievedPerQuery,
    int maxNumberGalleryImages, int numRetrieved, unsigned int numThreads) {
  retrieve(queries, galleryImgs, outRetrievedPerQuery, maxNumberGalleryImages,
           numRetrieved, numThreads, false);
}

void FbowRetrieval::retrieve(
    const std::vector<std::shared_ptr<Image>>& queries,
    const std::vector<std::shared_ptr<Image>>& galleryImgs,
    std::vector<std::vector<std::shared_ptr<Image>>>& outRetrievedPerQuery,
    int maxNumGalleryImgs, int numRetrieved, unsigned int numThreads,
    bool useDB) {
  std::cout << "Retrieving for " << queries.size() << " queries" << std::endl;

  fbow::fBow2 fBowInstance;
  fbow::Vocabulary voc;

  if (this->mVocabExists) {
    voc.readFromFile(this->mVocabPath);
  } else {
    this->createFbowVocabulary();
  }

  std::cout << "Calculating Query BoWs....." << std::flush;
  std::vector<fbow::fBow> queryBows;

  auto t00 = std::chrono::high_resolution_clock::now();
  for (auto& queryImage : queries) {
    cv::Mat desc;
    std::vector<cv::KeyPoint> kps;
    SiftHelpers::extractSiftFeatures(queryImage->path, desc, kps);

    queryBows.push_back(voc.transform(desc));
  }
  auto t01 = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration<double>(t01 - t00).count() << "s"
            << std::endl;

  std::cout << "Calculating scores........." << std::flush;
  std::vector<std::vector<std::pair<int, double>>> scores(queryBows.size());
  std::vector<std::string> files;

  auto t10 = std::chrono::high_resolution_clock::now();
  if (useDB) {
    calcScoreMultipleWithDB(*mDB, scores, queryBows);
  } else {
    if (galleryImgs.empty()) {
      QStringList filter;
      filter << "*.jpg"
             << "*.png"
             << "*.jpeg";
      QDirIterator it(QString::fromStdString(this->mGalleryDirPath), filter,
                      QDir::Files, QDirIterator::Subdirectories);

      int counter = 0;
      for (int i = 0; it.hasNext(); ++i) {
        files.push_back(it.next().toStdString());
        counter++;

        if (counter == maxNumGalleryImgs - 1 && maxNumGalleryImgs > 0) break;
      }
    } else {
      size_t limit = galleryImgs.size();
      if (maxNumGalleryImgs > 0) {
        limit = maxNumGalleryImgs;
      }
      for (size_t i = 0; i < limit; ++i) {
        files.push_back(galleryImgs[i]->path);
      }
    }

    size_t n = files.size();
    for (const auto& queryImage : queries) {
      if (!queryImage->csvrow) {
        queryImage->csvrow = std::shared_ptr<CSVRow>(new CSVRow);
      }
      queryImage->csvrow->gallerySize = n;
    }
    for (auto& s : scores) {
      s.resize(n);
    }

    if (numThreads > 1) {
      size_t perThread = n / numThreads;
      std::vector<std::thread> threads;

      size_t offset = 0;
      for (size_t t = 0; t < numThreads; ++t) {
        std::vector<std::string> input;
        for (size_t i = offset; i < n; ++i) {
          input.push_back(files[i]);

          if (t != numThreads - 1)
            if (i == offset + perThread - 1) break;
        }

        std::cout << input.size() << std::endl;

        threads.push_back(std::thread(calcScoreImage, input,
                                      std::ref(mVocabPath), std::ref(queryBows),
                                      std::ref(scores), offset));
        offset += perThread;
      }

      for (auto& t : threads) {
        t.join();
      }
    } else {
      calcScoreImage(files, mVocabPath, queryBows, scores, 0);
    }
  }
  auto t11 = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration<double>(t11 - t10).count() << "s"
            << std::endl;

  std::cout << "Sorting scores............." << std::flush;

  auto t20 = std::chrono::high_resolution_clock::now();
  for (auto& s : scores) {
    sort(s.begin(), s.end(), sortTupleListBySecElemID);
    if (s.size() > static_cast<size_t>(numRetrieved))
      s.resize(numRetrieved + 1);
  }
  auto t21 = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration<double>(t21 - t20).count() << "s"
            << std::endl;

  std::cout << "Populate Retrieved Vector.." << std::flush;
  outRetrievedPerQuery.resize(queries.size());

  auto t30 = std::chrono::high_resolution_clock::now();
  for (size_t queryIdx = 0; queryIdx < queries.size(); ++queryIdx) {
    for (const auto& score : scores[queryIdx]) {
      std::shared_ptr<Image> t = std::shared_ptr<Image>(new Image);
      if (useDB) {
        t->path = mDB->getPath(score.first);
      } else {
        t->path = files[score.first];
      }
      if (t->path == queries[queryIdx]->path) {
        std::cout << "Skipping query image in retrieved images " << std::endl;
        continue;
      }
      outRetrievedPerQuery[queryIdx].push_back(t);
    }
  }
  auto t31 = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration<double>(t31 - t30).count() << "s"
            << std::endl;
}

void calcScoreImage(std::vector<std::string> inputFiles,
                    const std::string& vocabPath,
                    std::vector<fbow::fBow>& queryBows,
                    std::vector<std::vector<std::pair<int, double>>>& scores,
                    size_t scoreVecOffset) {
  fbow::Vocabulary voc;
  voc.readFromFile(vocabPath);

  for (size_t fileIdx = 0; fileIdx < inputFiles.size(); ++fileIdx) {
    if (fileIdx % 1000 == 0) {
      std::cout << "finished " << fileIdx << " out of " << inputFiles.size()
                << " images " << std::endl;
    }
    double score = 0.0;
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keypoints;
    SiftHelpers::extractSiftFeatures(inputFiles[fileIdx], descriptors,
                                     keypoints);

    for (size_t i = 0; i < queryBows.size(); ++i) {
      try {
        const fbow::fBow& imgBow = voc.transform(descriptors);

        score = fbow::fBow::score(queryBows[i], imgBow);
      } catch (const std::exception& e) {
        std::cout << inputFiles[fileIdx] << ":" << e.what() << std::endl;
      }

      scores[i][scoreVecOffset + fileIdx] = {scoreVecOffset + fileIdx, score};
    }
  }
}

void calcScoreMultipleWithDB(
    Database& db, std::vector<std::vector<std::pair<int, double>>>& scores,
    const std::vector<fbow::fBow>& queryFbows) {
  auto callback = [&](int id, const QByteArray& fbow) -> bool {
    std::istringstream iss(std::ios::binary);
    iss.str(fbow.toStdString());
    fbow::fBow tempFb;
    tempFb.fromStream(iss);

    for (size_t i = 0; i < queryFbows.size(); ++i) {
      double score = 0.;
      try {
        score = fbow::fBow::score(queryFbows[i], tempFb);
      } catch (const std::exception& e) {
        std::cout << id << ":" << e.what() << std::endl;
      }

      scores[i].push_back({id, score});
    }

    return true;
  };

  db.getFBowAll(callback);
}
