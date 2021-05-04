#include <FbowRetrieval.h>
#include <SuperGlueMatcher.h>
#include <import/colmapimporter.h>
#include <utils/iohelpers.h>

#include <QDir>
#include <QFileInfo>
#include <QString>
#include <QTextStream>
#include <iostream>
#include <opencv2/opencv.hpp>

struct AppSettings {
  bool load(std::string configPath) {
    cv::FileStorage fs(configPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
      return false;
    }

    sgPath = QString::fromStdString(fs["superglue_path"]);
    workingDir = QString::fromStdString(fs["working_dir"]);
    matchesFile = QString::fromStdString(fs["match_pairs_file"]);

    superpointModel = QString::fromStdString(fs["superpoint_model"]);
    superglueModel = QString::fromStdString(fs["superglue_model"]);
    image1 = QString::fromStdString(fs["image1"]);
    image2 = QString::fromStdString(fs["image2"]);

    return true;
  }

  void printSettings() {
    std::cout << "Settings:\n"
              << "    Working Dir: " << workingDir.toStdString() << std::endl
              << "    SuperGlue Path: " << sgPath.toStdString() << std::endl
              << "    Match Pairs File: " << matchesFile.toStdString()
              << std::endl

              << "    SuperPoint model: " << superpointModel.toStdString()
              << std::endl
              << "    SuperGlue model: " << superpointModel.toStdString()
              << std::endl
              << "    Image1: " << image1.toStdString() << std::endl
              << "    Image2: " << image2.toStdString() << std::endl;
  }

  QString sgPath;
  QString workingDir;
  QString matchesFile;

  QString superpointModel;
  QString superglueModel;
  QString image1;
  QString image2;
};

void detectDNN(QString modelFile, QString image) {
  cv::dnn::Net net = cv::dnn::readNetFromONNX(modelFile.toStdString());

  cv::Mat img = cv::imread(image.toStdString(), cv::IMREAD_GRAYSCALE);
  cv::resize(img, img, cv::Size(640, 480));
  cv::Mat blob = cv::dnn::blobFromImage(img);

  net.setInput(blob);
  cv::Mat detections;
  net.forward(detections, "54");

  for (int i = 0; i < detections.dims; ++i)
    std::cout << detections.size[i] << std::endl;
}

void executePythonScripts(const AppSettings& settings) {
  QString execString =
      "python \"%2/match_pairs.py\" --input_pairs \"%1/%3\" --input_dir \"%1\" "
      "--output_dir \"%1\" --superglue outdoor --viz";
  execString = execString.arg(settings.workingDir)
                   .arg(settings.sgPath)
                   .arg(settings.matchesFile);

  std::cout << execString.toStdString() << std::endl;
  system(execString.toStdString().c_str());
}

void jitMatching(const AppSettings& settings) {
  SuperGlueMatcher matcher(settings.superpointModel.toStdString(),
                           settings.superglueModel.toStdString(), 1000);
  matcher.getMatchesForTwoImages(settings.image1.toStdString(),
                                 settings.image2.toStdString());
}

int main(int argc, char* argv[]) {
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

  // executePythonScripts(settings);
  jitMatching(settings);

  return 0;
}
