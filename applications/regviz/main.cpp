#include <import/colmapimporter.h>
#include <registration.h>
#include <types/image.h>
#include <types/worldpoint.h>
#include <utils/SiftHelpers.h>

#include <QFile>
#include <QGuiApplication>
#include <QTextStream>
#include <vector>

#include "window.h"

/**
 * @brief The AppSettings struct for loading the app settings from *.yml
 */
struct AppSettings {
  bool load(std::string configPath) {
    cv::FileStorage fs(configPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
      return false;
    }

    galleryDirPath = QString::fromStdString(fs["gallery_directory"]);
    reconstructionDirPath =
        QString::fromStdString(fs["reconstruction_directory"]);
    inputImageList = QString::fromStdString(fs["input_image_list"]);

    auto node = fs["use_superglue"];
    if (!node.isNone()) {
      useSuperglue = static_cast<int>(node);
    }

    node = fs["superpoint_model"];
    if (!node.isNone()) {
      superPointModel = QString::fromStdString(node);
    }
    node = fs["superglue_model"];
    if (!node.isNone()) {
      superGlueModel = QString::fromStdString(node);
    }
    node = fs["superpoint_resize_width"];
    if (!node.isNone()) {
      superPointResizeWidth = static_cast<int>(node);
    }

    return true;
  }

  void printSettings() const {
    std::cout << "Settings:\n"
              << "    Gallery Dir: " << galleryDirPath.toStdString()
              << std::endl
              << "    Reconstruction Dir: "
              << reconstructionDirPath.toStdString() << std::endl
              << "    Input Image List: " << inputImageList.toStdString()
              << std::endl
              << "    Registration: "
              << (useSuperglue ? "SuperGlue" : "Classic") << std::endl;
  }

  QString galleryDirPath;
  QString reconstructionDirPath;
  QString inputImageList;

  QString superGlueModel = "SuperGlue.zip";
  QString superPointModel = "SuperPoint.zip";
  int superPointResizeWidth = 500;
  bool useSuperglue = false;
};

void loadInputImages(const QString& file,
                     std::vector<std::shared_ptr<Image>>& galleryImages,
                     std::shared_ptr<Image>& outQueryImage,
                     std::vector<std::shared_ptr<Image>>& outReferenceImages) {
  QFile f(file);
  f.open(QFile::ReadOnly);
  QTextStream ts(&f);

  bool first = true;
  while (!ts.atEnd()) {
    std::string line = ts.readLine().trimmed().toStdString();

    bool found = false;
    for (auto& i : galleryImages) {
      if (i->path.compare(line) == 0) {
        found = true;

        if (first) {
          first = false;
          outQueryImage = i;
        } else {
          outReferenceImages.push_back(i);
        }
      }
    }

    if (!found) {
      std::cout << "Image \"" << line << "\" not found in reconstruction!"
                << std::endl;
    }
  }
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

  bool displayAllFrustums = false;

  QGuiApplication a(argc, argv);

  Window window;
  window.resize(1600, 900);
  window.show();

  std::vector<WorldPoint> pointsReconstruction;

  std::vector<std::shared_ptr<Image>> images;

  ColmapImporter imp;
  imp.importImages(settings.reconstructionDirPath, settings.galleryDirPath,
                   images);
  imp.import3DPoints(settings.reconstructionDirPath, settings.galleryDirPath,
                     pointsReconstruction);

  std::cout << "points size: " << pointsReconstruction.size() << std::endl;
  std::cout << "images size: " << images.size() << std::endl;

  std::shared_ptr<Image> queryImage;
  std::vector<std::shared_ptr<Image>> referenceImages;
  loadInputImages(settings.inputImageList, images, queryImage, referenceImages);

  std::cout << "Reference Images: " << referenceImages.size() << std::endl;

  // Add colmap reconstruction
  window.addPointcloud(pointsReconstruction);

  for (auto& i : (displayAllFrustums ? images : referenceImages)) {
    Frustum f;
    f.setup(i->extrinsics, i->intrinsics, cv::Vec3b(0, 100, 200));
    window.addFrustum(f);
  }

  // Ground Truth frustum
  Frustum gtFrustum;
  gtFrustum.setup(queryImage->extrinsics, queryImage->intrinsics,
                  cv::Vec3b(0, 255, 0));
  window.addFrustum(gtFrustum);

  SiftHelpers::extractSiftFeatures(
      queryImage->path, queryImage->siftDescriptors, queryImage->siftKeypoints);
  for (auto& i : referenceImages) {
    SiftHelpers::extractSiftFeatures(i->path, i->siftDescriptors,
                                     i->siftKeypoints);
  }

  Registration reg;
  Extrinsics calculated;
  std::vector<cv::Point3f> points3f;

  bool success;
  if (settings.useSuperglue) {
    reg.setupDeepLearningBasedPoseEstimation(
        settings.superPointModel.toStdString(),
        settings.superGlueModel.toStdString(), settings.superPointResizeWidth);
    success = reg.applyDeepLearningBasedPoseEstimation(
        false, queryImage, referenceImages, calculated, points3f);
  } else {
    success = reg.applyClassicPoseEstimation(false, queryImage, referenceImages,
                                             calculated, points3f);
  }

  if (!success) {
    std::cout << "Pose Estimation failed" << std::endl;
    return 0;
  }

  // Convert triangulated points to format accepted by pointcloud class
  std::vector<WorldPoint> triangulatedDisplay(points3f.size());
  for (size_t i = 0; i < points3f.size(); ++i) {
    triangulatedDisplay[i].pos =
        cv::Point3d(points3f[i].x, points3f[i].y, points3f[i].z);
    triangulatedDisplay[i].color = cv::Vec3b(255, 0, 0);
  }

  // Add calculated
  Frustum f;
  f.setup(calculated, queryImage->intrinsics, cv::Vec3b(255, 255, 0));
  window.addFrustum(f);

  window.addPointcloud(triangulatedDisplay, 10.f);

  return a.exec();
}
