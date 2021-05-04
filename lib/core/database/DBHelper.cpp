#include <import/colmapimporter.h>
#include <QtCore/QDir>
#include <utils/SiftHelpers.h>
#include "DBHelper.h"
#include <string>


DBHelper::DBHelper(Database& db) : mDB(db) {}

void DBHelper::fillDatabaseExtrIntr(const std::string &path) {
    std::cout << "saving extr and intr into database" << std::endl;
    ColmapImporter importer;

    // temp is a list of images
    std::vector<std::shared_ptr<Image>> temp;

    QDir dir(QString::fromStdString(path));

    QFileInfoList dirs = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);
    int id = 0;
    for (const auto & d : dirs)
    {
        std::cout << "Importing images and colmap parameters for " << d.filePath().toStdString() << std::endl;

        auto t0 = std::chrono::high_resolution_clock::now();
        importer.importImages(d.filePath() + "/model", d.filePath() + "/images", temp);
        auto t1 = std::chrono::high_resolution_clock::now();

        mDB.transaction();
        for (const auto& t : temp)
        {
            if (QFile::exists(QString::fromStdString(t->path)))
            {
                mDB.addPathExtrinsicsIntrinsics(id++, t->path, t->intrinsics, t->extrinsics);
            }
            else
            {
                std::cout << "Warning: File not found \"" << t->path << "\"" << std::endl;
            }
        }
        mDB.commit();

        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "  Import: " << std::chrono::duration<double>(t1 - t0).count() << "s - DB Write: "
                                << std::chrono::duration<double>(t2 - t1).count() << "s - " << temp.size() << " images" << std::endl;

        temp.clear();
    }
}

void DBHelper::fillDatabaseExtrIntr(const std::string &imgDir, const std::string &recDir) {
    std::vector<std::shared_ptr<Image>> temp;
    ColmapImporter importer;

    std::cout << "importing images and colmap parameters" << std::endl;
    importer.importImages(QString::fromStdString(recDir), QString::fromStdString(imgDir), temp);

    int id = 0;
    try {
        std::vector<int> idList = this->mDB.getIDList();
        if (idList.size() > 0) {
            id = idList.size();
            std::cout << "Already " << id << " images in the database. ID set to " << id << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "Database does not exist yet. ID set to 0." << std::endl;
    }

    int onePercent = temp.size()/100;
    for(uint i = 0; i < temp.size(); i++) {
        if(i%onePercent == 0) {
          double percentage = ((double)i/(double)temp.size())*100.0;
          std::cout << "Saved extr and intr of " << percentage << "% of images" << std::endl;
        }
        std::shared_ptr<Image> img = temp[i];
        if (QFile(QString::fromStdString(img->path)).exists()) {
            //std::cout << "saving " << img->path << " in DB" << std::endl;
            this->mDB.addPathExtrinsicsIntrinsics(id, img->path, img->intrinsics, img->extrinsics);
            id++;
        }
    }
    temp.clear();
}

bool DBHelper::checkIfPathInDB(const std::string &path) {
    std::vector<int> idList = this->mDB.getIDList();
    for (const auto &id: idList) {
        if (this->mDB.getPath(id) == path) {
            return true;
        }
    }
    return false;
}

void DBHelper::fillDatabaseSIFT() {
    std::cout << "saving sift into database" << std::endl;

    std::vector<std::shared_ptr<Image>> imgList;
    std::shared_ptr<Image> img = std::shared_ptr<Image>(new Image);
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::Point2f> points;
    cv::Mat descriptors;

    std::vector<int> idList = this->mDB.getIDList();

    int onePercent = idList.size()/100;
    for (uint i = 0; i < idList.size(); i++) {
        int id = idList[i];
        if(i%onePercent == 0) {
          double percentage = ((double)i/(double)idList.size())*100.0;
          std::cout << "Saved SIFT features of " << percentage << "% of images" << std::endl;
        }
        img->path = this->mDB.getPath(id);
        SiftHelpers::extractSiftFeatures(img->path, descriptors, keypoints);

        cv::KeyPoint::convert(keypoints, points);

        // this->mDB.addKeyPointAndSift(id, points, descriptors);
        // keypoints and sift are not in the same table
        this->mDB.addKeyPoint(id, points);
        this->mDB.addSift(id, descriptors);
    }

    std::cout << "\nsift saved" << std::endl;
}

void DBHelper::fillDatabase(std::string &path) {
    this->fillDatabaseExtrIntr(path);
    this->fillDatabaseSIFT();
}


void DBHelper::getAllImages(std::vector<std::shared_ptr<Image>> &images) {
    std::shared_ptr<Image> img;
    std::vector<int> idList = this->mDB.getIDList();

    for (const auto &id : idList) {
        this->getImage(id, img);
        images.push_back(img);
    }
}

void DBHelper::getImagesByList(std::vector<int> &idList, std::vector<std::shared_ptr<Image>> &images) {
    std::shared_ptr<Image> img;

    for (const auto &id : idList) {
        this->getImage(id, img);
        images.push_back(img);
    }
}

void DBHelper::getImage(const int id, std::shared_ptr<Image> &img) {
    img->path = this->mDB.getPath(id);
    img->extrinsics = this->mDB.getCameraExtrinsics(id);
    img->intrinsics = this->mDB.getCameraIntrinsics(id);
    img->siftDescriptors = this->mDB.getSift(id);
    cv::KeyPoint::convert(this->mDB.getKeyPoint(id), img->siftKeypoints);
}

bool DBHelper::getImageByPath(std::string &imgPath, std::shared_ptr<Image> &img) {
    std::vector<int> idList = this->mDB.getIDList();

    for(const auto &id: idList) {
        if (this->mDB.getPath(id) == imgPath) {
            getImage(id, img);
            return true;
        }
    }
    return false;
}

/*
void DBHelper::fillDatabase(std::string &dbPath, std::string &vocabPath, std::vector<std::shared_ptr<Image>> &gallery) {
    this->mDB.createConnection(QString::fromStdString(dbPath));

    fbow::Vocabulary voc;
    voc.readFromFile(vocabPath);
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::Point2f> points;
    cv::Mat descriptors;
    fbow::fBow bow;
    QByteArray data;

    int id = 0;
    for (const auto &img: gallery) {

        SiftHelpers::extractSiftFeatures(img->path, descriptors, keypoints);
        bow = voc.transform(descriptors);
        img->bow = std::shared_ptr<BoW>(bow);
        cv::KeyPoint::convert(keypoints, points);

        db.addKeyPoint(id, points);
        db.addPath(id, img->path);
        db.addSift(id, descriptors);
        db.addCameraIntrinsics(id, img->intrinsics);
        db.addCameraExtrinsics(id, img->extrinsics);

        //data = QByteArray::fromRawData((const char *)&bow, sizeof(bow));

        data = img->bow->toByteArray();
        db.addFbow(id, data);
        id++;
        std::cout << points << std::endl;
        break;
    }
}
 */
