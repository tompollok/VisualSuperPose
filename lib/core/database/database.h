#ifndef DATABASE_H
#define DATABASE_H

#include <QSqlDatabase>
#include <QSqlQuery>

#include <opencv2/opencv.hpp>

#include <vector>
#include <string>

#include "../core/types/intrinsics.h"
#include "../core/types/extrinsics.h"

#include "ppbafloc-core_export.h"

/**
 * @brief The Database class provide the storage for all
 * paramters into a sqlite3 database
 */
class PPBAFLOC_CORE_EXPORT Database
{
public:
    /**
     * @brief Database: create a database instance with default settings
     */
    Database();

    /**
     * @brief Database: supply a compression method so save storage usage
     * but may increase runtime
     */
    Database(bool);

    /**
     * @brief createConnection: create connection with sqlite3 server
     */
    bool createConnection(QString file);

    // get all ids
    /**
     * @brief getIDList: get a list of id of all existent data
     */
    std::vector<int> getIDList();

    /**
     * @brief getNumImages: get the amount of existent data
     */
    size_t getNumImages();


    /**
     * @brief getPathList: get all the path and conresponding id from database
     */
    bool getPathList(std::vector<std::pair<int, std::string>>& outPaths);

    // ==================== get data with id ====================
    /**
     * @brief getPath get path with the given id
     */
    std::string getPath(int id);                       // 1. path -> string

    /**
     * @brief getSift get SIFT feature with the given id
     */
    cv::Mat getSift(int id);                           // 2. sift -> 128 float -> cv::Mat
    /**
     * @brief getKeyPoint get list of SIFT keypoints with the given id
     */
    std::vector<cv::Point2f> getKeyPoint(int id);                  // 3. features:keypoint -> cv::KeyPoint
    int getLandmarkID(int id);                         // 4. landmark -> integer
    /**
     * @brief getFbow get Fbow ByteArray with the given id
     */
    QByteArray getFbow(int id);                        // 5. fbow -> byte array (implementation dependend)
    /**
     * @brief getHashVector get hash matrix with the given id
     */
    cv::Mat getHashVector(int id);         // 6. hash_vector -> vector<double>
    /**
     * @brief getCameraIntrinsics get the intrinsics parameters with the given id
     */
    Intrinsics getCameraIntrinsics(int id);            // 7. camera_parameters -> vector<double>

    /**
     * @brief getCameraExtrinsics get the extrinsics parameters with the given id
     */
    Extrinsics getCameraExtrinsics(int id);

    // ==================== get data for all ids ====================
    /**
     * @brief getFBowAll get all fbow ByteArray in the database with the given id
     */
    bool getFBowAll(std::function<bool (int id, const QByteArray& fbow)> callback);

    /**
     * @brief getHashAll get all Hash Matrix in the database with the given id
     */
    bool getHashAll(std::function<bool (int, QByteArray &)> callback);

    /**
     * @brief getFBowPathAll get all fbow and path in the database with the given id
     */
    bool getFBowPathAll(std::function<bool (const QString& path, const QByteArray& fbow)> callback);

    /**
     * @brief getHashPathAll get all hash matrix and path in the database with the given id
     */
    bool getHashPathAll(std::function<bool (const QString& path, QByteArray& hash)> callback);

    // ==================== insert data with id ====================
    /**
     * @brief addPath add path into the database with the given id
     */
    bool addPath(int id, std::string path);

    /**
     * @brief addSift add SIFT features into the database with the given id
     */
    bool addSift(int id, const cv::Mat &sift);

    /**
     * @brief addKeyPoint add list of SIFT keypoints into the database with the given id
     */
    bool addKeyPoint(int id, const std::vector<cv::Point2f> &keypoint);

    /**
     * @brief addLandmarkID add landmark into the database with the given id
     */
    bool addLandmarkID(int id, int landmarkID);

    /**
     * @brief addFbow add fbow ByteArray into the database with the given id
     */
    bool addFbow(int id, const QByteArray &fbowVector);

    /**
     * @brief addHashVector add Hash Matrix into the database with the given id
     */
    bool addHashVector(int id, const cv::Mat &hashVector);

    /**
     * @brief addCameraIntrinsics add intrinsics parameters into the database with the given id
     */
    bool addCameraIntrinsics(int id, const Intrinsics &cameraIntrinsics);

    /**
     * @brief addCameraExtrinsics add extrinsics parameters into the database with the given id
     */
    bool addCameraExtrinsics(int id, const Extrinsics &cameraExtrinsics);

    /**
     * @brief addKeyPointAndSift  add SIFT features and list of keypoints into the database with the given id
     */
    bool addKeyPointAndSift(int id, const std::vector<cv::Point2f> &keypoint, const cv::Mat &sift);

    /**
     * @brief addPathExtrinsicsIntrinsics  add path, extrinsics parameters and intrinsics parameters into the database with the given id
     */
    bool addPathExtrinsicsIntrinsics(int id, const std::string& path, const Intrinsics &cameraIntrinsics, const Extrinsics &cameraExtrinsics);

    /**
     * @brief addPathExtrinsicsIntrinsicsBatch add path, extrinsics parameters and intrinsics parameters into the database with the given
     * list of ids
     */
    bool addPathExtrinsicsIntrinsicsBatch(const std::vector<int>& ids,
                                          const std::vector<const std::string *> &paths,
                                          const std::vector<const Intrinsics *> &cameraIntrinsics,
                                          const std::vector<const Extrinsics *> &cameraExtrinsics,
                                          size_t size = 0);

    /**
     * @brief updateFBoWBatch update multiple FBoW values in one go
     * @param ids image ids
     * @param bows values to update
     * @param size determines how many values from bows are written, if -1 it is set to the size of ids
     */
    bool updateFBoWBatch(const std::vector<int>& ids, std::vector<QByteArray>& bows, int size = -1);

    /**
     * @brief transaction start transaction
     */
    bool transaction() {return db.transaction();}

    /**
     * @brief commit current transaction
     */
    bool commit() {return db.commit();}

    /**
     * @brief rollback cancel current transaction
     */
    bool rollback() {return db.rollback();}

private:
    QSqlDatabase db;
    bool doCompress = false;

    bool setID(int id, std::string tableName);

    QSqlQuery mQSaveFbow;

    bool addCameraPose(int id, std::vector<double> cameraPose);
    std::vector<double> getCameraPose(int id);


    void intrinsicsToByteArray(const Intrinsics& intr, QByteArray& outArray);
    void extrinsicsToByteArray(const Extrinsics& extr, QByteArray& outArray);
};

#endif // DATABASE_H
