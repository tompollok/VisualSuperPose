#include "database.h"

#include <QSqlError>
#include <QDataStream>
#include <QDebug>

Database::Database()
{

}

Database::Database(bool Compress)
{
    if(Compress)
        this->doCompress = true;
}

bool Database::createConnection(QString file)
{
    if (!QSqlDatabase::contains("database"))
        db = QSqlDatabase::addDatabase("QSQLITE", "database");
    else
        db = QSqlDatabase::database("database");

    db.setDatabaseName(file);

    if(!db.open()) {
        qDebug() << "ERROR: open the database";
    }


    QSqlQuery query(db);

    query.exec("PRAGMA page_size = 16384");
    query.exec("PRAGMA cache_size = 131072");
    query.exec("PRAGMA temp_store = MEMORY");
    query.exec("PRAGMA locking_mode = EXCLUSIVE");
    query.exec("PRAGMA synchronous = OFF");

    query.exec("PRAGMA journal_mode = OFF");
    //query.exec("PRAGMA journal_mode = MEMORY");


    // ===========create a table of images===========

    // create 1. table with id
    query.prepare("CREATE TABLE IF NOT EXISTS mytable("
                  "id                   INTEGER PRIMARY KEY,"
                  "path                 TEXT,"
                  "landmark             INTEGER,"
                  "intrinsics           BLOB,"
                  "extrinsics           BLOB,"
                  "CONSTRAINT name_unique UNIQUE(path) ON CONFLICT REPLACE"
                  ");");
    if (!query.exec()) {
        qDebug() << "ERROR: CREATE TABLE FAILED mytable: " << query.lastError().text();
        return false;
    }

    // create 2. table with sift
    query.prepare("CREATE TABLE IF NOT EXISTS siftTable("
                  "id                   INTEGER PRIMARY KEY,"
                  "sift                 BLOB);");
    if(!query.exec())
    {
        qDebug() << "ERROR: CREATE TABLE FAILED sift table: " << query.lastError().text();
        return false;
    }

    // create 3. table with keypoint
    query.prepare("CREATE TABLE IF NOT EXISTS keypointTable("
                  "id                   INTEGER PRIMARY KEY,"
                  "keypoint             BLOB);");
    if(!query.exec())
    {
        qDebug() << "ERROR: CREATE TABLE FAILED keypoint table: " << query.lastError().text();
        return false;
    }

    // create 4. table with hash vector
    query.prepare("CREATE TABLE IF NOT EXISTS hashTable("
                  "id                   INTEGER PRIMARY KEY,"
                  "hash_vector          BLOB);");
    if(!query.exec())
    {
        qDebug() << "ERROR: CREATE TABLE FAILED hash table: " << query.lastError().text();
        return false;
    }

    // create 5. table with fbow
    query.prepare("CREATE TABLE IF NOT EXISTS fbowTable("
                  "id                   INTEGER PRIMARY KEY,"
                  "fbow                 BLOB);");
    if(!query.exec())
    {
        qDebug() << "ERROR: CREATE TABLE FAILED fbow table: " << query.lastError().text();
        return false;
    }
    query.finish();


    mQSaveFbow = QSqlQuery(db);
    mQSaveFbow.prepare("UPDATE fbowTable SET fbow=:fbow WHERE id=:id;");

    return true;
}

// ==================== get all ids ====================
std::vector<int> Database::getIDList()
{
    std::vector<int> idList;

    QSqlQuery query(db);
    query.prepare("SELECT id FROM mytable;");
    if(!query.exec()) {
        qDebug() << "ERROR: getIDList" << query.lastError().text();
        std::abort();

    } else {
        while(query.next()) {
            idList.push_back(query.value(0).toInt());
        }
        query.finish();
        return idList;
    }
}

size_t Database::getNumImages()
{
    QSqlQuery query(db);
    query.prepare("SELECT COUNT(id) FROM mytable;");
    if(!query.exec()) {
        qDebug() << "ERROR: getImageNumber" << query.lastError().text();
        std::abort();

    }
    query.next();
    return query.value(0).toUInt();
}

bool Database::getPathList(std::vector<std::pair<int, std::string>> &outPaths)
{
    QSqlQuery query(db);
    query.prepare("SELECT id, path FROM mytable;");
    if(!query.exec()) {
        qDebug() << "ERROR: getIDList" << query.lastError().text();
        return false;
    }

    while(query.next())
    {
        outPaths.push_back({query.value(0).toInt(), query.value(1).toString().toStdString()});
    }
    return true;
}

// ==================== set id ====================
bool Database::setID(int id, std::string tableName)
{
    std::string dbCommand = "INSERT INTO "+tableName+"(id) "
                            "SELECT :id "
                            "WHERE NOT EXISTS"
                            "(SELECT * FROM "+tableName+" WHERE id=:id);";

    QSqlQuery query(db);
    query.prepare(QString::fromStdString(dbCommand));
    query.bindValue(":id", id);
    if(!query.exec()) {
        qDebug() << "ERROR: setID" << query.lastError().text();
        return false;
    }
    query.finish();
    return true;

}

// ==================== get info from images ====================
std::string Database::getPath(int id)
{
    QSqlQuery query(db);
    query.prepare("SELECT path FROM mytable WHERE id = :id;");
    query.bindValue(":id", id);
    if(!query.exec()) {
        qDebug() << "ERROR: getPath" << query.lastError().text();
        std::abort();
    } else {
        query.next();
        auto path = query.value(0).toString().toStdString();
        query.finish();
        return path;
    }
}

cv::Mat Database::getSift(int id)
{
    QSqlQuery query(db);
    query.prepare("SELECT sift FROM siftTable WHERE id = :id;");
    query.bindValue(":id", id);
    if(!query.exec()) {
        qDebug() << "ERROR: getSift" << query.lastError().text();
        std::abort();
    } else {
        query.next();
        QByteArray data = query.value(0).toByteArray();

        if(doCompress == true)
            data = qUncompress(data);

        QDataStream stream(&data, QIODevice::ReadOnly);

        int matType, rows, cols;
        stream >> matType >> rows >> cols;
        QByteArray siftByte;
        stream >> siftByte;
        auto sift = cv::Mat(rows, cols, matType, (void*)siftByte.data()).clone();
        query.finish();
        return sift;
    }
}

std::vector<cv::Point2f> Database::getKeyPoint(int id)
{
    QSqlQuery query(db);
    query.prepare("SELECT keypoint FROM keypointTable WHERE id = :id;");
    query.bindValue(":id", id);
    if(!query.exec()) {
        qDebug() << "ERROR: getKeypoint" << query.lastError().text();
        std::abort();
    }
    else
    {
        query.next();
        QByteArray data = query.value(0).toByteArray();

        // create QDataStream to decode QByteArray data
        QDataStream stream(&data, QIODevice::ReadOnly);

        // get size from stream
        int size;
        stream >> size;

        // create std::vector according to size
        std::vector<cv::Point2f> keypoint(size);

        // loop to get all parameters
        for (int i=0; i<size; i++) {
            stream >> keypoint[i].x;
            stream >> keypoint[i].y;
        }
        query.finish();
        return keypoint;
    }
}

int Database::getLandmarkID(int id)
{
    QSqlQuery query(db);
    query.prepare("SELECT landmark FROM mytable WHERE id = :id;");
    query.bindValue(":id", id);
    if(!query.exec()) {
        qDebug() << "ERROR: getLandmarkID" << query.lastError().text();
        std::abort();
    } else {
        query.next();
        auto id = query.value(0).toInt();
        query.finish();
        return id;
    }

}

// ---------- fbow ----------
QByteArray Database::getFbow(int id)
{
    QSqlQuery query(db);
    query.prepare("SELECT fbow FROM fbowTable WHERE id = :id;");
    query.bindValue(":id", id);
    if(!query.exec()) {
        qDebug() << "ERROR: getFbow" << query.lastError().text();
        std::abort();
    } else {
        query.next();
        QByteArray data = query.value(0).toByteArray();
        query.finish();
        return data;
    }
}

bool Database::getFBowAll(std::function<bool (int, const QByteArray &)> callback)
{
    QSqlQuery query(db);
    query.prepare("SELECT id, fbow FROM fbowTable");
    if (!query.exec())
    {
        qDebug() << "ERROR: getFBowPathAll" << query.lastError().text();
        return false;
    }

    int id;
    QByteArray fbow;
    while (query.next())
    {
        id = query.value(0).toInt();
        fbow = query.value(1).toByteArray();
        if (!callback(id, fbow))
        {
            break;
        }
    }
    return true;
}

bool Database::getFBowPathAll(std::function<bool (const QString&, const QByteArray &)> callback)
{
    QSqlQuery query(db);
    query.prepare("SELECT path, fbow FROM fbowTable INNER JOIN mytable ON fbowTable.id = mytable.id;");
    if (!query.exec())
    {
        qDebug() << "ERROR: getFBowPathAll" << query.lastError().text();
        return false;
    }

    QString path;
    QByteArray fbow;
    while (query.next())
    {
        path = query.value(0).toString();
        fbow = query.value(1).toByteArray();
        if (!callback(path, fbow))
        {
            break;
        }
    }
    return true;
}

// ---------- camera pose ======
std::vector<double> Database::getCameraPose(int id)
{
    QSqlQuery query(db);
    query.prepare("SELECT extrinsics FROM mytable WHERE id = :id;");
    query.bindValue(":id", id);
    if(!query.exec()) {
        std::cerr << "ERROR: getExtrinsics" << query.lastError().text().toStdString() << std::endl;
        throw std::runtime_error(query.lastError().text().toStdString());
    } else {
        query.next();
        QByteArray data = query.value(0).toByteArray();
        QDataStream stream(&data, QIODevice::ReadOnly);

        std::vector<double> pose(6);
        stream >> pose[0];
        stream >> pose[1];
        stream >> pose[2];
        stream >> pose[3];
        stream >> pose[4];
        stream >> pose[5];
        query.finish();
        return pose;
    }
}

cv::Mat Database::getHashVector(int id)
{
    QSqlQuery query(db);
    query.prepare("SELECT hash_vector FROM hashTable WHERE id = :id;");
    query.bindValue(":id", id);
    if(!query.exec())
    {
        qDebug() << "ERROR: getHashVector" << query.lastError().text();
        std::abort();
    } else
    {
        query.next();
        QByteArray data = query.value(0).toByteArray();
        QDataStream stream(&data, QIODevice::ReadOnly);

        int matType, rows, cols;
        stream >> matType >> rows >> cols;
        QByteArray hashByte;
        stream >> hashByte;
        query.finish();
        return cv::Mat(rows, cols, matType, (void*)hashByte.data()).clone();
    }

}

bool Database::getHashAll(std::function<bool (int, QByteArray &)> callback)
{
    QSqlQuery query(db);
    query.prepare("SELECT id, hash_vector FROM hashTable");
    if (!query.exec())
    {
        qDebug() << "ERROR: getHashAll" << query.lastError().text();
        return false;
    }

    int id;
    QByteArray hash;
    while (query.next())
    {
        id = query.value(0).toInt();
        hash = query.value(1).toByteArray();
        if (!callback(id, hash))
        {
            break;
        }
    }
    return true;
}

bool Database::getHashPathAll(std::function<bool (const QString &, QByteArray &)> callback)
{
    QSqlQuery query(db);
    query.prepare("SELECT path, hash_vector FROM hashTable INNER JOIN mytable ON hashTable.id = mytable.id;");
    if (!query.exec())
    {
        qDebug() << "ERROR: getHashAll" << query.lastError().text();
        return false;
    }

    QString path;
    QByteArray hash;
    while (query.next())
    {
        path = query.value(0).toString();
        hash = query.value(1).toByteArray();
        if (!callback(path, hash))
        {
            break;
        }
    }
    return true;
}


// ---------- camera intrinsics parameters ----------
Intrinsics Database::getCameraIntrinsics(int id)
{
    QSqlQuery query(db);
    query.prepare("SELECT intrinsics FROM mytable WHERE id = :id;");
    query.bindValue(":id", id);
    if(!query.exec()) {
        qDebug() << "ERROR: getCameraIntrinsics" << query.lastError().text();
        std::abort();
    } else {
        query.next();
        QByteArray data = query.value(0).toByteArray();
        QDataStream stream(&data, QIODevice::ReadOnly);

        // craete variance
        QByteArray temp;

        // get imageSize from stream
        stream >> temp;
        cv::Size imageSize = *((cv::Size *)temp.data());
        // get focalLength from stream
        stream >> temp;
        cv::Point2d focalLength = *((cv::Point2d *)temp.data());
        // get principalPoint from stream
        stream >> temp;
        cv::Point2d principalPoint = *((cv::Point2d *)temp.data());
        // get distorionCoefficients from stream
        stream >> temp;

        // the data get from Intrinsics is a std::vector<double>
        // however to create an Intrinsics, the distrorionCoefficients should be a 8x1 Matrix
        // so it need to be transform first
        cv::Mat distorionCoefficientsMatrix = cv::Mat(8, 1, CV_64F, temp.data());

        // create Intrinsics and return
        query.finish();
        return Intrinsics(imageSize, focalLength, principalPoint, distorionCoefficientsMatrix);

    }
}

// ---------- camera extrinsics parameters ----------
Extrinsics Database::getCameraExtrinsics(int id)
{
    // get data from getCameraPose
    std::vector<double> data = getCameraPose(id);
    cv::Vec3d rotation = cv::Vec3d(data[0], data[1], data[2]);
    cv::Vec3d translation = cv::Vec3d(data[3], data[4], data[5]);
    Extrinsics::TransformationDirection direction = Extrinsics::TransformationDirection::Ref2Local;
    // create a instance from class Extrinsics
    Extrinsics cameraExtrinsics(rotation, translation, direction);

    return cameraExtrinsics;
}


// ==================== insert info into DB ====================
// ---------- image path ----------
bool Database::addPath(int id, std::string path)
{
    // set id if not exists
    Database::setID(id, "mytable");

    QString qpath = QString::fromStdString(path);
    QSqlQuery query(db);
    query.prepare("UPDATE mytable SET path=:path WHERE id=:id;");
    query.bindValue(":id", id);
    query.bindValue(":path", qpath);
    if(!query.exec())
    {
        qDebug() << "ERROR: addPath" << query.lastError().text();
        return false;
    } else
    {
        query.finish();
        return true;
    }
}

// ---------- sift matrix ----------
bool Database::addSift(int id, const cv::Mat &sift)
{
    // set id if not exists
    Database::setID(id, "siftTable");

    //
    QByteArray data;  // = QByteArray((const char *)&sift, sizeof(sift));
    QDataStream stream(&data, QIODevice::WriteOnly);

    stream << sift.type();
    stream << sift.rows;
    stream << sift.cols;
    const size_t data_size = sift.cols * sift.rows * sift.elemSize();
    QByteArray siftByte = QByteArray::fromRawData( (const char*)sift.ptr(), data_size );

    stream << siftByte;

    // compression rank from 1~9
    if(doCompress == true)
        data = qCompress(data);

    QSqlQuery query(db);
    query.prepare("UPDATE siftTable SET sift=:sift WHERE id=:id;");
    query.bindValue(":id", id);
    query.bindValue(":sift", data);

    if(!query.exec())
    {
        qDebug() << "ERROR: addSift" << query.lastError().text();
        return false;
    } else
    {
        query.finish();
        return true;
    }
}

// ---------- keypoint ----------
bool Database::addKeyPoint(int id, const std::vector<cv::Point2f> &keypoint)
{
    // set id if not exists
    Database::setID(id, "keypointTable");

    // get size of std::vector
    int size = keypoint.size();

    // create a QDataStream to storage all parameters from vector
    QByteArray data;
    QDataStream stream(&data, QIODevice::WriteOnly);

    // save size
    stream << size;

    // loop the full size to storage keypoints
    for (int i=0; i<size; i++) {
        stream << keypoint[i].x;
        stream << keypoint[i].y;
    }

    QSqlQuery query(db);
    query.prepare("UPDATE keypointTable SET keypoint=:keypoint WHERE id=:id;");
    query.bindValue(":id", id);
    query.bindValue(":keypoint", data);
    if(!query.exec()) {
        qDebug() << "ERROR: addKeypoint" << query.lastError().text();
        return false;
    } else {
        query.finish();
        return true;
    }
}

bool Database::addKeyPointAndSift(int id, const std::vector<cv::Point2f> &keypoint, const cv::Mat &sift)
{
    return addKeyPoint(id, keypoint) && addSift(id, sift);
}

bool Database::addLandmarkID(int id, int landmarkID)
{
    // set id if not exists
    Database::setID(id, "mytable");

    // insert into database mytable
    QSqlQuery query(db);
    query.prepare("UPDATE mytable SET landmark=:landmark WHERE id=:id;");

    query.bindValue(":id", id);
    query.bindValue(":landmark", landmarkID);
    if(!query.exec()) {
        qDebug() << "ERROR: add LandmarkID" << query.lastError().text();
        return false;
    } else {
        query.finish();
        return true;
    }
}

// ---------- fbow ----------
bool Database::addFbow(int id, const QByteArray& fbowVector)
{
    // set id if not exists
    Database::setID(id, "fbowTable");

    QSqlQuery query(db);
    query.prepare("UPDATE fbowTable SET fbow=:fbow WHERE id=:id;");

    query.bindValue(":id", id);
    query.bindValue(":fbow", fbowVector);
    if(!query.exec()) {
        qDebug() << "ERROR: add addFbow" << query.lastError().text();
        query.finish();
        return false;
    } else {
        query.finish();
        return true;
    }
}

bool Database::updateFBoWBatch(const std::vector<int> &ids, std::vector<QByteArray> &bows, int size)
{
    if (size <= 0)
    {
        size = ids.size();
    }

    if (ids.size() < static_cast<size_t>(size) || bows.size() < static_cast<size_t>(size))
    {
        throw std::runtime_error("updateFBoWBatch: Invalid argument sizes");
    }

    QVariantList vlIds, vlFbow;
    for (size_t i = 0; i < static_cast<size_t>(size); ++i)
    {
        vlIds << ids[i];
        vlFbow << bows[i];
    }

    db.transaction();
    mQSaveFbow.bindValue(":id", vlIds);
    mQSaveFbow.bindValue(":fbow", vlFbow);
    if (!mQSaveFbow.execBatch())
    {
        db.rollback();
        std::cerr << "ERROR query failed in updateFBoWBatch: " << mQSaveFbow.lastError().text().toStdString() << std::endl;
        return false;
    }
    db.commit();
    return true;
}

// ---------- pose ----------
bool Database::addCameraPose(int id, const std::vector<double> cameraPose)
{
    // set id if not exists
    Database::setID(id, "myTable");

    // save 6 camerapose -> QByteArray
    QByteArray data;
    QDataStream stream(&data, QIODevice::WriteOnly);
    stream << cameraPose[0];
    stream << cameraPose[1];
    stream << cameraPose[2];
    stream << cameraPose[3];
    stream << cameraPose[4];
    stream << cameraPose[5];

    // insert into database mytable
    QSqlQuery query(db);
    query.prepare("UPDATE mytable SET extrinsics=:camera_pose WHERE id=:id;");
    query.bindValue(":id", id);
    query.bindValue(":camera_pose", data);
    if (!query.exec()) {
        qDebug() << "ERROR: addExtrinsics" << query.lastError().text();
        return false;
    } else {
        query.finish();
        return true;
    }
}

// ---------- hash vector ----------
bool Database::addHashVector(int id, const cv::Mat &hashVector)
{
    // set id if no exits
    Database::setID(id, "hashTable");

    QByteArray data;
    QDataStream stream(&data, QIODevice::WriteOnly);

    stream << hashVector.type();
    stream << hashVector.rows;
    stream << hashVector.cols;

    const size_t data_size = hashVector.cols * hashVector.rows * hashVector.elemSize();
    QByteArray hashByte = QByteArray::fromRawData((const char*)hashVector.ptr(), data_size);
    stream << hashByte;

    // SQL
    QSqlQuery query(db);
    query.prepare("UPDATE hashTable SET hash_vector=:hash_vector WHERE id=:id;");
    query.bindValue(":id", id);
    query.bindValue(":hash_vector", data);

    if(!query.exec()) {
        qDebug() << "ERROR: addHashVector" << query.lastError().text();
        return false;
    } else {
        query.finish();
        return true;
    }
}

// ---------- camera intrinsics parameters ----------
bool Database::addCameraIntrinsics(int id, const Intrinsics &cameraIntrinsics)
{
    // set id if not exists
    Database::setID(id, "mytable");

    // get parameters from Intrinsics
    cv::Size imageSize = cameraIntrinsics.imageSize();
    cv::Point2d focalLength = cameraIntrinsics.focalLength();
    cv::Point2d principalPoint = cameraIntrinsics.principalPoint();
    std::vector<double> distorionCoefficients = cameraIntrinsics.distorionCoefficients();

    // create a QByteArray and a Stream to storage
    QByteArray data;
    QDataStream stream(&data, QIODevice::WriteOnly);
    stream << QByteArray((const char *)&imageSize, sizeof (imageSize));
    stream << QByteArray((const char *)&focalLength, sizeof(focalLength));
    stream << QByteArray((const char *)&principalPoint, sizeof(principalPoint));
    stream << QByteArray((const char *)distorionCoefficients.data(), sizeof(double) * distorionCoefficients.size());

    // insert into database mytable
    QSqlQuery query(db);
    query.prepare("UPDATE mytable SET intrinsics=:intrinsics WHERE id=:id;");

    query.bindValue(":id", id);
    query.bindValue(":intrinsics", data);
    if(!query.exec()) {
        qDebug() << "ERROR: addCameraIntrinsics" << query.lastError().text();
        return false;
    } else {
        query.finish();
        return true;
    }
}

// ---------- camera extrinsics parameters ----------
bool Database::addCameraExtrinsics(int id, const Extrinsics &cameraExtrinsics)
{    
    // set direction as Ref2Local
    Extrinsics::TransformationDirection direction = Extrinsics::TransformationDirection::Ref2Local;

    // get 3 from getRotationRodrigues
    cv::Vec3d rotation = cameraExtrinsics.getRotationRodrigues(direction);
    // get 3 from getTranslation
    cv::Vec3d translation = cameraExtrinsics.getTranslation(direction);

    // set 6 double parameters -> std::vector<double>
    std::vector<double> cameraPose(6);
    cameraPose[0] = (double)rotation[0];
    cameraPose[1] = (double)rotation[1];
    cameraPose[2] = (double)rotation[2];
    cameraPose[3] = (double)translation[0];
    cameraPose[4] = (double)translation[1];
    cameraPose[5] = (double)translation[2];

    // call addCameraPose to save 6 double parameters
    return Database::addCameraPose(id, cameraPose);
}

bool Database::addPathExtrinsicsIntrinsics(int id, const std::string& path, const Intrinsics &cameraIntrinsics, const Extrinsics &cameraExtrinsics)
{
  // set id if not exists
  Database::setID(id, "mytable");

  QString qpath = QString::fromStdString(path);

  QByteArray dataIntrinsics, dataExtrinsics;
  intrinsicsToByteArray(cameraIntrinsics, dataIntrinsics);
  extrinsicsToByteArray(cameraExtrinsics, dataExtrinsics);

  QSqlQuery query(db);
  query.prepare("UPDATE mytable SET path=:path, intrinsics=:intrinsics, extrinsics=:camera_pose WHERE id=:id;");
  query.bindValue(":id", id);
  query.bindValue(":path", qpath);
  query.bindValue(":intrinsics", dataIntrinsics);
  query.bindValue(":camera_pose", dataExtrinsics);
  if(!query.exec()) {
      qDebug() << "ERROR: addPathExtrinsicsIntrinsics" << query.lastError().text();
      return false;
  } else {
      query.finish();
      return true;
  }
}

bool Database::addPathExtrinsicsIntrinsicsBatch(const std::vector<int>& ids,
                                                const std::vector<const std::string*>& paths,
                                                const std::vector<const Intrinsics*>& cameraIntrinsics,
                                                const std::vector<const Extrinsics*>& cameraExtrinsics,
                                                size_t size)
{
    if (size == 0)
    {
        size = ids.size();
    }

    if (ids.size() < size || paths.size() < size || cameraIntrinsics.size() < size || cameraExtrinsics.size() < size)
    {
        throw std::runtime_error("addPathExtrinsicsIntrinsics: Invalid argument sizes");
    }

    db.transaction();

    QSqlQuery query(db);
    query.prepare("INSERT INTO mytable (id, path, intrinsics, extrinsics) VALUES (:id, :path, :intrinsics, :extrinsics);");

    for (size_t i = 0; i < size; ++i)
    {
        QByteArray dataIntrinsics, dataExtrinsics;
        intrinsicsToByteArray(*cameraIntrinsics[i], dataIntrinsics);
        extrinsicsToByteArray(*cameraExtrinsics[i], dataExtrinsics);

        query.bindValue(":id", ids[i]);
        query.bindValue(":path", QString::fromStdString(*paths[i]));
        query.bindValue(":intrinsics", dataIntrinsics);
        query.bindValue(":extrinsics", dataExtrinsics);
        if (!query.execBatch())
        {
            db.rollback();
            std::cerr << "ERROR query failed in addPathExtrinsicsIntrinsicsBatch: " << query.lastError().text().toStdString() << std::endl;
            return false;
        }
    }

    db.commit();

    return true;
}

void Database::intrinsicsToByteArray(const Intrinsics& cameraIntrinsics, QByteArray& outArray)
{
    cv::Size imageSize = cameraIntrinsics.imageSize();
    cv::Point2d focalLength = cameraIntrinsics.focalLength();
    cv::Point2d principalPoint = cameraIntrinsics.principalPoint();
    std::vector<double> distorionCoefficients = cameraIntrinsics.distorionCoefficients();

    QDataStream streamIntrinsics(&outArray, QIODevice::WriteOnly);
    streamIntrinsics << QByteArray((const char *)&imageSize, sizeof (imageSize));
    streamIntrinsics << QByteArray((const char *)&focalLength, sizeof(focalLength));
    streamIntrinsics << QByteArray((const char *)&principalPoint, sizeof(principalPoint));
    streamIntrinsics << QByteArray((const char *)distorionCoefficients.data(), sizeof(double) * distorionCoefficients.size());
}

void Database::extrinsicsToByteArray(const Extrinsics& cameraExtrinsics, QByteArray& outArray)
{
    Extrinsics::TransformationDirection direction = Extrinsics::TransformationDirection::Ref2Local;

    cv::Vec3d rotation = cameraExtrinsics.getRotationRodrigues(direction);
    cv::Vec3d translation = cameraExtrinsics.getTranslation(direction);

    QDataStream streamExtrinsics(&outArray, QIODevice::WriteOnly);
    streamExtrinsics << (double)rotation[0];
    streamExtrinsics << (double)rotation[1];
    streamExtrinsics << (double)rotation[2];
    streamExtrinsics << (double)translation[0];
    streamExtrinsics << (double)translation[1];
    streamExtrinsics << (double)translation[2];
}
