//
// Created by johanna on 23.12.20.
//

#ifndef PPBAFLOC_DBHELPER_H
#define PPBAFLOC_DBHELPER_H

#include <string>
#include <memory>
#include <types/image.h>
#include "ppbafloc-core_export.h"
#include <database/database.h>

class PPBAFLOC_CORE_EXPORT DBHelper {
public:
    DBHelper(Database& db);

    /**
     * @brief filling up the database with extrinsic and intrinsic parameters for all images in path directory
     * @param path the directory to save extr and intr for; assumes that it contains an "images" folder that holds all images
     * and that it has a "model" folder that contains all the reconstruction information
     */
    void fillDatabaseExtrIntr(const std::string &path);

    /**
     * @brief filling up the database with extrinsic and intrinsic parameters for all images in imgDir
     * @param imgDir Directory that contains all image files
     * @param reconstrDir Directory that contains all reconstruction information
     */
    void fillDatabaseExtrIntr(const std::string &imgDir, const std::string &reconstrDir);

    /**
     * @brief Calculating SIFT keypoints and descriptors for all images contained in the database
     */
    void fillDatabaseSIFT();

    /**
     * @brief this function just calls fillDatabaseExtrIntr and fillDatabaseSIFT; convenience method
     * @param path the directory to save extr and intr for; assumes that it contains an "images" folder that holds all images
     * and that it has a "model" folder that contains all the reconstruction information
     */
    void fillDatabase(std::string &path);

    /**
     * @brief get image from database by its ID
     * @param id the identifier of the image in the database
     * @param img the image returned from the database
     */
    void getImage(int id, std::shared_ptr<Image> &img);

    /**
     * @brief gets all images contained in the database
     * @param images list of images returned from database
     */
    void getAllImages(std::vector<std::shared_ptr<Image>> &images);

    /**
     * @brief returns list of images from database given a list of ids
     * @param idList list of query image identifiers
     * @param images list of images returned from database
     */
    void getImagesByList(std::vector<int> &idList, std::vector<std::shared_ptr<Image>> &images);

    /**
     * @brief get an image from the database by its path
     * @param imgPath the path of the image to be returned
     * @param img the returned image from the database
     * @return true if operation was successful (imgPath found in database), false if operation was unsuccessful
     */
    bool getImageByPath(std::string &imgPath,  std::shared_ptr<Image> &img);


private:
    Database& mDB;
    QString mDBPath;

    bool checkIfPathInDB(const std::string &path);
};


#endif //PPBAFLOC_DBHELPER_H
