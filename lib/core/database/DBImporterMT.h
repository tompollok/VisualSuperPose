#pragma once

#include <memory>
#include <queue>

#include <QString>

#include "database.h"
#include "../types/image.h"

#include "ppbafloc-core_export.h"

/**
 * @brief Threadsafe FIFO queue
 */
class PPBAFLOC_CORE_EXPORT Queue
{
public:
    /**
     * @brief push an image to the front of the queue
     */
    void push(std::shared_ptr<Image> img);
    /**
     * @brief pop: remove image from the back of the queue
     * @return the element from the back or nullptr if queue is empty
     */
    std::shared_ptr<Image> pop();

    /**
     * @brief setFinished: Message mechanism to let consumer threads know that
     *                      no more elements are coming.
     */
    void setFinished();
    /**
     * @brief isFinished: Message mechanism to let consumer threads know that
     *                      no more elements are coming.
     * @return true if queue is empty and setFinished() was called by a producer thread
     */
    bool isFinished() const;

    /**
     * @return current number of elements in the queue
     */
    size_t size() const;

private:
    bool mFinished = false;
    std::queue<std::shared_ptr<Image>> mQ;
    mutable std::mutex mLock;
};

/**
 * @brief Utility class for filling the database
 *      optimized to use multiple threads
 */
class PPBAFLOC_CORE_EXPORT DBImporterMT
{
public:
    /**
     * @param db Database instance to fill
     * @param siftThreads number of threads to use for SIFT calculation,
     *               4 additional but lightweight threads are also created
     */
    DBImporterMT(Database& db, int siftThreads);

    /**
     * @brief importSingleReconstruction: Fill DB with a single COLMAP reconstruction
     * @param imageDir: images directory of COLMAP reconstruction
     * @param reconstructionDir: model directory of COLMAP reconstruction
     */
    void importSingleReconstruction(const QString& imageDir, const QString& reconstructionDir);

    /**
     * @brief importRecursive: Fill DB by providing a root folder containing multiple COLMAP reconstructions
     *                         reconstructions must have images in the "images" folder and
     *                         the sparse reconstruction in the "model" folder
     * @param rootDir: parent directory which is recursively searched for COLMAP files
     */
    void importRecursive(const QString& rootDir);

private:
    void import(const QString& img, const QString& model);

private:
    Queue mImportQ;
    Queue mSiftQ;
    Queue mSaveQ;

    Database& mDB;

    int mSiftThreads = 1;
};
