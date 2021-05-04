#include "DBImporterMT.h"

#include <thread>

#include <QDir>

#include "database.h"
#include "../import/colmapimporter.h"
#include "../utils/SiftHelpers.h"

void Queue::push(std::shared_ptr<Image> img)
{
    std::lock_guard<std::mutex> l(mLock);
    mQ.push(img);
}

std::shared_ptr<Image> Queue::pop()
{
    std::shared_ptr<Image> rtn = nullptr;

    {
        std::lock_guard<std::mutex> l(mLock);
        if (!mQ.empty())
        {
            rtn = mQ.front();
            mQ.pop();
        }
    }

    return rtn;
}

void Queue::setFinished()
{
    mFinished = true;
}

bool Queue::isFinished() const
{
    std::lock_guard<std::mutex> l(mLock);
    return mFinished && mQ.empty();
}

size_t Queue::size() const
{
    std::lock_guard<std::mutex> l(mLock);
    return mQ.size();
}


DBImporterMT::DBImporterMT(Database& db, int siftThreads)
 : mDB(db), mSiftThreads(siftThreads)
{

}

void checkAndPush(std::shared_ptr<Image>& img, Queue& q, int& id)
{
    if (QFile::exists(QString::fromStdString(img->path)))
    {
        q.push(img);
        id++;
    }
    else
    {
        std::string inf = "Warning: File not found \"" + img->path + "\"";
        std::cout << inf << std::endl;
    }
}

void runImportReconstructionSingle(const QString& imagesDir, const QString& reconstructionDir,
                                   Queue& out, int& id)
{
    std::vector<std::shared_ptr<Image>> images;
    ColmapImporter importer;
    importer.importImages(reconstructionDir, imagesDir, images);
    for (auto& i : images)
    {
        checkAndPush(i, out, id);
    }
}

void runImportReconstructionMultiple(const QString& rootDir, Queue& out, int& id)
{
    ColmapImporter importer;

    std::vector<std::shared_ptr<Image>> temp;

    QDir dir(rootDir);
    QFileInfoList dirs = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);

    for (const auto & d : dirs)
    {
        importer.importImages(d.filePath() + "/model", d.filePath() + "/images", temp);

        for (auto& t : temp)
        {
            checkAndPush(t, out, id);
        }
        temp.clear();
    }
}

void runLoadImage(Queue& in, Queue& out, size_t maxQSize)
{
    while (!in.isFinished())
    {
        auto i = in.pop();
        if (i == nullptr)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        i->loadImageGrayscale();
        while (out.size() >= maxQSize)
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        out.push(i);
    }
}

void runSIFT(Queue& in, Queue& out)
{
    while (!in.isFinished())
    {
        auto i = in.pop();
        if (i == nullptr)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        SiftHelpers::extractSiftFeatures(i->grayscaleImage, i->siftDescriptors, i->siftKeypoints);
        i->forgetGrayscaleImage();
        out.push(i);
    }
}

void saveBatch(std::vector<std::shared_ptr<Image>>& batch, Database& db, int& id)
{
    db.transaction();

    for (auto& i : batch)
    {
        db.addPathExtrinsicsIntrinsics(id, i->path, i->intrinsics, i->extrinsics);

        std::vector<cv::Point2f> points;
        cv::KeyPoint::convert(i->siftKeypoints, points);

        db.addKeyPointAndSift(id, points, i->siftDescriptors);
        id++;
    }

    db.commit();
}

void runSaveDB(Queue& in, Database& db, int batchSize, int& id)
{
    std::vector<std::shared_ptr<Image>> batch;
    batch.reserve(batchSize);
    while (!in.isFinished())
    {
        auto i = in.pop();
        if (i == nullptr)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        batch.push_back(i);
        if (batch.size() >= static_cast<size_t>(batchSize))
        {
            saveBatch(batch, db, id);
            batch.clear();
        }
    }

    if (!batch.empty())
    {
        saveBatch(batch, db, id);
    }
}

void runPrintStatusInfo(int& idsReconstructionLoaded, int& idsDBSaved)
{
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (idsReconstructionLoaded == 0)
            continue;

        std::cout << "\rProcessing " << idsDBSaved << "/" << idsReconstructionLoaded << std::flush;

        if (idsDBSaved >= idsReconstructionLoaded)
            break;
    }
    std::cout << std::endl;
}


void DBImporterMT::import(const QString& img, const QString& model)
{
    const int DBSaveBatchSize = 100;
    const int maxLoadedImages = 100;

    std::unique_ptr<std::thread> importerThread;
    std::vector<std::thread> siftThreads;

    int idsReconstructionLoaded = 0;
    int idsDBSaved = 0;

    if (model.isEmpty())
    {
        importerThread = std::make_unique<std::thread>(runImportReconstructionMultiple, img, std::ref(mImportQ), std::ref(idsReconstructionLoaded));
    }
    else
    {
        importerThread = std::make_unique<std::thread>(runImportReconstructionSingle, img, model, std::ref(mImportQ), std::ref(idsReconstructionLoaded));
    }

    std::thread imgLoadThread(runLoadImage, std::ref(mImportQ), std::ref(mSiftQ), maxLoadedImages);
    for (int i = 0; i < mSiftThreads; ++i)
    {
        siftThreads.push_back(std::thread(runSIFT, std::ref(mSiftQ), std::ref(mSaveQ)));
    }
    std::thread saveDBThread(runSaveDB, std::ref(mSaveQ), std::ref(mDB), DBSaveBatchSize, std::ref(idsDBSaved));

    std::thread consoleOutputThread(runPrintStatusInfo, std::ref(idsReconstructionLoaded), std::ref(idsDBSaved));

    importerThread->join();
    mImportQ.setFinished();
    imgLoadThread.join();
    mSiftQ.setFinished();
    for (auto& t : siftThreads)
    {
        t.join();
    }
    mSaveQ.setFinished();
    saveDBThread.join();
    consoleOutputThread.join();
}

void DBImporterMT::importRecursive(const QString &rootDir)
{
    import(rootDir, "");
}

void DBImporterMT::importSingleReconstruction(const QString &imageDir, const QString &reconstructionDir)
{
    import(imageDir, reconstructionDir);
}
