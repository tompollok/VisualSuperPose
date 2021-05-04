#include "colmapimporter.h"
#include <opencv2/opencv.hpp>

#include <QDir>
#include <QTextStream>
#include <QDataStream>
#include <QElapsedTimer>
#include <QCollator>
#include <future>
#include <unordered_map>
#include <omp.h>

#include "types/worldpoint.h"
#include "types/image.h"
#include "types/extrinsics.h"
#include "utils/iohelpers.h"
enum class ColmapIntrinsicsType
{
    SIMPLE_PINHOLE,
    PINHOLE,
    SIMPLE_RADIAL,
    RADIAL,
    OPENCV,
    FULL_OPENCV
};

int intrinsicsTypeIDs[] =             {0,                1,         2,               3,        4,        6};
const char* intrinsicsTypeStrings[] = {"SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV"};


ColmapImporter::ColmapImporter()
{

}

bool ColmapImporter::importImages(
        const QString &mp,
        const QString &ip,
        std::vector<std::shared_ptr<Image> > &images)
{
    QString modelPath = IOHelpers::appendSlash(mp);
    QString imagePath = IOHelpers::appendSlash(ip);
    mIntrinsics.clear();
    if (IOHelpers::existsFile(modelPath + "cameras.bin") &&
             IOHelpers::existsFile(modelPath + "images.bin") &&
             IOHelpers::existsFile(modelPath + "points3D.bin")) {
        if (!importCamerasBin(modelPath + "cameras.bin"))
            return false;
        if (!importImagesWithoutCorBin(modelPath + "images.bin", imagePath, images))
            return false;
        return true;
    }
    else if (IOHelpers::existsFile(modelPath + "cameras.txt") &&
        IOHelpers::existsFile(modelPath + "images.txt") &&
        IOHelpers::existsFile(modelPath + "points3D.txt"))
    {
        if (!importCamerasText(modelPath + "cameras.txt"))
            return false;
        if (!importImagesWithoutCorText(modelPath + "images.txt", imagePath, images))
            return false;
        return true;
    }

    return false;
}

bool ColmapImporter::importImagesFromSubdirs(
        const QString &dirP,
        std::vector<std::shared_ptr<Image> > &images)
{
    QString modelPath = IOHelpers::appendSlash(dirP);
    QDir dir(dirP);
    QFileInfoList dirs = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);

    std::vector<std::future<std::vector<std::shared_ptr<Image>>>> ftrs;
    for (const auto & d : dirs) {
        ftrs.push_back(std::async( std::launch::async, [](QString modelpath, QString imagepath){
            std::vector<std::shared_ptr<Image>> imagelist;
            ColmapImporter importer;
            importer.importImages(modelpath, imagepath, imagelist);
            return imagelist;
        }, d.filePath() + "/model", d.filePath() + "/images"));
    }
    for (auto &ftr : ftrs) {
        std::vector<std::shared_ptr<Image>> tempList = ftr.get();
        images.insert(images.end(), tempList.begin(), tempList.end());
    }
    return true;
}

bool ColmapImporter::loadEvalForImages(const QString &cornellRoot,
                                       std::vector<std::shared_ptr<Image>> &images)
{
    QString r = IOHelpers::appendSlash(cornellRoot);
    return loadEvalForImages(r + "Alamo/model/",
                             r + "Ellis_Island/model/",
                             r + "Gendarmenmarkt/model/",
                             r + "Madrid_Metropolis/model/",
                             r + "Montreal_Notre_Dame/model/",
                             r + "NYC_Library/model/",
                             r + "Piazza_del_Popolo/model/",
                             r + "Piccadilly/model/",
                             r + "Roman_Forum/model/",
                             r + "Tower_of_London/model/",
                             r + "Trafalgar/model/",
                             r + "Union_Square/model/",
                             r + "Vienna_Cathedral/model/",
                             r + "Yorkminster/model/", images);
}

bool ColmapImporter::loadEvalForImages(
        const QString &alamoModelDir,
        const QString &ellisIslandModelDir,
        const QString &gendarmenmarktModelDir,
        const QString &madridMetropolisModelDir,
        const QString &montrealNotreDameModelDir,
        const QString &nycLibraryModelDir,
        const QString &piazzaDelPopoloModelDir,
        const QString &piccadillyModelDir,
        const QString &romanForumModelDir,
        const QString &towerOfLondonModelDir,
        const QString &trafalgarModelDir,
        const QString &unionSquareModelDir,
        const QString &viennaCathedralModelDir,
        const QString &yorkminsterModelDir,
        std::vector<std::shared_ptr<Image> > &images)
{
    std::vector<std::shared_ptr<Image>> alamoImages, ellisIslandImages,
            gendarmenmarktImages, madridMetropolisImages, montrealNotreDameImages,
            nycLibraryImages, piazzaDelPopoloImages, piccadillyImages,
            romanForumImages, towerOfLondonImages, trafalgarImages,
            unionSquareImages, viennaCathedralImages, yorkminsterImages;

    std::vector<TempImage> alamoTempImages, ellisIslandTempImages,
            gendarmenmarktTempImages, madridMetropolisTempImages,
            montrealNotreDameTempImages, nycLibraryTempImages,
            piazzaDelPopoloTempImages, piccadillyTempImages,
            romanForumTempImages, towerOfLondonTempImages,
            trafalgarTempImages, unionSquareTempImages,
            viennaCathedralTempImages, yorkminsterTempImages;
    auto ftralamo = std::async( std::launch::async, [](QString alamoModelDir){
        std::vector<TempImage> alamoTempImages;
        if (!loadImagePoints(alamoModelDir + "images.txt", alamoTempImages))
            std::cout << "WARNING: \"" << alamoModelDir.toStdString() << "images.txt\" not found. Evaluation for Alamo will probably not work." << std::endl;
        return alamoTempImages;
    }, alamoModelDir);

    auto ftrellis = std::async( std::launch::async, [](QString ellisIslandModelDir){
        std::vector<TempImage> ellisIslandTempImages;
        if (!loadImagePoints(ellisIslandModelDir + "images.txt", ellisIslandTempImages))
            std::cout << "WARNING: \"" << ellisIslandModelDir.toStdString() << "images.txt\" not found. Evaluation for Ellis Island will probably not work." << std::endl;
        return ellisIslandTempImages;
    }, ellisIslandModelDir);
    auto ftrgend = std::async( std::launch::async, [](QString gendarmenmarktModelDir){
        std::vector<TempImage> gendarmenmarktTempImages;
        if (!loadImagePoints(gendarmenmarktModelDir + "images.txt", gendarmenmarktTempImages))
            std::cout << "WARNING: \"" << gendarmenmarktModelDir.toStdString() << "images.txt\" not found. Evaluation for Gendarmenmarkt will probably not work." << std::endl;
        return gendarmenmarktTempImages;
    }, gendarmenmarktModelDir);

    auto ftrmadrid = std::async( std::launch::async, [](QString madridMetropolisModelDir){
        std::vector<TempImage> madridMetropolisTempImages;
        if (!loadImagePoints(madridMetropolisModelDir + "images.txt", madridMetropolisTempImages))
            std::cout << "WARNING: \"" << madridMetropolisModelDir.toStdString() << "images.txt\" not found. Evaluation for Madrid Metropolis will probably not work." << std::endl;
        return madridMetropolisTempImages;
    }, madridMetropolisModelDir);

    auto ftrmontreal = std::async( std::launch::async, [](QString montrealNotreDameModelDir){
        std::vector<TempImage> montrealNotreDameTempImages;
        if (!loadImagePoints(montrealNotreDameModelDir + "images.txt", montrealNotreDameTempImages))
            std::cout << "WARNING: \"" << montrealNotreDameModelDir.toStdString() << "images.txt\" not found. Evaluation for Montreal Notre Dame will probably not work." << std::endl;
        return montrealNotreDameTempImages;
    }, montrealNotreDameModelDir);

    auto ftrnyc = std::async( std::launch::async, [](QString nycLibraryModelDir){
        std::vector<TempImage> nycLibraryTempImages;
        if (!loadImagePoints(nycLibraryModelDir + "images.txt", nycLibraryTempImages))
            std::cout << "WARNING: \"" << nycLibraryModelDir.toStdString() << "images.txt\" not found. Evaluation for NYC Library will probably not work." << std::endl;
        return nycLibraryTempImages;
    }, nycLibraryModelDir);

    auto ftrpiazza = std::async( std::launch::async, [](QString piazzaDelPopoloModelDir){
        std::vector<TempImage> piazzaDelPopoloTempImages;
        if (!loadImagePoints(piazzaDelPopoloModelDir + "images.txt", piazzaDelPopoloTempImages))
            std::cout << "WARNING: \"" << piazzaDelPopoloModelDir.toStdString() << "images.txt\" not found. Evaluation for Piazza Del Popolo will probably not work." << std::endl;
        return piazzaDelPopoloTempImages;
    }, piazzaDelPopoloModelDir);

    auto ftrpiccadilly = std::async( std::launch::async, [](QString piccadillyModelDir){
        std::vector<TempImage> piccadillyTempImages;
        if (!loadImagePoints(piccadillyModelDir + "images.txt", piccadillyTempImages))
            std::cout << "WARNING: \"" << piccadillyModelDir.toStdString() << "images.txt\" not found. Evaluation for Piccadilly will probably not work." << std::endl;
        return piccadillyTempImages;
    }, piccadillyModelDir);

    auto ftrroman = std::async( std::launch::async, [](QString romanForumModelDir){
        std::vector<TempImage> romanForumTempImages;
        if (!loadImagePoints(romanForumModelDir + "images.txt", romanForumTempImages))
            std::cout << "WARNING: \"" << romanForumModelDir.toStdString() << "images.txt\" not found. Evaluation for Roman Forum will probably not work." << std::endl;
        return romanForumTempImages;
    }, romanForumModelDir);

    auto ftrtower = std::async( std::launch::async, [](QString towerOfLondonModelDir){
        std::vector<TempImage> towerOfLondonTempImages;
        if (!loadImagePoints(towerOfLondonModelDir + "images.txt", towerOfLondonTempImages))
            std::cout << "WARNING: \"" << towerOfLondonModelDir.toStdString() << "images.txt\" not found. Evaluation for Tower Of London will probably not work." << std::endl;
        return towerOfLondonTempImages;
    }, towerOfLondonModelDir);

    auto ftrtrafalgar = std::async( std::launch::async, [](QString trafalgarModelDir){
        std::vector<TempImage> trafalgarTempImages;
        if (!loadImagePoints(trafalgarModelDir + "images.txt", trafalgarTempImages))
            std::cout << "WARNING: \"" << trafalgarModelDir.toStdString() << "images.txt\" not found. Evaluation for Trafalgar will probably not work." << std::endl;
        return trafalgarTempImages;
    }, trafalgarModelDir);

    auto ftrunion = std::async( std::launch::async, [](QString unionSquareModelDir){
        std::vector<TempImage> unionSquareTempImages;
        if (!loadImagePoints(unionSquareModelDir + "images.txt", unionSquareTempImages))
            std::cout << "WARNING: \"" << unionSquareModelDir.toStdString() << "images.txt\" not found. Evaluation for Union Square will probably not work." << std::endl;
        return unionSquareTempImages;
    }, unionSquareModelDir);

    auto ftrvienna = std::async( std::launch::async, [](QString viennaCathedralModelDir){
        std::vector<TempImage> viennaCathedralTempImages;
        if (!loadImagePoints(viennaCathedralModelDir + "images.txt", viennaCathedralTempImages))
            std::cout << "WARNING: \"" << viennaCathedralModelDir.toStdString() << "images.txt\" not found. Evaluation for Vienna Cathedral will probably not work." << std::endl;
        return viennaCathedralTempImages;
    }, viennaCathedralModelDir);

    auto ftryork = std::async( std::launch::async, [](QString yorkminsterModelDir){
        std::vector<TempImage> yorkminsterTempImages;
        if (!loadImagePoints(yorkminsterModelDir + "images.txt", yorkminsterTempImages))
            std::cout << "WARNING: \"" << yorkminsterModelDir.toStdString() << "images.txt\" not found. Evaluation for Yorkminster will probably not work." << std::endl;
        return yorkminsterTempImages;
    }, yorkminsterModelDir);

    QHash<unsigned long long, std::shared_ptr<ColMapWorldPoint>> //id -> worldpoint
            alamoWorldPoints, ellisIslandWorldPoints,
            gendarmenmarktWorldPoints, madridMetropolisWorldPoints,
            montrealNotreDameWorldPoints, nycLibraryWorldPoints,
            piazzaDelPopoloWorldPoints, piccadillyWorldPoints,
            romanForumWorldPoints, towerOfLondonWorldPoints,
            trafalgarWorldPoints, unionSquareWorldPoints,
            viennaCathedralWorldPoints, yorkminsterWorldPoints;

    auto ftralamo2 = std::async( std::launch::async, [](QString alamoModelDir){
        QHash<unsigned long long, std::shared_ptr<ColMapWorldPoint>> alamoWorldPoints;
        if (!loadWorldPoints(alamoModelDir + "points3D.txt", alamoWorldPoints))
            std::cout << "WARNING: \"" << alamoModelDir.toStdString() << "points3D.txt\" not found. Evaluation for Alamo will probably not work." << std::endl;
        return alamoWorldPoints;
    }, alamoModelDir);

    auto ftrellis2 = std::async( std::launch::async, [](QString ellisIslandModelDir){
        QHash<unsigned long long, std::shared_ptr<ColMapWorldPoint>> ellisIslandWorldPoints;
        if (!loadWorldPoints(ellisIslandModelDir + "points3D.txt", ellisIslandWorldPoints))
            std::cout << "WARNING: \"" << ellisIslandModelDir.toStdString() << "points3D.txt\" not found. Evaluation for Ellis Island will probably not work." << std::endl;
        return ellisIslandWorldPoints;
    }, ellisIslandModelDir);

    auto ftrgend2 = std::async( std::launch::async, [](QString gendarmenmarktModelDir){
        QHash<unsigned long long, std::shared_ptr<ColMapWorldPoint>> gendarmenmarktWorldPoints;
        if (!loadWorldPoints(gendarmenmarktModelDir + "points3D.txt", gendarmenmarktWorldPoints))
            std::cout << "WARNING: \"" << gendarmenmarktModelDir.toStdString() << "points3D.txt\" not found. Evaluation for Gendarmenmarkt will probably not work." << std::endl;
        return gendarmenmarktWorldPoints;
    }, gendarmenmarktModelDir);

    auto ftrmadrid2 = std::async( std::launch::async, [](QString madridMetropolisModelDir){
        QHash<unsigned long long, std::shared_ptr<ColMapWorldPoint>> madridMetropolisWorldPoints;
        if (!loadWorldPoints(madridMetropolisModelDir + "points3D.txt", madridMetropolisWorldPoints))
            std::cout << "WARNING: \"" << madridMetropolisModelDir.toStdString() << "points3D.txt\" not found. Evaluation for Madrid Metropolis will probably not work." << std::endl;
        return madridMetropolisWorldPoints;
    }, madridMetropolisModelDir);

    auto ftrmontreal2 = std::async( std::launch::async, [](QString montrealNotreDameModelDir){
        QHash<unsigned long long, std::shared_ptr<ColMapWorldPoint>> montrealNotreDameWorldPoints;
        if (!loadWorldPoints(montrealNotreDameModelDir + "points3D.txt", montrealNotreDameWorldPoints))
            std::cout << "WARNING: \"" << montrealNotreDameModelDir.toStdString() << "points3D.txt\" not found. Evaluation for Montreal Notre Dame will probably not work." << std::endl;
        return montrealNotreDameWorldPoints;
    }, montrealNotreDameModelDir);

    auto ftrnyc2 = std::async( std::launch::async, [](QString nycLibraryModelDir){
        QHash<unsigned long long, std::shared_ptr<ColMapWorldPoint>> nycLibraryWorldPoints;
        if (!loadWorldPoints(nycLibraryModelDir + "points3D.txt", nycLibraryWorldPoints))
            std::cout << "WARNING: \"" << nycLibraryModelDir.toStdString() << "points3D.txt\" not found. Evaluation for NYC Library will probably not work." << std::endl;
        return nycLibraryWorldPoints;
    }, nycLibraryModelDir);

    auto ftrpiazza2 = std::async( std::launch::async, [](QString piazzaDelPopoloModelDir){
        QHash<unsigned long long, std::shared_ptr<ColMapWorldPoint>> piazzaDelPopoloWorldPoints;
        if (!loadWorldPoints(piazzaDelPopoloModelDir + "points3D.txt", piazzaDelPopoloWorldPoints))
            std::cout << "WARNING: \"" << piazzaDelPopoloModelDir.toStdString() << "points3D.txt\" not found. Evaluation for Piazza Del Popolo will probably not work." << std::endl;
        return piazzaDelPopoloWorldPoints;
    }, piazzaDelPopoloModelDir);

    auto ftrpiccadilly2 = std::async( std::launch::async, [](QString piccadillyModelDir){
        QHash<unsigned long long, std::shared_ptr<ColMapWorldPoint>> piccadillyWorldPoints;
        if (!loadWorldPoints(piccadillyModelDir + "points3D.txt", piccadillyWorldPoints))
            std::cout << "WARNING: \"" << piccadillyModelDir.toStdString() << "points3D.txt\" not found. Evaluation for Piccadilly will probably not work." << std::endl;
        return piccadillyWorldPoints;
    }, piccadillyModelDir);

    auto ftrroman2 = std::async( std::launch::async, [](QString romanForumModelDir){
        QHash<unsigned long long, std::shared_ptr<ColMapWorldPoint>> romanForumWorldPoints;
        if (!loadWorldPoints(romanForumModelDir + "points3D.txt", romanForumWorldPoints))
            std::cout << "WARNING: \"" << romanForumModelDir.toStdString() << "points3D.txt\" not found. Evaluation for Roman Forum will probably not work." << std::endl;
        return romanForumWorldPoints;
    }, romanForumModelDir);

    auto ftrtower2 = std::async( std::launch::async, [](QString towerOfLondonModelDir){
        QHash<unsigned long long, std::shared_ptr<ColMapWorldPoint>> towerOfLondonWorldPoints;
        if (!loadWorldPoints(towerOfLondonModelDir + "points3D.txt", towerOfLondonWorldPoints))
            std::cout << "WARNING: \"" << towerOfLondonModelDir.toStdString() << "points3D.txt\" not found. Evaluation for Tower Of London will probably not work." << std::endl;
        return towerOfLondonWorldPoints;
    }, towerOfLondonModelDir);

    auto ftrtrafalgar2 = std::async( std::launch::async, [](QString trafalgarModelDir){
        QHash<unsigned long long, std::shared_ptr<ColMapWorldPoint>> trafalgarWorldPoints;
        if (!loadWorldPoints(trafalgarModelDir + "points3D.txt", trafalgarWorldPoints))
            std::cout << "WARNING: \"" << trafalgarModelDir.toStdString() << "points3D.txt\" not found. Evaluation for Trafalgar will probably not work." << std::endl;
        return trafalgarWorldPoints;
    }, trafalgarModelDir);

    auto ftrunion2 = std::async( std::launch::async, [](QString unionSquareModelDir){
        QHash<unsigned long long, std::shared_ptr<ColMapWorldPoint>> unionSquareWorldPoints;
        if (!loadWorldPoints(unionSquareModelDir + "points3D.txt", unionSquareWorldPoints))
            std::cout << "WARNING: \"" << unionSquareModelDir.toStdString() << "points3D.txt\" not found. Evaluation for Union Square will probably not work." << std::endl;
        return unionSquareWorldPoints;
    }, unionSquareModelDir);

    auto ftrvienna2 = std::async( std::launch::async, [](QString viennaCathedralModelDir){
        QHash<unsigned long long, std::shared_ptr<ColMapWorldPoint>> viennaCathedralWorldPoints;
        if (!loadWorldPoints(viennaCathedralModelDir + "points3D.txt", viennaCathedralWorldPoints))
            std::cout << "WARNING: \"" << viennaCathedralModelDir.toStdString() << "points3D.txt\" not found. Evaluation for Vienna Cathedral will probably not work." << std::endl;
        return viennaCathedralWorldPoints;
    }, viennaCathedralModelDir);

    auto ftryork2 = std::async( std::launch::async, [](QString yorkminsterModelDir){
        QHash<unsigned long long, std::shared_ptr<ColMapWorldPoint>> yorkminsterWorldPoints;
        if (!loadWorldPoints(yorkminsterModelDir + "points3D.txt", yorkminsterWorldPoints))
            std::cout << "WARNING: \"" << yorkminsterModelDir.toStdString() << "points3D.txt\" not found. Evaluation for Yorkminster will probably not work." << std::endl;
        return yorkminsterWorldPoints;
    }, yorkminsterModelDir);

    alamoTempImages = ftralamo.get();
    ellisIslandTempImages = ftrellis.get();
    gendarmenmarktTempImages = ftrgend.get();
    madridMetropolisTempImages = ftrmadrid.get();
    montrealNotreDameTempImages = ftrmontreal.get();
    nycLibraryTempImages = ftrnyc.get();
    piazzaDelPopoloTempImages = ftrpiazza.get();
    piccadillyTempImages = ftrpiccadilly.get();
    romanForumTempImages = ftrroman.get();
    towerOfLondonTempImages = ftrtower.get();
    trafalgarTempImages = ftrtrafalgar.get();
    unionSquareTempImages = ftrunion.get();
    viennaCathedralTempImages = ftrvienna.get();
    yorkminsterTempImages = ftryork.get();

    alamoWorldPoints = ftralamo2.get();
    ellisIslandWorldPoints = ftrellis2.get();
    gendarmenmarktWorldPoints = ftrgend2.get();
    madridMetropolisWorldPoints = ftrmadrid2.get();
    montrealNotreDameWorldPoints = ftrmontreal2.get();
    nycLibraryWorldPoints = ftrnyc2.get();
    piazzaDelPopoloWorldPoints = ftrpiazza2.get();
    piccadillyWorldPoints = ftrpiccadilly2.get();
    romanForumWorldPoints = ftrroman2.get();
    towerOfLondonWorldPoints = ftrtower2.get();
    trafalgarWorldPoints = ftrtrafalgar2.get();
    unionSquareWorldPoints = ftrunion2.get();
    viennaCathedralWorldPoints = ftrvienna2.get();
    yorkminsterWorldPoints = ftryork2.get();

    QHash<QString, std::shared_ptr<Image>> imageHash;
    for (auto &img : images) {
        QFile file(QString::fromStdString(img->path));
        imageHash.insert(file.fileName().split("/").last(), img);
    }
    int miss = 0;
    for (auto &img : alamoTempImages) {
        if (!imageHash.contains(img.filename)) {
            miss++;
            continue;
        }
        std::shared_ptr<Image> image = imageHash.find(img.filename).value();
        for (auto &pnt : img.points) {
            std::shared_ptr<ColMapImagePoint> imagepoint(
                        new ColMapImagePoint({pnt.pos, image,
                                              alamoWorldPoints.find(pnt.id3D).value()}));
            imagepoint->worldpoint->imagepoints.push_back(imagepoint);
            image->imagepoints.push_back(imagepoint);
        }
    }
    for (auto &img : ellisIslandTempImages) {
        if (!imageHash.contains(img.filename)) {
            miss++;
            continue;
        }
        std::shared_ptr<Image> image = imageHash.find(img.filename).value();
        for (auto &pnt : img.points) {
            std::shared_ptr<ColMapImagePoint> imagepoint(
                        new ColMapImagePoint({pnt.pos, image,
                                              ellisIslandWorldPoints.find(pnt.id3D).value()}));
            imagepoint->worldpoint->imagepoints.push_back(imagepoint);
            image->imagepoints.push_back(imagepoint);
        }
    }
    for (auto &img : gendarmenmarktTempImages) {
        if (!imageHash.contains(img.filename)) {
            miss++;
            continue;
        }
        std::shared_ptr<Image> image = imageHash.find(img.filename).value();
        for (auto &pnt : img.points) {
            std::shared_ptr<ColMapImagePoint> imagepoint(
                        new ColMapImagePoint({pnt.pos, image,
                                              gendarmenmarktWorldPoints.find(pnt.id3D).value()}));
            imagepoint->worldpoint->imagepoints.push_back(imagepoint);
            image->imagepoints.push_back(imagepoint);
        }
    }
    for (auto &img : madridMetropolisTempImages) {
        if (!imageHash.contains(img.filename)) {
            miss++;
            continue;
        }
        std::shared_ptr<Image> image = imageHash.find(img.filename).value();
        for (auto &pnt : img.points) {
            std::shared_ptr<ColMapImagePoint> imagepoint(
                        new ColMapImagePoint({pnt.pos, image,
                                              madridMetropolisWorldPoints.find(pnt.id3D).value()}));
            imagepoint->worldpoint->imagepoints.push_back(imagepoint);
            image->imagepoints.push_back(imagepoint);
        }
    }
    for (auto &img : montrealNotreDameTempImages) {
        if (!imageHash.contains(img.filename)) {
            miss++;
            continue;
        }
        std::shared_ptr<Image> image = imageHash.find(img.filename).value();
        for (auto &pnt : img.points) {
            std::shared_ptr<ColMapImagePoint> imagepoint(
                        new ColMapImagePoint({pnt.pos, image,
                                              montrealNotreDameWorldPoints.find(pnt.id3D).value()}));
            imagepoint->worldpoint->imagepoints.push_back(imagepoint);
            image->imagepoints.push_back(imagepoint);
        }
    }
    for (auto &img : nycLibraryTempImages) {
        if (!imageHash.contains(img.filename)) {
            miss++;
            continue;
        }
        std::shared_ptr<Image> image = imageHash.find(img.filename).value();
        for (auto &pnt : img.points) {
            std::shared_ptr<ColMapImagePoint> imagepoint(
                        new ColMapImagePoint({pnt.pos, image,
                                              nycLibraryWorldPoints.find(pnt.id3D).value()}));
            imagepoint->worldpoint->imagepoints.push_back(imagepoint);
            image->imagepoints.push_back(imagepoint);
        }
    }
    for (auto &img : piazzaDelPopoloTempImages) {
        if (!imageHash.contains(img.filename)) {
            miss++;
            continue;
        }
        std::shared_ptr<Image> image = imageHash.find(img.filename).value();
        for (auto &pnt : img.points) {
            std::shared_ptr<ColMapImagePoint> imagepoint(
                        new ColMapImagePoint({pnt.pos, image,
                                              piazzaDelPopoloWorldPoints.find(pnt.id3D).value()}));
            imagepoint->worldpoint->imagepoints.push_back(imagepoint);
            image->imagepoints.push_back(imagepoint);
        }
    }
    for (auto &img : piccadillyTempImages) {
        if (!imageHash.contains(img.filename)) {
            miss++;
            continue;
        }
        std::shared_ptr<Image> image = imageHash.find(img.filename).value();
        for (auto &pnt : img.points) {
            std::shared_ptr<ColMapImagePoint> imagepoint(
                        new ColMapImagePoint({pnt.pos, image,
                                              piccadillyWorldPoints.find(pnt.id3D).value()}));
            imagepoint->worldpoint->imagepoints.push_back(imagepoint);
            image->imagepoints.push_back(imagepoint);
        }
    }
    for (auto &img : romanForumTempImages) {
        if (!imageHash.contains(img.filename)) {
            miss++;
            continue;
        }
        std::shared_ptr<Image> image = imageHash.find(img.filename).value();
        for (auto &pnt : img.points) {
            std::shared_ptr<ColMapImagePoint> imagepoint(
                        new ColMapImagePoint({pnt.pos, image,
                                              romanForumWorldPoints.find(pnt.id3D).value()}));
            imagepoint->worldpoint->imagepoints.push_back(imagepoint);
            image->imagepoints.push_back(imagepoint);
        }
    }
    for (auto &img : towerOfLondonTempImages) {
        if (!imageHash.contains(img.filename)) {
            miss++;
            continue;
        }
        std::shared_ptr<Image> image = imageHash.find(img.filename).value();
        for (auto &pnt : img.points) {
            std::shared_ptr<ColMapImagePoint> imagepoint(
                        new ColMapImagePoint({pnt.pos, image,
                                              towerOfLondonWorldPoints.find(pnt.id3D).value()}));
            imagepoint->worldpoint->imagepoints.push_back(imagepoint);
            image->imagepoints.push_back(imagepoint);
        }
    }
    for (auto &img : trafalgarTempImages) {
        if (!imageHash.contains(img.filename)) {
            miss++;
            continue;
        }
        std::shared_ptr<Image> image = imageHash.find(img.filename).value();
        for (auto &pnt : img.points) {
            std::shared_ptr<ColMapImagePoint> imagepoint(
                        new ColMapImagePoint({pnt.pos, image,
                                              trafalgarWorldPoints.find(pnt.id3D).value()}));
            imagepoint->worldpoint->imagepoints.push_back(imagepoint);
            image->imagepoints.push_back(imagepoint);
        }
    }
    for (auto &img : unionSquareTempImages) {
        if (!imageHash.contains(img.filename)) {
            miss++;
            continue;
        }
        std::shared_ptr<Image> image = imageHash.find(img.filename).value();
        for (auto &pnt : img.points) {
            std::shared_ptr<ColMapImagePoint> imagepoint(
                        new ColMapImagePoint({pnt.pos, image,
                                              unionSquareWorldPoints.find(pnt.id3D).value()}));
            imagepoint->worldpoint->imagepoints.push_back(imagepoint);
            image->imagepoints.push_back(imagepoint);
        }
    }
    for (auto &img : viennaCathedralTempImages) {
        if (!imageHash.contains(img.filename)) {
            miss++;
            continue;
        }
        std::shared_ptr<Image> image = imageHash.find(img.filename).value();
        for (auto &pnt : img.points) {
            std::shared_ptr<ColMapImagePoint> imagepoint(
                        new ColMapImagePoint({pnt.pos, image,
                                              viennaCathedralWorldPoints.find(pnt.id3D).value()}));
            imagepoint->worldpoint->imagepoints.push_back(imagepoint);
            image->imagepoints.push_back(imagepoint);
        }
    }
    for (auto &img : yorkminsterTempImages) {
        if (!imageHash.contains(img.filename)) {
            miss++;
            continue;
        }
        std::shared_ptr<Image> image = imageHash.find(img.filename).value();
        for (auto &pnt : img.points) {
            std::shared_ptr<ColMapImagePoint> imagepoint(
                        new ColMapImagePoint({pnt.pos, image,
                                              yorkminsterWorldPoints.find(pnt.id3D).value()}));
            imagepoint->worldpoint->imagepoints.push_back(imagepoint);
            image->imagepoints.push_back(imagepoint);
        }
    }
    std::cout << miss << " missed\n";
    return true;
}

bool ColmapImporter::loadImagePoints(
        const QString &file,
        std::vector<ColmapImporter::TempImage> &images)
{
    images.clear();

    QFile camsTxt(file);
    if (!camsTxt.open(QFile::ReadOnly))
        return false;

    QTextStream fileStream(&camsTxt);

    QString line;
    bool newCamLine = true;

    TempImage frame;
    while(!fileStream.atEnd()) {
        line = fileStream.readLine().trimmed();
        if (line.isEmpty() || line.startsWith("#"))
            continue;
        QTextStream lineStream(&line);
        if (newCamLine) {
            int id, cameraID;
            double qw, qx, qy, qz, tx, ty, tz;
            QString name;
            lineStream >> id >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> cameraID >> name;
            frame = TempImage({name,std::vector<TempImagePoint>()});
        } else {
            std::vector<cv::Point2f> imagePoints;
            std::vector<unsigned long long> worldPointIDs;
            while (!lineStream.atEnd()) {
                lineStream.skipWhiteSpace();
                if (lineStream.atEnd())
                    break;
                double x, y;
                unsigned long long id3D;
                lineStream >> x >> y >> id3D;
                if (id3D == static_cast<unsigned long long>(-1))
                    continue;
                frame.points.push_back(TempImagePoint({cv::Point2f(static_cast<float>(x), static_cast<float>(y)),id3D}));
            }
            images.push_back(frame);
        }
        newCamLine = !newCamLine;
    }

    return true;
}

bool ColmapImporter::loadWorldPoints(
        const QString &file,
        QHash<unsigned long long, std::shared_ptr<ColMapWorldPoint>> &worldpoints)
{
    worldpoints.clear();

    QFile txt(file);
    if (!txt.open(QFile::ReadOnly))
        return false;

    QTextStream fileStream(&txt);

    QString line;
    while(!fileStream.atEnd()) {
        line = fileStream.readLine().trimmed();
        if (line.isEmpty() || line.startsWith("#"))
            continue;
        unsigned long long id; cv::Point3d pos;
        int r, g, b;
        double error;
        QTextStream lineStream(&line);
        lineStream >> id >> pos.x >> pos.y >> pos.z >> r >> g >> b >> error;
        worldpoints.insert(id, std::shared_ptr<ColMapWorldPoint>(new ColMapWorldPoint{pos, std::vector<std::shared_ptr<ColMapImagePoint>>()}));
    }
    return true;
}

bool ColmapImporter::import3DPoints(
        const QString &mp,
        const QString &ip,
        std::vector<WorldPoint> &points3d)
{
    QString modelPath = IOHelpers::appendSlash(mp);
    QString imagePath = IOHelpers::appendSlash(ip);

    if (IOHelpers::existsFile(modelPath + "cameras.bin") &&
             IOHelpers::existsFile(modelPath + "images.bin") &&
             IOHelpers::existsFile(modelPath + "points3D.bin")) {
        if (!import3DPointsBin(modelPath + "points3D.bin", points3d))
            return false;
        return true;
    }
    else if (IOHelpers::existsFile(modelPath + "cameras.txt") &&
        IOHelpers::existsFile(modelPath + "images.txt") &&
        IOHelpers::existsFile(modelPath + "points3D.txt"))
    {
        if (!import3DPointsText(modelPath + "points3D.txt", points3d))
            return false;
        return true;
    }

    return false;
}

bool ColmapImporter::importCamera(int id, const QString& typeText, int typeBin, int width, int height, QTextStream* textStream, QDataStream* dataStream)
{
    std::vector<double> dist(8, 0.0);
    cv::Point2d focalLength;
    cv::Point2d principalPoint;

    if (typeText == intrinsicsTypeStrings[static_cast<int>(ColmapIntrinsicsType::SIMPLE_PINHOLE)]
            || typeBin == intrinsicsTypeIDs[static_cast<int>(ColmapIntrinsicsType::SIMPLE_PINHOLE)] )
    {
        double f, cx, cy;
        if (textStream)
            (*textStream) >> f >> cx >> cy;
        else
            (*dataStream) >> f >> cx >> cy;

        focalLength = cv::Point2d(f, f);
        principalPoint = cv::Point2d(cx, cy);
    }
    else if (typeText == intrinsicsTypeStrings[static_cast<int>(ColmapIntrinsicsType::PINHOLE)]
             || typeBin == intrinsicsTypeIDs[static_cast<int>(ColmapIntrinsicsType::PINHOLE)])
    {
        double fx, fy, cx, cy;
        if (textStream)
            (*textStream) >> fx >> fy >> cx >> cy;
        else
            (*dataStream) >> fx >> fy >> cx >> cy;

        focalLength = cv::Point2d(fx, fy);
        principalPoint = cv::Point2d(cx, cy);
    }
    else if (typeText == intrinsicsTypeStrings[static_cast<int>(ColmapIntrinsicsType::SIMPLE_RADIAL)]
             || typeBin == intrinsicsTypeIDs[static_cast<int>(ColmapIntrinsicsType::SIMPLE_RADIAL)])
    {
        double f, cx, cy, k;
        if (textStream)
            (*textStream) >> f >> cx >> cy >> k;
        else
            (*dataStream) >> f >> cx >> cy >> k;

        focalLength = cv::Point2d(f, f);
        principalPoint = cv::Point2d(cx, cy);
        dist[0] = k;
    }
    else if (typeText == intrinsicsTypeStrings[static_cast<int>(ColmapIntrinsicsType::RADIAL)]
             || typeBin == intrinsicsTypeIDs[static_cast<int>(ColmapIntrinsicsType::RADIAL)])
    {
        double f, cx, cy, k1 , k2;
        if (textStream)
            (*textStream) >> f >> cx >> cy >> k1 >> k2;
        else
            (*dataStream) >> f >> cx >> cy >> k1 >> k2;


        focalLength = cv::Point2d(f, f);
        principalPoint = cv::Point2d(cx, cy);
        cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
        dist[0] = k1; dist[1] = k2;
    }
    else if (typeText == intrinsicsTypeStrings[static_cast<int>(ColmapIntrinsicsType::OPENCV)]
             || typeBin == intrinsicsTypeIDs[static_cast<int>(ColmapIntrinsicsType::OPENCV)])
    {
        double fx, fy, cx, cy, k1 , k2, p1, p2;
        if (textStream)
            (*textStream) >> fx >> fy >> cx >> cy >> k1 >> k2 >> p1 >> p2;
        else
            (*dataStream) >> fx >> fy >> cx >> cy >> k1 >> k2 >> p1 >> p2;

        focalLength = cv::Point2d(fx, fy);
        principalPoint = cv::Point2d(cx, cy);
        dist[0] = k1; dist[1] = k2; dist[2] = p1; dist[3] = p2;
    }
    else if (typeText == intrinsicsTypeStrings[static_cast<int>(ColmapIntrinsicsType::FULL_OPENCV)]
             || typeBin == intrinsicsTypeIDs[static_cast<int>(ColmapIntrinsicsType::FULL_OPENCV)])
    {
        double fx, fy, cx, cy, k1 , k2, p1, p2, k3, k4, k5, k6;
        if (textStream)
            (*textStream) >> fx >> fy >> cx >> cy >> k1 >> k2 >> p1 >> p2 >> k3 >> k4 >> k5 >> k6;
        else
            (*dataStream) >> fx >> fy >> cx >> cy >> k1 >> k2 >> p1 >> p2 >> k3 >> k4 >> k5 >> k6;

        focalLength = cv::Point2d(fx, fy);
        principalPoint = cv::Point2d(cx, cy);
        dist = {k1, k2, p1, p2, k3, k4, k5, k6};
    }
    else
        return false;

    mIntrinsics.insert({id, Intrinsics(cv::Size(width, height), focalLength, principalPoint, cv::Mat(dist))});

    return true;
}

bool ColmapImporter::importCamerasText(const QString &file)
{
    QFile camsTxt(file);
    if (!camsTxt.open(QFile::ReadOnly))
        return false;

    QTextStream fileStream(&camsTxt);

    QString line;
    int id;
    QString type;
    int width, height;
    while(!fileStream.atEnd())
    {
        line = fileStream.readLine().trimmed();
        if (line.isEmpty() || line.startsWith("#"))
            continue;

        QTextStream lineStream(&line);
        lineStream >> id >> type >> width >> height;

        if (!importCamera(id, type, -1, width, height, &lineStream, nullptr))
            return false;
    }

    return true;
}

bool ColmapImporter::importCamerasBin(const QString &file)
{
    QFile camsTxt(file);
    if (!camsTxt.open(QFile::ReadOnly))
        return false;

    QDataStream fileStream(&camsTxt);
    fileStream.setByteOrder(QDataStream::LittleEndian);

    quint64 numCams;
    fileStream >> numCams;

    int id, type;
    quint64 width, height;
    for (size_t i = 0; i < numCams; i++)
    {
        fileStream >> id >> type >> width >> height;

        if (!importCamera(id, "", type, static_cast<int>(width), static_cast<int>(height), nullptr, &fileStream))
            return false;
    }

    return true;
}

bool ColmapImporter::importImage(int cameraID, const QString& name, const QString &imagePath, double qw, double qx,
                                 double qy, double qz, double tx, double ty, double tz, std::shared_ptr<Image>& outImage)
{
    Intrinsics intr;
    try {
        intr = mIntrinsics.at(cameraID);
    }  catch (...) {
        std::cout << "no intrinsics with id " << cameraID << std::endl;
        return false;
    }

    Extrinsics e(cv::Vec4d(qw, qx, qy, qz), cv::Vec3d(tx, ty, tz), Extrinsics::Ref2Local);

    outImage = std::make_shared<Image>();
    outImage->path = imagePath.toStdString() + name.toStdString();
    outImage->intrinsics = intr;
    outImage->extrinsics = e;
    return true;
}

bool ColmapImporter::importImagesWithoutCorText(
        const QString &file,
        const QString &imagePath,
        std::vector<std::shared_ptr<Image> > &images)
{
    QFile camsTxt(file);
    if (!camsTxt.open(QFile::ReadOnly))
        return false;

    QTextStream fileStream(&camsTxt);

    QString line;
    bool newCamLine = true;


    while(!fileStream.atEnd())
    {
        line = fileStream.readLine().trimmed();
        if (line.isEmpty() || line.startsWith("#"))
            continue;

        QTextStream lineStream(&line);

        if (newCamLine) // define new image
        {
            int id, cameraID;
            double qw, qx, qy, qz, tx, ty, tz;
            QString name;

            lineStream >> id >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> cameraID >> name;
            std::shared_ptr<Image> image = nullptr;

            if (!importImage(cameraID, name, imagePath, qw, qx, qy, qz, tx, ty, tz, image))
                return false;
            if (image != nullptr)
                images.push_back(image);

        }
        else // skip points for image
        {
            while (!lineStream.atEnd())
            {
                lineStream.skipWhiteSpace();
                if (lineStream.atEnd())
                    break;

                double x, y;
                unsigned long long id3D;
                lineStream >> x >> y >> id3D;

            }

        }

        newCamLine = !newCamLine;
    }

    return true;
}

bool ColmapImporter::importImagesWithoutCorBin(
        const QString &file,
        const QString &imagePath,
        std::vector<std::shared_ptr<Image> > &images)
{
    QFile binDataFile(file);
    if (!binDataFile.open(QFile::ReadOnly))
        return false;

    QDataStream* fileStream = nullptr;

    if(file.size() > 1024*1024*1024)
    {
        fileStream = new QDataStream(&binDataFile);
    }
    else
    {
        QByteArray data = binDataFile.readAll();
        fileStream = new QDataStream(data);
    }

    fileStream->setByteOrder(QDataStream::LittleEndian);

    quint64 numImages;
    *fileStream >> numImages;

    std::shared_ptr<Image> lastImage = nullptr;

    for (size_t i = 0; i < numImages; i++)
    {
        quint32 id;
        quint32 cameraID;
        double qw, qx, qy, qz, tx, ty, tz;
        QString name = "";

        *fileStream >> id >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> cameraID;

        char c;
        do {
            fileStream->readRawData(&c, 1);
            if (c != '\0')
                name.append(c);
        } while (c != '\0');


        if (!importImage( static_cast<int>(cameraID), name, imagePath, qw, qx, qy, qz,
                          tx, ty, tz, lastImage))
            return false;

        quint64 numPoints;
        *fileStream >> numPoints;

        for (size_t j = 0; j < numPoints; j++)
        {
            double x, y;
            quint64 id3D;
            *fileStream >> x >> y >> id3D;
        }
    }

    delete fileStream; fileStream = nullptr;
    binDataFile.close();

    return true;
}

bool ColmapImporter::import3DPointsText(const QString &file, std::vector<WorldPoint> &worldpoints)
{
    QFile txt(file);
    if (!txt.open(QFile::ReadOnly))
        return false;

    QTextStream fileStream(&txt);

    QString line;
    while(!fileStream.atEnd())
    {
        line = fileStream.readLine().trimmed();
        if (line.isEmpty() || line.startsWith("#"))
            continue;

        unsigned long long id; cv::Point3d pos;
        int r, g, b;
        double error;
        QTextStream lineStream(&line);
        lineStream >> id >> pos.x >> pos.y >> pos.z >> r >> g >> b >> error;
        worldpoints.push_back(WorldPoint{id, pos, cv::Vec3b(r, g, b), error});
    }

    return true;
}

bool ColmapImporter::import3DPointsBin(const QString &file, std::vector<WorldPoint> &worldpoints)
{
    QFile binDataFile(file);
    if (!binDataFile.open(QFile::ReadOnly))
        return false;

    QDataStream* fileStream = nullptr;

    if(file.size() > 1024*1024*1024)
    {
        fileStream = new QDataStream(&binDataFile);
    }
    else
    {
        QByteArray data = binDataFile.readAll();
        fileStream = new QDataStream(data);
    }

    fileStream->setByteOrder(QDataStream::LittleEndian);

    quint64 numPoints;
    *fileStream >> numPoints;

    for (size_t i = 0; i < numPoints; i++)
    {
        quint64 id;
        cv::Point3d pos;
        quint8 r, g, b;
        double error;

        *fileStream >> id >> pos.x >> pos.y >> pos.z;

        // color + error
        //fileStream.skipRawData(3 * sizeof(quint8) + sizeof(double));
        *fileStream >> r >> g >> b >> error;

        quint64 trackLength;
        *fileStream >> trackLength;

        // Tracks: tacklength * (camera_id (32bit) + point2D_id (32bit))
        fileStream->skipRawData(static_cast<int>(trackLength * (sizeof(quint32) + sizeof(quint32))));

        worldpoints.push_back(WorldPoint{id, pos, cv::Vec3b(r, g, b), error});
    }

    delete fileStream; fileStream = nullptr;

    return true;
}
