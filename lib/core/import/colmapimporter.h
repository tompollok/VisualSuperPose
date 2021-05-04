#ifndef COLMAPIMPORTER_H
#define COLMAPIMPORTER_H

#include <unordered_map>
#include "genericimporter.h"
#include <vector>
#include <QHash>

class QTextStream;
class QDataStream;
struct Image;
struct WorldPoint;

class PPBAFLOC_CORE_EXPORT ColmapImporter : GenericImporter
{
public:
    ColmapImporter();

    // GenericImporter interface
public:
    /**
     * @brief imports images with extrinsics, intrinsics and path
     *  !!! no correspondences are loaded as we cant use them !!!
     */
    bool importImages(
            const QString &modelPath,
            const QString &imagePath,
            std::vector<std::shared_ptr<Image>> &images) override;
    /**
     * @brief imports 3d points - not correspondences - for visualisation only
     */
    bool import3DPoints(
            const QString &modelPath,
            const QString &imagePath,
            std::vector<WorldPoint> &points3d) override;

    bool importImagesFromSubdirs(
            const QString &dir,
            std::vector<std::shared_ptr<Image> > &images);
    /**
     * @brief loadEvalForImages load colmap 2d-3d correspondences for evaluation
     * @param modelDirs of the dataset, if no images from dataset are in your imagelist,
     *  you can give an empty string instead. Needs a "/" at the end
     * @param images - that will be used for evaluation
     * @return if there was no error while loading
     */
    bool loadEvalForImages(
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
            std::vector<std::shared_ptr<Image>> &images //need to be loaded already with path
            );
    /**
     * @brief imports 2D-3D correspondences for images. !For evaluation only!
     */
    bool loadEvalForImages(const QString& cornellRoot, std::vector<std::shared_ptr<Image>>& images);
private:
    //for eval
    struct TempImagePoint { cv::Point2f pos; unsigned long long id3D; };
    struct TempImage { QString filename /*with ending */;  std::vector<TempImagePoint> points; };
    /**
     * @brief loadImagePoints loads 2D Points
     */
    bool static loadImagePoints(const QString &file, std::vector<TempImage> &images);
    /**
     * @brief loadImagePoints loads 3D Points
     */
    bool static loadWorldPoints(const QString &file, QHash<unsigned long long, std::shared_ptr<ColMapWorldPoint>> &worldpoints);
    //for eval end

    //for *.txt COLMAP reconstructions
    bool importCamerasText(const QString& file);
    bool import3DPointsText(const QString& file, std::vector<WorldPoint> &worldpoints);
    bool importImagesWithoutCorText(
            const QString& file,
            const QString &imagePath,
            std::vector<std::shared_ptr<Image> > &images);
    //for *.bin COLMAP reconstructions
    bool importCamerasBin(const QString& file);
    bool import3DPointsBin(const QString& file, std::vector<WorldPoint> &worldpoints);
    bool importImagesWithoutCorBin(
            const QString& file,
            const QString &imagePath,
            std::vector<std::shared_ptr<Image> > &images);
    bool importCamera(
            int id,
            const QString& typeText,
            int typeBin,
            int width,
            int height,
            QTextStream* textStream,
            QDataStream* dataStream);
    bool importImage(
            int cameraID,
            const QString& name,
            const QString &imagePath,
            double qw,  double qx,  double qy,  double qz,
            double tx, double ty, double tz,
            std::shared_ptr<Image> &outImage);


    std::unordered_map<int, Intrinsics> mIntrinsics;
};

#endif // COLMAPIMPORTER_H
