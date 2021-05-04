#include "genericimporter.h"

GenericImporter::GenericImporter()
{

}

bool GenericImporter::importImages(const QString &modelPath, const QString &imagePath, std::vector<std::shared_ptr<Image> > &images)
{
    return false;
}

bool GenericImporter::import3DPoints(const QString &modelPath, const QString &imagePath, std::vector<WorldPoint> &points3d)
{
    return false;
}
