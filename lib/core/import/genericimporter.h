#ifndef GENERICIMPORTER_H
#define GENERICIMPORTER_H

#include <QString>
#include <memory>
#include <vector>
#include <opencv2/core.hpp>
#include "types/image.h"
#include "types/worldpoint.h"
#include "ppbafloc-core_export.h"


class PPBAFLOC_CORE_EXPORT GenericImporter
{
public:
    GenericImporter();

    /**
     * @brief imports images with extrinsics, intrinsics and path
     */
    virtual bool importImages(
            const QString& modelPath,
            const QString& imagePath,
            std::vector<std::shared_ptr<Image>> &images);
    /**
     * @brief for debug visualisation if available
     */
    virtual bool import3DPoints(
            const QString& modelPath,
            const QString& imagePath,
            std::vector<WorldPoint> &points3d);
    //Maybe a function für correspondence import, if needed?



};

#endif // GENERICIMPORTER_H
