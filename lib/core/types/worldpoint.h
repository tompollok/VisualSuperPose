#ifndef WORLDPOINT_H
#define WORLDPOINT_H

#include <opencv2/core.hpp>
#include "ppbafloc-core_export.h"
/**
 * @brief The WorldPoint struct: Needed vor regviz visualization.
 */
struct PPBAFLOC_CORE_EXPORT WorldPoint
{
    uint64  id;
    cv::Point3d         pos;
    cv::Vec3b           color;
    double              error;
};

#endif // WORLDPOINT_H
