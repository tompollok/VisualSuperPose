#pragma once

#include <QString>

#include "ppbafloc-core_export.h"
/**
 * @brief The IOHelpers class: some usefull functions.
 */
class PPBAFLOC_CORE_EXPORT IOHelpers
{
public:
    static QString appendSlash(const QString& input);
    static QString getFilenameFromPath(const QString& input);
    static QString filenameWithoutExtension(const QString& input);
    static bool isInDir(const QString& file, const QString& dir);
    static bool existsFile(const QString& file);
};
