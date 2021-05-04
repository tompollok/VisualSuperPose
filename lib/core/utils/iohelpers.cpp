#include "iohelpers.h"

#include <QFileInfo>

QString IOHelpers::appendSlash(const QString& input)
{
    if (!input.isEmpty() && input[input.length() - 1] != '/')
    {
        return QString(input).append('/');
    }

    return input;
}

QString IOHelpers::getFilenameFromPath(const QString& input)
{
    QFileInfo inf(input);
    return inf.fileName();
}

QString IOHelpers::filenameWithoutExtension(const QString &input)
{
    return input.left(input.lastIndexOf("."));
}

bool IOHelpers::isInDir(const QString& file, const QString& dir)
{
    return file.startsWith(dir);
}

bool IOHelpers::existsFile(const QString& file)
{
    QFileInfo i(file);
    return i.exists();
}
