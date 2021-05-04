#include<iostream>
#include<QtCore>
#include<QFile>
#include <QFileInfo>
#include <QDir>
#include"csvHelper.h"


CSVHelper::CSVHelper(std::string savePath)
{
    this->savePath = QString::fromStdString(savePath);

    // link the file name to QFile
    this->file.setFileName(this->savePath);

    QFileInfo i(this->savePath);
    QDir d(i.path());
    if (!d.exists())
    {
        d.mkpath(i.path());
    }

    // open the file
    if(!file.open(QIODevice::WriteOnly))
        std::cout <<"Please check the path for saving CSV"<< std::endl;

    // link textstream to file
    this->toCSV.setDevice(&this->file);

    // write title
    this->writeTitle();
}


CSVHelper::~CSVHelper()
{
    // write all data from memory to disk
    this->writeFile();

    this->file.close();
}


// save data into memory
void CSVHelper::saveRetrieval(
        const std::string &queryImage,
        const int &gallerySize,
        const double &AP10,
        const int &rightAP10,
        const double &AP25,
        const int &rightAP25,
        const double &AP100,
        const int &rightAP100,
        const int &maxRightResults,
        const std::vector<std::pair<std::string, bool>> &retrievelImages)
{
    // delete the path, get the file name
    //std::string queryName = getOnlyImageName(queryImage);

    // check if the query exists
    // return -1 for not
    // return index
    int lineNum = findIfExists(queryImage);

    // the query image not exists
    if(lineNum == -1)
    {
        CSVRow temp;
        temp.queryName = queryImage;
        temp.gallerySize = gallerySize;
        temp.AP10 = AP10;
        temp.rightAP10 = rightAP10;
        temp.AP25 = AP25;
        temp.rightAP25 = rightAP25;
        temp.AP100 = AP100;
        temp.rightAP100 = rightAP100;
        temp.retrievelImages = retrievelImages;
        temp.maxRightResults = maxRightResults;
        this->CSV.push_back(temp);
    }

    // the query image is existed
    else
    {
        CSVRow &temp = CSV[lineNum];
        temp.gallerySize = gallerySize;
        temp.AP10 = AP10;
        temp.rightAP10 = rightAP10;
        temp.AP25 = AP25;
        temp.rightAP25 = rightAP25;
        temp.AP100 = AP100;
        temp.rightAP100 = rightAP100;
        temp.retrievelImages = retrievelImages;
        temp.maxRightResults = maxRightResults;
    }
}


// save data into memory
void CSVHelper::saveRegistration(
                      const std::string &queryImage,
                      const double &x,
                      const double &y,
                      const double &z,
                      const double &distance,
                      const double &row,
                      const double &pitch,
                      const double &yaw,
                      const double &winkel)
{
    // delete the path, get the file name
    //std::string queryName = getOnlyImageName(queryImage);

    // check if the query exists
    // return -1 for not
    // return index
    int lineNum = findIfExists(queryImage);

    // the query image not exists
    if(lineNum == -1)
    {
        CSVRow temp;
        temp.queryName = queryImage;
        temp.distanceX = x;
        temp.distanceY = y;
        temp.distanceZ = z;
        temp.distance = distance;
        temp.roll = row;
        temp.pitch = pitch;
        temp.yaw = yaw;
        temp.angle = winkel;
        this->CSV.push_back(temp);
    }

    // the query image is existed
    else
    {
        CSVRow &temp = CSV[lineNum];
        temp.distanceX = x;
        temp.distanceY = y;
        temp.distanceZ = z;
        temp.distance = distance;
        temp.roll = row;
        temp.pitch = pitch;
        temp.yaw = yaw;
        temp.angle = winkel;
    }
}

void CSVHelper::saveRegistration(std::shared_ptr<Image> image)
{
    saveRegistration(image->csvrow->queryName,
                     image->csvrow->distanceX,
                     image->csvrow->distanceY,
                     image->csvrow->distanceZ,
                     image->csvrow->distance,
                     image->csvrow->roll,
                     image->csvrow->pitch,
                     image->csvrow->yaw,
                     image->csvrow->angle);
}


void CSVHelper::saveRetrievalAndRegistration(std::shared_ptr<Image> image)
{
    // delete the path, get the file name
    //std::string queryName = getOnlyImageName(image->csvrow->queryName);

    // check if the query exists
    // return -1 for not
    // return index
    int lineNum = findIfExists(image->csvrow->queryName);

    // the query image not exists
    if(lineNum == -1)
    {
        CSV.push_back(*image->csvrow);
    }

    // the query image is existed
    else
    {
        CSV[lineNum] = *image->csvrow;
    }
}

void CSVHelper::saveRetrievalAndRegistration(std::vector<std::shared_ptr<Image>> imageList)
{
    for(const auto &image:imageList)
        this->saveRetrievalAndRegistration(image);
}

// 37 columns
void CSVHelper::writeTitle()
{
    toCSV << "sep=\t\n";
    toCSV << "query_name\t"
             "gallery_size\t"
             "x\t"
             "y\t"
             "z\t"
             "distance\t"
             "roll\t"
             "pitch\t"
             "yaw\t"
             "angle\t"
             "max_right_results\t"
             "AP10\t"
             "right_in_AP10\t"
             "AP25\t"
             "right_in_AP25\t"
             "AP100\t"
             "right_in_AP100\t"
             "retrieval_0\t"
             "is_right_0\t"
             "retrieval_1\t"
             "is_right_1\t"
             "retrieval_2\t"
             "is_right_2\t"
             "retrieval_3\t"
             "is_right_3\t"
             "retrieval_4\t"
             "is_right_4\t"
             "retrieval_5\t"
             "is_right_5\t"
             "retrieval_6\t"
             "is_right_6\t"
             "retrieval_7\t"
             "is_right_7\t"
             "retrieval_8\t"
             "is_right_8\t"
             "retrieval_9\t"
             "is_right_9\n";

}


void CSVHelper::csvSetVal(double val)
{
    if (val == std::numeric_limits<double>::max())
    {
        toCSV << "";
    }
    else
    {
        toCSV << val;
    }
    toCSV << sep;
}

void CSVHelper::csvSetVal(QString val)
{
    toCSV << val << sep;
}

void CSVHelper::csvSetVal(int val)
{
    toCSV << val << sep;
}

void CSVHelper::csvSetRetrieved(const std::vector<std::pair<std::string, bool>>& retrievelImages, int index)
{
    toCSV << (retrievelImages.size() > static_cast<size_t>(index) ? QString::fromStdString(retrievelImages[index].first) : "") << sep;
    toCSV << (retrievelImages.size() > static_cast<size_t>(index) ? retrievelImages[index].second : -1) << sep;
}

void CSVHelper::writeFile()
{
    for(const auto &line:CSV)
    {
        csvSetVal(QString::fromStdString(line.queryName));
        csvSetVal(line.gallerySize);
        csvSetVal(line.distanceX);
        csvSetVal(line.distanceY);
        csvSetVal(line.distanceZ);
        csvSetVal(line.distance);
        csvSetVal(line.roll);
        csvSetVal(line.pitch);
        csvSetVal(line.yaw);
        csvSetVal(line.angle);
        csvSetVal(line.maxRightResults);
        csvSetVal(line.AP10);
        csvSetVal(line.rightAP10);
        csvSetVal(line.AP25);
        csvSetVal(line.rightAP25);
        csvSetVal(line.AP100);
        csvSetVal(line.rightAP100);
        for (int i = 0; i < 10; ++i)
        {
            csvSetRetrieved(line.retrievelImages, i);
        }
        toCSV << "\n";
    }
}


// check if the query name exists in CSV vector
int CSVHelper::findIfExists(const std::string &queryName)
{
    for(size_t i=0; i<this->CSV.size(); i++)
    {
        if(queryName==CSV[i].queryName)
            return i;
    }
    return -1;
}


// get the name
std::string CSVHelper::getOnlyImageName(const std::string &path)
{
    QFileInfo imageFile(QString::fromStdString(path));
    QString imgName = imageFile.fileName();
    imgName = imgName.split(".", QString::SkipEmptyParts).at(0);
    return imgName.toStdString();
}
