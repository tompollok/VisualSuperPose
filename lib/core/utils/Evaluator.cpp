#include "Evaluator.h"

#include <stdlib.h>
#include <numeric>

#include <QFileInfo>
#include <QFile>
#include <QMap>

#include "../core.h"
#include "../types/image.h"
#include "../import/colmapimporter.h"
#include "iohelpers.h"

namespace {
    bool sortByIndex(const std::tuple<int, std::string, std::string> &a,
                     const std::tuple<int, std::string, std::string> &b) {
        return std::get<0>(a) < std::get<0>(b);
    }

    QString getFullPathByIdentifier(const QString &evalDir, const QString identifier) {
        QString subDirs = "";
        subDirs.append(identifier[0]);
        subDirs.append("/");
        subDirs.append(identifier[1]);
        subDirs.append("/");
        subDirs.append(identifier[2]);
        subDirs.append("/");
        subDirs.append(identifier);
        subDirs.append(".jpg");

        QString resultDir = IOHelpers::appendSlash(evalDir).append(subDirs);
        return resultDir;
    }


    std::vector<std::shared_ptr<Image>> filterQueryImgFromList(const std::vector<std::shared_ptr<Image>> &imgList, const std::shared_ptr<Image> &queryImage) {
        std::vector<std::shared_ptr<Image>> filteredImages;
        for (const auto &img: imgList) {
            if (img->path != queryImage->path) {
                filteredImages.push_back(img);
            } else {
                //std::cout << "Query image in retrieved images. Removing image..." << std::endl;
            }
        }
        return filteredImages;
    }

    QString getOnlyImageName(const std::string &path) {
        QFileInfo imageFile(QString::fromStdString(path));
        QString imgName = imageFile.fileName();
        imgName = imgName.split(".", QString::SkipEmptyParts).at(0);
        return imgName;
    }

    int countNumPositives(std::vector<bool>& results) {
        int counter = 0;
        for (const auto &res: results)
            if (res) {
                counter++;
            }
        return counter;
    }

    std::vector<std::pair<std::string, bool>> getBoolPathList(std::vector<std::shared_ptr<Image>>& imageList, std::vector<bool>& predResults) {
        int counter = 0;
        std::vector<std::pair<std::string, bool>> imagePaths;
        for (size_t i = 0; i < imageList.size(); i++) {
            if (counter < 10) {
                imagePaths.push_back({imageList[i]->path, predResults[i]});
            }
            counter++;
        }
        return imagePaths;
    }

    std::vector<bool> createPredResults(const std::vector<std::tuple<int, std::string, std::string>> &pathLabelList, const std::string &gtLabel) {
        std::vector<bool> resultList;
        for (const auto &pathLabelPair: pathLabelList) {
            resultList.push_back(std::get<2>(pathLabelPair) == gtLabel);
        }
        return resultList;
    }
}


std::vector<std::pair<double, double>> Evaluator::calculatePRCurve(
        const std::vector<bool> &predictionResults,
        int numberPossiblePosSamples) {
    numberPossiblePosSamples = std::min(static_cast<int>(predictionResults.size()), numberPossiblePosSamples);
    std::vector<std::pair<double, double>> prCurve;
    double precision ;
    double recall;

    int truePositivesSum = 0;
    int falseNegativesSum = 0;
    int counter = 0;

    for (const auto &res: predictionResults) {
        if (res) {
            truePositivesSum += 1;
            //std::cout << "index correct is " << counter << std::endl;
        } else {
            falseNegativesSum += 1;
        }
        precision = double(truePositivesSum) / (truePositivesSum + falseNegativesSum);
        recall = double(truePositivesSum) / numberPossiblePosSamples;
        prCurve.push_back({recall, precision});
        counter++;
    }
    return prCurve;
}


double Evaluator::calculateMeanAP(std::vector<std::pair<double, double>> &prValues, std::vector<bool> &predictionResults, int numberPossiblePosSamples) {
        assert(predictionResults.size() == prValues.size());

        numberPossiblePosSamples = std::min(static_cast<int>(predictionResults.size()), numberPossiblePosSamples);
        double meanAP = 0;

        for (uint i = 0; i < prValues.size(); i++) {
            meanAP += prValues[i].second * int(predictionResults[i]);
        }
        meanAP = meanAP / numberPossiblePosSamples;
        return meanAP;
}

double Evaluator::calculateAPMutlipleQueries(const std::vector<double> &aps) {
    return (double(std::accumulate(aps.begin(), aps.end(), 0.0) / aps.size()));
}

void Evaluator::getGoodQueryImages(const std::string &csvPath, const std::string &evalImagesDir, int minNumberImages, int numQueryImages) {
    QFile file(QString::fromStdString(csvPath));
    QStringList lineList;
    QString line;
    int counter = 0;
    std::vector<QString> imgNames;

    if (!file.open(QIODevice::ReadOnly)) {
        std::cout << "Error opening csv train file1" << std::endl;
        return;
    }
    std::cout << "Printing possible query images from train clean csv" << std::endl;
    while (!file.atEnd()) {
        line = file.readLine();
        if (line.length() == 1 || line.isEmpty()) {
            continue;
        }
        if (line.split(',')[1].split(' ').size() >= minNumberImages) {
            int numberImages = line.split(',')[1].split(' ').size() - 1;
            int chosenIndex = rand() % numberImages;
            QString fullPath = getFullPathByIdentifier(QString::fromStdString(evalImagesDir), line.split(',')[1].split(' ')[chosenIndex]);

            if (IOHelpers::existsFile(fullPath)) {
                std::cout << fullPath.toStdString() << std::endl;
                counter++;
            }
        }
        if (counter > numQueryImages) {
            return;
        }
    }
}

double Evaluator::evaluateMultipleQueries(const std::string &csvPath, std::vector<std::vector<std::shared_ptr<Image>>> &imageLists, std::vector<std::shared_ptr<Image>> &queryList, int numRetrieved) {
    std::vector<double> mAPList;
    assert(imageLists.size() == queryList.size());
    for (uint i = 0; i < imageLists.size(); i++) {
        mAPList.push_back(this->evaluateRetrieval(csvPath, imageLists[i], queryList[i], numRetrieved));
    }
    return calculateAPMutlipleQueries(mAPList);
}


double Evaluator::evaluateRetrieval(const std::string &csvPath, const std::vector<std::shared_ptr<Image>> &imageList, std::shared_ptr<Image> &queryImg, int numRetrieved) {

    QString queryName = getOnlyImageName(queryImg->path);

    QFile file(QString::fromStdString(csvPath));
    QStringList lineList;
    QString gtLabel;
    std::vector<QString> labelList;
    QString line;

    std::vector<std::shared_ptr<Image>> filteredList = filterQueryImgFromList(imageList, queryImg);
    std::vector<std::tuple<int, std::string, std::string>> pathLabelList;

    if (numRetrieved < 0) numRetrieved = filteredList.size();

    std::vector<std::pair<int, QString>> imgNames;
    int index = 0;
    for (const auto &img: filteredList) {
        if (imgNames.size() < numRetrieved) {
            imgNames.push_back({index, getOnlyImageName(img->path)});
            index++;
        }
    }

    if (!file.open(QIODevice::ReadOnly)) {
        std::cout << "Error opening csv train file2" << std::endl;
        return {};
    }

    int numberPossiblePosSamples = 0;
    while (!file.atEnd()) {
        line = file.readLine();
        if (line.length() == 1 || line.isEmpty()) {
            continue;
        }
        if (line.contains(queryName, Qt::CaseSensitive))  {
            gtLabel = line.split(',')[0];
            numberPossiblePosSamples = line.split(',')[1].split(' ').size();
            std::cout << "GT Label found: " << gtLabel.toStdString() << std::endl;
        }
        createLabelListOneLine(line, imgNames, pathLabelList);
    }

    sort(pathLabelList.begin(), pathLabelList.end(), sortByIndex);
    numberPossiblePosSamples = std::min(numberPossiblePosSamples, int(imgNames.size()));

    queryImg->csvrow->maxRightResults = numberPossiblePosSamples;
    auto predResults = createPredResults(pathLabelList, gtLabel.toStdString());

    queryImg->csvrow->retrievelImages = getBoolPathList(filteredList, predResults);

    auto prCurve = calculatePRCurve(predResults, numberPossiblePosSamples);
    double meanAP = calculateMeanAP(prCurve, predResults, numberPossiblePosSamples);
    int numPositives = countNumPositives(predResults);
    std::cout << "meanAP is " << meanAP << " for query image " << queryImg->path << std::endl;

    if(numRetrieved == 10) {
        queryImg->csvrow->AP10 = meanAP;
        queryImg->csvrow->rightAP10 = numPositives;
    } else if (numRetrieved == 25) {
        queryImg->csvrow->AP25 = meanAP;
        queryImg->csvrow->rightAP25 = numPositives;
    } else if (numRetrieved == 100) {
        queryImg->csvrow->AP100 = meanAP;
        queryImg->csvrow->rightAP100 = numPositives;
    }
    queryImg->csvrow->queryName = queryImg->path;
    return meanAP;
}

std::vector<bool> Evaluator::evaluateRetrievalLandmarks(
        const std::string &csvPath,
        std::vector<std::shared_ptr<Image> > &imageList,
        std::shared_ptr<Image> &queryImg)
{
    QString queryName = getOnlyImageName(queryImg->path);

    QFile file(QString::fromStdString(csvPath));
    QStringList lineList;
    QString gtLabel;
    std::vector<QString> labelList;
    QString line;

    std::vector<std::shared_ptr<Image>> filteredList = filterQueryImgFromList(imageList, queryImg);
    imageList = filteredList;
    if (filteredList.size() < 100) {
        std::cout
                << "WARNING: Not enough retrieved Images for Evaluation. "
                << "Results are not representative and shouldn't be used.";
    }
    std::vector<std::tuple<int, std::string, std::string>> pathLabelList;

    std::vector<std::pair<int, QString>> imgNames;
    int index = 0;
    for (const auto &img: filteredList) {
        if (imgNames.size() < filteredList.size()) {
            imgNames.push_back({index, getOnlyImageName(img->path)});
            index++;
        }
    }

    if (!file.open(QIODevice::ReadOnly)) {
        std::cout << "Error opening csv train file3" << std::endl;
        return {};
    }

    int possivleRetCount = 0;
    QStringList list;
    while (!file.atEnd()) {
        line = file.readLine();
        if (line.length() == 1 || line.isEmpty()) {
            continue;
        }
        if (line.contains(queryName, Qt::CaseSensitive))  {
            list = line.split(',');
            gtLabel = list.takeAt(0);
            list = list[0].split(' ');
            possivleRetCount = list.size();
            std::cout << "GT Label found: " << gtLabel.toStdString() << std::endl;
            break;
        }
        createLabelListOneLine(line, imgNames, pathLabelList);
    }
    std::vector<bool> isGoodRefFrame;
    for (auto &img : filteredList) {
        QString refName = getOnlyImageName(img->path);
        if (list.contains(refName)) {
            isGoodRefFrame.push_back(true);
        } else {
            isGoodRefFrame.push_back(false);
        }
    }

    auto ret = isGoodRefFrame;
    isGoodRefFrame.resize(std::min(isGoodRefFrame.size(),
                                   static_cast<size_t>(100)));
    std::vector<std::pair<double, double>> rpValues =
            Evaluator::calculatePRCurve(isGoodRefFrame, possivleRetCount);
    double meanAP100 =
            Evaluator::calculateMeanAP(rpValues, isGoodRefFrame, possivleRetCount);
    int goodresults100 = 0;
    for (auto i : isGoodRefFrame)
        if (i)
            goodresults100++;
    isGoodRefFrame.resize(std::min(isGoodRefFrame.size(),
                                   static_cast<size_t>(25)));
    rpValues =
            Evaluator::calculatePRCurve(isGoodRefFrame, possivleRetCount);
    double meanAP25 =
            Evaluator::calculateMeanAP(rpValues, isGoodRefFrame, possivleRetCount);
    int goodresults25 = 0;
    for (auto i : isGoodRefFrame)
        if (i)
            goodresults25++;
    isGoodRefFrame.resize(std::min(isGoodRefFrame.size(),
                                   static_cast<size_t>(10)));
    rpValues =
            Evaluator::calculatePRCurve(isGoodRefFrame, possivleRetCount);
    double meanAP10 =
            Evaluator::calculateMeanAP(rpValues, isGoodRefFrame, possivleRetCount);
    int goodresults10 = 0;
    for (auto i : isGoodRefFrame)
        if (i)
            goodresults10++;
    std::cout << "results for " << queryImg->path << ":\n";
    std::cout << "goodresults: " << goodresults10 << "/10\n";
    std::cout << "meanAP10: " << meanAP10 << "\n";
    std::cout << "goodresults: " << goodresults25 << "/25\n";
    std::cout << "meanAP25: " << meanAP25 << "\n";
    std::cout << "goodresults: " << goodresults100 << "/100\n";
    std::cout << "meanAP100: " << meanAP100 << "\n";

    filteredList.resize(10);
    for (size_t i = 0; i < filteredList.size(); i++) {
        queryImg->csvrow->retrievelImages.push_back(std::pair<std::string, bool>(filteredList.at(i)->path, isGoodRefFrame.at(i)));
    }
    queryImg->csvrow->AP10 = meanAP10;
    queryImg->csvrow->rightAP10 = goodresults10;
    queryImg->csvrow->AP25 = meanAP25;
    queryImg->csvrow->rightAP25 = goodresults25;
    queryImg->csvrow->AP100 = meanAP100;
    queryImg->csvrow->rightAP100 = goodresults100;
    queryImg->csvrow->queryName = queryImg->path;
    queryImg->csvrow->maxRightResults = std::min(possivleRetCount, 100);
    return ret;
}

std::vector<bool> Evaluator::evaluateRetrievalCornell(std::vector<std::shared_ptr<Image> > &imageList,
        std::shared_ptr<Image> &queryImg)
{
    QStringList lineList;
    std::vector<QString> labelList;

    std::vector<std::shared_ptr<Image>> filteredList = filterQueryImgFromList(imageList, queryImg);
    imageList = filteredList;
    std::vector<std::tuple<int, std::string, std::string>> pathLabelList;

    if (filteredList.size() < 100) {
        std::cout
                << "WARNING: Not enough retrieved Images for Evaluation. "
                << "Results are not representative and shouldn't be used.";
    }
    int threshold = 10;
    QMap<std::shared_ptr<Image>, int> map;
    for (auto imgpnt : queryImg->imagepoints) {
        for (auto refpnt : imgpnt->worldpoint->imagepoints) {
            if (refpnt->image == queryImg)
                continue;
            if (map.contains(refpnt->image)) {
                map.find(refpnt->image).value()++;
            } else {
                map.insert(refpnt->image, 1);
            }
        }
    }
    QMapIterator<std::shared_ptr<Image>, int> it(map);
    int possivleRetCount = 0;
    while (it.hasNext()) {
        it.next();
        if (it.value() >= threshold) {
            possivleRetCount++;
        }
    }
    std::cout << "Possible retrievable ref images: " << possivleRetCount << "\n";
    //find corrs to retrieved images
    std::vector<bool> isGoodRefFrame;
    for (auto &img : filteredList) {
        if (IsGoodReferenceFrame(queryImg, img, threshold)) {
            isGoodRefFrame.push_back(true);
        } else {
            isGoodRefFrame.push_back(false);
        }
    }
    auto ret = isGoodRefFrame;
    isGoodRefFrame.resize(std::min(isGoodRefFrame.size(),
                                   static_cast<size_t>(100)));
    std::vector<std::pair<double, double>> rpValues =
            Evaluator::calculatePRCurve(isGoodRefFrame, possivleRetCount);
    double meanAP100 =
            Evaluator::calculateMeanAP(rpValues, isGoodRefFrame, possivleRetCount);
    int goodresults100 = 0;
    for (auto i : isGoodRefFrame)
        if (i)
            goodresults100++;
    isGoodRefFrame.resize(std::min(isGoodRefFrame.size(),
                                   static_cast<size_t>(25)));
    rpValues =
            Evaluator::calculatePRCurve(isGoodRefFrame, possivleRetCount);
    double meanAP25 =
            Evaluator::calculateMeanAP(rpValues, isGoodRefFrame, possivleRetCount);
    int goodresults25 = 0;
    for (auto i : isGoodRefFrame)
        if (i)
            goodresults25++;
    isGoodRefFrame.resize(std::min(isGoodRefFrame.size(),
                                   static_cast<size_t>(10)));
    rpValues =
            Evaluator::calculatePRCurve(isGoodRefFrame, possivleRetCount);
    double meanAP10 =
            Evaluator::calculateMeanAP(rpValues, isGoodRefFrame, possivleRetCount);
    int goodresults10 = 0;
    for (auto i : isGoodRefFrame)
        if (i)
            goodresults10++;

    std::cout << "results for " << queryImg->path << ":\n";
    std::cout << "goodresults: " << goodresults10 << "/10\n";
    std::cout << "meanAP10: " << meanAP10 << "\n";
    std::cout << "goodresults: " << goodresults25 << "/25\n";
    std::cout << "meanAP25: " << meanAP25 << "\n";
    std::cout << "goodresults: " << goodresults100 << "/100\n";
    std::cout << "meanAP100: " << meanAP100 << "\n";

    filteredList.resize(10);
    for (size_t i = 0; i < filteredList.size(); i++) {
        queryImg->csvrow->retrievelImages.push_back(std::pair<std::string, bool>(filteredList.at(i)->path, isGoodRefFrame.at(i)));
    }
    queryImg->csvrow->AP10 = meanAP10;
    queryImg->csvrow->rightAP10 = goodresults10;
    queryImg->csvrow->AP25 = meanAP25;
    queryImg->csvrow->rightAP25 = goodresults25;
    queryImg->csvrow->AP100 = meanAP100;
    queryImg->csvrow->rightAP100 = goodresults100;
    queryImg->csvrow->queryName = queryImg->path;
    queryImg->csvrow->maxRightResults = std::min(possivleRetCount, 100);
    return ret;
}


void Evaluator::evaluateRetrievalMultipleModels(const std::string &csvPath, const std::vector<std::pair<std::string, std::vector<std::shared_ptr<Image>>>> &imageModelPairs, std::shared_ptr<Image> &queryImg, int numRetrieved) {
    std::vector<std::shared_ptr<Image>> retrievedImgs;
    std::vector<std::pair<std::string, double>> modelScorePairs;
    std::pair<double, int> resPair;

    for (const auto &pair: imageModelPairs) {
        retrievedImgs = pair.second;
        resPair = std::pair<double, int>(evaluateRetrieval(csvPath, retrievedImgs, queryImg, numRetrieved), 0);
        if(numRetrieved == 10) {
            queryImg->csvrow->AP10 = resPair.first;
            queryImg->csvrow->rightAP10 = resPair.second;
        } else if (numRetrieved == 25) {
            queryImg->csvrow->AP25 = resPair.first;
            queryImg->csvrow->rightAP25 = resPair.second;
        } else if (numRetrieved == 100) {
            queryImg->csvrow->AP100 = resPair.first;
            queryImg->csvrow->rightAP100 = resPair.second;
        }
        modelScorePairs.push_back({pair.first, resPair.first});
    }

    sort(modelScorePairs.begin(), modelScorePairs.end(), sortTupleListBySecElemPath);
    std::cout << "Listing models sorted by precision of results" << std::endl;
    for (const auto &pair: modelScorePairs) {
        std::cout << "Precision score for model " << pair.first << " is " << pair.second << std::endl;
    }
}


void Evaluator::evaluateRetrievalMultipleModels(const std::string &csvPath, const std::vector<std::pair<std::string, std::vector<std::vector<std::shared_ptr<Image>>>>> &imageModelPairs, std::vector<std::shared_ptr<Image>> &queryImgs, const int numRetrieved) {
    std::vector<std::shared_ptr<Image>> retrievedImgs;
    std::vector<std::pair<std::string, double>> modelScorePairs;
    std::vector<double> averagePrecision;
    std::pair<double, int> resPair;

    for (const auto &pair: imageModelPairs) {
        for (size_t i = 0; i < queryImgs.size(); i++) {
            retrievedImgs = pair.second[i];
            resPair = std::pair<double, int>(evaluateRetrieval(csvPath, retrievedImgs, queryImgs[i], numRetrieved), 0);
            averagePrecision.push_back(resPair.first);
            if(numRetrieved == 10) {
                queryImgs[i]->csvrow->AP10 = resPair.first;
                queryImgs[i]->csvrow->rightAP10 = resPair.second;
            } else if (numRetrieved == 25) {
                queryImgs[i]->csvrow->AP25 = resPair.first;
                queryImgs[i]->csvrow->rightAP25 = resPair.second;
            } else if (numRetrieved == 100) {
                queryImgs[i]->csvrow->AP100 = resPair.first;
                queryImgs[i]->csvrow->rightAP100 = resPair.second;
            }
            queryImgs[i]->csvrow->queryName = queryImgs[i]->path;
        }
        modelScorePairs.push_back({pair.first, this->calculateAPMutlipleQueries(averagePrecision)});
        averagePrecision.clear();
    }

    sort(modelScorePairs.begin(), modelScorePairs.end(), sortTupleListBySecElemPath);
    std::cout << "Listing models sorted by precision of results" << std::endl;
    for (const auto &pair: modelScorePairs) {
        std::cout << "Precision score for model " << pair.first << " is " << pair.second << std::endl;
    }
}


double Evaluator::evaluateCorrectDirectory(const std::vector<std::shared_ptr<Image>> &imageList,
                                           std::shared_ptr<Image> &queryImg) {

    QString queryDir;
    queryDir = QString::fromStdString(queryImg->path).split("/").end()[-3];
    std::cout << "queryDir is " << queryDir.toStdString() << std::endl;
    int correctCounter = 0;

    for (const auto &img: imageList) {
        if (QString::fromStdString(img->path).split("/").end()[-3] == queryDir) {
            correctCounter++;
        }
    }
    return correctCounter / imageList.size();
}

bool Evaluator::createLabelListOneLine(QString &searchString, std::vector<std::pair<int, QString>> &stringList, std::vector<std::tuple<int, std::string, std::string>> &pathLabelList)  {
    int counter = 0;
    bool contained = false;
    for (const auto &str: stringList) {
        if (searchString.contains(str.second, Qt::CaseSensitive)) {
            counter++;
            contained = true;
            pathLabelList.push_back({str.first, str.second.toStdString(), searchString.split(',')[0].toStdString()});
        }
    }
    return contained;
}

