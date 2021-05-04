#ifndef PPBAFLOC_EVALUATOR_H
#define PPBAFLOC_EVALUATOR_H

#include <string>
#include <memory>

#include "../types/image.h"

#include "ppbafloc-core_export.h"

class PPBAFLOC_CORE_EXPORT Evaluator {
public:
    /**
     * @brief calculatePRCurve calculates the <recall, precision> values for meanAP
     * @param predictionResults input: true positive and false positive retrieved images.
     * @param numberPossiblePosSamples: Max possible true positives
     * @return list of precision - recall values
     */
    static std::vector<std::pair<double, double>> calculatePRCurve(const std::vector<bool> &predictionResults, int numberPossiblePosSamples);
    /**
     * @brief calculateMeanAP
     * @param prValues vector of <recall, precision> (i know, the name is confusing)
     * @param predictionResults input: true positive and false positive retrieved images.
     * @param numberPossiblePosSamples: Max possible true positives
     */
    static double calculateMeanAP(std::vector<std::pair<double, double>> &prValues, std::vector<bool> &predictionResults, int numberPossiblePosSamples);
    double evaluateRetrieval(const std::string &csvPath, const std::vector<std::shared_ptr<Image>> &imageList, std::shared_ptr<Image> &queryImg, int numRetrieved);
    /**
     * @brief evaluateRetrievalLandmarks evaluates the image retrieval for google landmarks v2 (clean).
     * @param imageList needs 100 retrieved images without queryimage. 101 if imageList contains queryImg.
     */
    std::vector<bool> evaluateRetrievalLandmarks(
            const std::string &csvPath,
            std::vector<std::shared_ptr<Image>> &imageList,
            std::shared_ptr<Image> &queryImg);
    /**
     * @brief evaluateRetrievalCornell evaluates the image retrieval for cornell.
     * @param imageList needs 100 retrieved images without queryimage. 101 if imageList contains queryImg.
     */
    std::vector<bool> evaluateRetrievalCornell(
            std::vector<std::shared_ptr<Image>> &imageList,
            std::shared_ptr<Image> &queryImg);
    /**
     * @brief evaluateCorrectDirectory evaluates, if retreived images is in the same folder as queryimage.
     * Deprecated, not used anywhere
     */
    double evaluateCorrectDirectory(const std::vector<std::shared_ptr<Image>> &imageList, std::shared_ptr<Image> &queryImg);
    /**
     * @brief getGoodQueryImages finds numQueryImages query images from labels with at least minNumberImages in Clean Google csvPath that exist in evalImagesDir.
     */
    void getGoodQueryImages(
            const std::string &csvPath,
            const std::string &evalImagesDir,
            int minNumberImages,
            int numQueryImages);
    /**
     * @brief calculateAPMutlipleQueries calculates the mean AP
     */
    double calculateAPMutlipleQueries(
            const std::vector<double> &aps);

private:
    /**
     * @brief creates a list of labels from one line of the csv file
     * @param searchString string to be searched in
     * @param stringList  strings to be checked whether they are contained in searchString
     * @param pathLabelList list of labels contained in searchString
     * @return true if searchString contained any of stringList, else false
     */
    bool createLabelListOneLine(QString &searchString, std::vector<std::pair<int, QString>> &stringList, std::vector<std::tuple<int, std::string, std::string>> &pathLabelList);
    /**
     * @brief evaluateRetrievalMultipleModels Deprecated, not used anywhere
     */
    void evaluateRetrievalMultipleModels(const std::string &csvPath,
                                         const std::vector<std::pair<std::string, std::vector<std::vector<std::shared_ptr<Image>>>>> &imageModelPairs,
                                         std::vector<std::shared_ptr<Image>> &queryImgs, int numRetrieved);
    /**
     * @brief evaluateRetrievalMultipleModels Deprecated, not used anywhere
     */
    void evaluateRetrievalMultipleModels(const std::string &csvPath,
                                         const std::vector<std::pair<std::string, std::vector<std::shared_ptr<Image>>>> &imageModelPairs,
                                         std::shared_ptr<Image> &queryImg, int numRetrieved);
    /**
     * @brief evaluateMultipleQueries Deprecated, not used anywhere
     */
    double evaluateMultipleQueries(const std::string &csvPath, std::vector<std::vector<std::shared_ptr<Image>>> &imageLists,
                            std::vector<std::shared_ptr<Image>> &queryList, int numRetrieved);
};


#endif //PPBAFLOC_EVALUATOR_H
