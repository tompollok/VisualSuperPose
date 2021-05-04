#ifndef PPBAFLOC_CSVHELPER_H
#define PPBAFLOC_CSVHELPER_H

#include<vector>
#include<QFile>
#include<QtCore>
#include"types/image.h"

/**
 * @brief The CSVHelper class: For collecting and saving evaluation results.
 */
class PPBAFLOC_CORE_EXPORT CSVHelper
{
public:
    /**
     * @brief CSVHelper: create a csvHelper and initialize it
     * @param savePath: the path to sace csv file
     */
    CSVHelper(std::string savePath); /// "/home/../../results.csv"
    //Only calling this Destructor will write the data into the *.csv
    ~CSVHelper();

    // for retrieval
    /**
     * @brief saveRetrieval: save the retrieval results
     */
    void saveRetrieval(
            const std::string &queryImage,
            const int &gallerySize,
            const double &AP10,
            const int &rightAP10,
            const double &AP25,
            const int &rightAP25,
            const double &AP100,
            const int &rightAP100,
            const int &maxRightResults,
            const std::vector<std::pair<std::string, bool>> &retrievelImages);

    // for registration
    /**
     * @brief saveRegistration: save the registration results
     */
    void saveRegistration(const std::string &queryImage,
                          const double &x,
                          const double &y,
                          const double &z,
                          const double &distance,
                          const double &row,
                          const double &pitch,
                          const double &yaw,
                          const double &winkel);
    /**
     * @brief saveRegistration: save the registration results
     */
    void saveRegistration(std::shared_ptr<Image>);


    /**
     * @brief saveRetrievalAndRegistration: save both results for one image
     */
    void saveRetrievalAndRegistration(std::shared_ptr<Image>);

    /**
     * @brief saveRetrievalAndRegistration: save both results for list of images
     */
    void saveRetrievalAndRegistration(std::vector<std::shared_ptr<Image>>);

private:
    //some help functions
    std::string getOnlyImageName(const std::string &path);
    int findIfExists(const std::string &queryName);
    void writeFile();
    void writeTitle();

    void csvSetVal(double val);
    void csvSetVal(QString val);
    void csvSetVal(int val);
    void csvSetRetrieved(const std::vector<std::pair<std::string, bool>>& retrievelImages, int index);

    std::vector<CSVRow> CSV; //To save some RAM this could be a shared_ptr


    /// variante for saving
    QString savePath;
    QFile file;
    QTextStream toCSV; //This should not be here. It delets the Copy-Constructor of this class.
    QString sep = "\t";
    };





#endif //PPBAFLOC_CSVHELPER_H
