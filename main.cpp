#include <limits>
#include <numeric>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <stdexcept>
#include <sstream>

#include <cassert>
#include <cmath>
#include <cstdio>

#include "NoveltyDetection.cpp"

typedef std::map<int, std::vector<std::vector<double>>> FEATURE_SPACE;

FEATURE_SPACE readCSV(std::string filepath);
void showPrecisionRecall_noveltyDetection();
void showAccuracy_KNN(int k, FEATURE_SPACE fspace, FEATURE_SPACE tspace, int ref_label, int test_label, size_t fspace_size, size_t tspace_size, std::vector<int> *flabel, std::vector<int> *tlabel, double threshold);

int main()
{

    FEATURE_SPACE fspace;
    FEATURE_SPACE tspace;
    size_t fspace_size;
    size_t tspace_size;
    std::vector<int> *flabel;
    std::vector<int> *tlabel;
    double threshold = 0.93278782;

    fspace = readCSV("csv/baseline_500_ref.csv");
    tspace = readCSV("csv/baseline_500_test.csv");
    int ref_label = fspace.size();
    int test_label = tspace.size();
    flabel = new std::vector<int>(ref_label);
    tlabel = new std::vector<int>(test_label);

    tspace_size = 0;
    fspace_size = 0;
    for (int i = 0; i < tspace.size(); i++)
        tspace_size += tspace[i].size();
    for (int i = 0; i < fspace.size(); i++)
        fspace_size += fspace[i].size();

    for (int i = 1; i <= ref_label; i++)
        (*flabel)[i - 1] = i;
    for (int i = 1; i <= test_label; i++)
        (*tlabel)[i - 1] = i;

    showAccuracy_KNN(5, fspace, tspace, ref_label, test_label, fspace_size, tspace_size, flabel, tlabel, threshold);

    //showPrecisionRecall_noveltyDetection();

    return 0;
}

FEATURE_SPACE readCSV(std::string filepath)
{
    std::ifstream csv_file(filepath);
    if (!csv_file.is_open())
        throw std::runtime_error("Could not open file");

    FEATURE_SPACE result;
    std::string line, colname;
    double val;
    int label;

    if (csv_file.good())
    {
        std::getline(csv_file, line); // ignore first line (because : column line)
        while (std::getline(csv_file, line))
        {
            std::stringstream ss(line);
            ss >> label;
            if (result.find(label) == result.end()) // if there is no label
                result[label] = std::vector<std::vector<double>>{};
            result[label].push_back(std::vector<double>{});
            ss.ignore(); // ignore first comma
            while (ss >> val)
            {
                result[label].back().push_back(val);
                if (ss.peek() == ',')
                    ss.ignore();
            }
        }
    }
    csv_file.close();

    return result;
}

// void showPrecisionRecall_noveltyDetection()
// {
//     double ct = 0;
//     double TP, FP, TN, FN;
//     TP = FP = TN = FN = 0;
//     int outlier_label = (*tlabel).back();
//     double accuracy = 0;

//     for (const int &test_label : *tlabel)
//     {
//         for (const auto &test_feature : tspace[test_label])
//         {
//             if (test_label != outlier_label)
//             {
//                 if (noveltyDetection.reject(test_feature, threshold))
//                     FP++;
//                 else
//                     TN++;
//             }
//             else // outlier
//             {
//                 if (noveltyDetection.reject(test_feature, threshold))
//                     TP++;
//                 else
//                     FN++;
//             }
//             printf("Testing noveltyDetection progress: %2.1f%%\r",
//                    100 * (++ct / tspace_size));
//         }
//     }

//     std::cout << std::endl;
//     // std::cout << "TN: " << TN << std::endl;
//     // std::cout << "FP: " << FP << std::endl;
//     // std::cout << "TP: " << TP << std::endl;
//     // std::cout << "FN: " << FN << std::endl;
//     // std::cout << "TP + FN + TN + FP = " << TP + TN + FP + FN << std::endl;
//     std::cout << std::endl;
//     std::cout << "noveltyDetection Precision\t" << TP / (TP + FP) * 100 << '%' << std::endl;
//     std::cout << "noveltyDetection Recall\t" << TP / (TP + FN) * 100 << '%' << std::endl;
//     std::cout << "noveltyDetection accuray\t" << 100 * ((TP + TN) / (TP + TN + FP + FN)) << '%' << std::endl;
// }

void showAccuracy_KNN(int k, FEATURE_SPACE fspace, FEATURE_SPACE tspace, int ref_label, int test_label, size_t fspace_size, size_t tspace_size, std::vector<int> *flabel, std::vector<int> *tlabel, double threshold)
{
    double accuracy = 0;
    double rej_accuracy = 0;
    double ct = 0;
    int outlier_label = (*tlabel).back();
    NoveltyDetection n_detect = new NoveltyDetection(fspace, tspace, ref_label, test_label, fspace_size, tspace_size, &flabel, &tlabel, threshold);

    for (const int &test_label : *tlabel)
    {
        for (const auto &test_feature : tspace[test_label])
        {

            if (test_label != outlier_label)
            {
                if (!n_detect.reject(test_feature, threshold))
                    // && (test_label == knn(test_feature, k)))
                    rej_accuracy++;

                // if (test_label == knn(test_feature, k))
                accuracy++;
                printf("Testing knn progress: %2.1f%%\r", 100 * (++ct / fspace_size));
            }
        }
    }
    std::cout << std::endl;
    std::cout << fspace_size << std::endl;
    std::cout << "KNN accuracy without rejection function: " << 100 * (accuracy / fspace_size) << "%" << std::endl;
    std::cout << "KNN accuracy with rejection function: " << 100 * (rej_accuracy / fspace_size) << "%" << std::endl;
}