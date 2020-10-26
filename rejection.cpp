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

#include "rejection.hpp"

FEATURE_SPACE Rejection::readCSV(std::string filepath)
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

double Rejection::distance(std::vector<double> v, std::vector<double> u)
{
    assert(v.size() == u.size());
    double accm = 0.;
    for (int i = 0; i < v.size(); i++)
        accm += pow(v[i] - u[i], 2);
    return sqrt(accm);
}

double Rejection::minAverage(std::vector<double> target)
{
    double min_average = std::numeric_limits<double>::infinity();
    int min_label;
    for (const int &label : *this->flabel)
    {
        double average = 0.;
        for (const auto &u : this->fspace[label])
            average += distance(u, target);
        average /= this->fspace[label].size();
        if (min_average > average)
            min_average = average;
    }
    return min_average;
}

bool Rejection::noveltyDetection(std::vector<double> target, double threshold)
{
    if (minAverage(target) < threshold)
        return false;
    else
        return true; // outlier
}

int Rejection::knn(std::vector<double> target, int k)
{
    // first k'th minimum <label, distance>
    std::vector<std::pair<int, double>> min_distances;

    for (const int &label : *this->flabel)
    {
        for (const auto &feature : this->fspace[label])
        {
            double target_distance = distance(feature, target);
            // fullfill first k'th <label, distance>
            if (min_distances.size() < k)
                min_distances.push_back(std::make_pair(label, target_distance));
            else
            {
                double max = -std::numeric_limits<float>::infinity();
                int max_index;
                // get max value and index from minimum <label, distance>
                for (int i = 0; i < min_distances.size(); i++)
                {
                    if (max < min_distances[i].second)
                    {
                        max = min_distances[i].second;
                        max_index = i;
                    }
                }
                // remove max value and push new minimum value
                if (target_distance < max)
                {
                    min_distances.erase(min_distances.begin() + max_index);
                    min_distances.push_back(std::make_pair(label, target_distance));
                }
            }
        }
    }

    // count each label
    std::map<int, int> classified_labels;
    for (const auto &min_distance : min_distances)
        classified_labels[min_distance.first]++;

    // save maximum counted label into result
    int result;
    unsigned currentMax = 0;
    for (const auto &classified_label : classified_labels)
    {
        if (classified_label.second > currentMax)
        {
            result = classified_label.first;
            currentMax = classified_label.second;
        }
    }

    return result;
}

Rejection::~Rejection()
{
    delete this->flabel;
    delete this->tlabel;
}

Rejection::Rejection(std::string ref_filepath,
                     std::string test_filepath,
                     double threshold)
    : threshold(threshold)
{
    this->fspace = readCSV("csv/baseline_500_ref.csv");
    this->tspace = readCSV("csv/baseline_500_test.csv");
    int ref_label = fspace.size();
    int test_label = tspace.size();
    this->flabel = new std::vector<int>(ref_label);
    this->tlabel = new std::vector<int>(test_label);

    this->tspace_size = 0;
    this->fspace_size = 0;
    for (int i = 0; i < tspace.size(); i++)
        this->tspace_size += tspace[i].size();
    for (int i = 0; i < fspace.size(); i++)
        this->fspace_size += fspace[i].size();

    for (int i = 1; i <= ref_label; i++)
        (*this->flabel)[i - 1] = i;
    for (int i = 1; i <= test_label; i++)
        (*this->tlabel)[i - 1] = i;
}

Rejection::Rejection(double threshold)
    : threshold(threshold)
{
}

void Rejection::SetTestInfo(FEATURE_SPACE test_space)
{
    this->tspace = test_space;
    int test_label = tspace.size();
    this->tlabel = new std::vector<int>(test_label);
    this->tspace_size = 0;
    for (int i = 0; i < tspace.size(); i++)
        this->tspace_size += tspace[i].size();
    for (int i = 1; i <= test_label; i++)
        (*this->tlabel)[i - 1] = i;
}
void Rejection::SetRefInfo(FEATURE_SPACE ref_space)
{
    this->fspace = ref_space;
    int ref_label = fspace.size();
    this->flabel = new std::vector<int>(ref_label);
    this->fspace_size = 0;
    for (int i = 0; i < fspace.size(); i++)
        this->fspace_size += fspace[i].size();
    for (int i = 1; i <= ref_label; i++)
        (*this->flabel)[i - 1] = i;
}

void Rejection::showPrecisionRecall_noveltyDetection()
{
    double ct = 0;
    double TP, FP, TN, FN;
    TP = FP = TN = FN = 0;
    int outlier_label = (*this->tlabel).back();
    double accuracy = 0;

    for (const int &test_label : *this->tlabel)
    {
        for (const auto &test_feature : this->tspace[test_label])
        {
            if (test_label != outlier_label)
            {
                if (noveltyDetection(test_feature, this->threshold))
                    FP++;
                else
                    TN++;
            }
            else // outlier
            {
                if (noveltyDetection(test_feature, this->threshold))
                    TP++;
                else
                    FN++;
            }
            printf("Testing noveltyDetection progress: %2.1f%%\r",
                   100 * (++ct / this->tspace_size));
        }
    }

    std::cout << std::endl;
    // std::cout << "TN: " << TN << std::endl;
    // std::cout << "FP: " << FP << std::endl;
    // std::cout << "TP: " << TP << std::endl;
    // std::cout << "FN: " << FN << std::endl;
    // std::cout << "TP + FN + TN + FP = " << TP + TN + FP + FN << std::endl;
    std::cout << std::endl;
    std::cout << "noveltyDetection Precision\t" << TP / (TP + FP) * 100 << '%' << std::endl;
    std::cout << "noveltyDetection Recall\t" << TP / (TP + FN) * 100 << '%' << std::endl;
    std::cout << "noveltyDetection accuray\t" << 100 * ((TP + TN) / (TP + TN + FP + FN)) << '%' << std::endl;
}

void Rejection::showAccuracy_KNN(int k)
{
    double accuracy = 0;
    double rej_accuracy = 0;
    double ct = 0;
    int outlier_label = (*this->tlabel).back();

    for (const int &test_label : *this->tlabel)
    {
        for (const auto &test_feature : tspace[test_label])
        {

            if (test_label != outlier_label)
            {
                if (!noveltyDetection(test_feature, this->threshold) && (test_label == knn(test_feature, k)))
                    rej_accuracy++;

                if (test_label == knn(test_feature, k))
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
