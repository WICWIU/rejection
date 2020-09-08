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

FEATURE_SPACE Rejection::readCSV(std::string filename)
{
    std::ifstream csv_file(filename);
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

double Rejection::vectorDistance(std::vector<double> v, std::vector<double> u)
{
    assert(v.size() == u.size());
    double accm = 0.;
    for (int i = 0; i < v.size(); i++)
        accm += pow(v[i] - u[i], 2);
    return sqrt(accm);
}

double Rejection::minAverage(FEATURE_SPACE fspace,
                    std::vector<double> target)
{
    double min_average = std::numeric_limits<double>::infinity();
    int min_label;
    for (const int &l : *this->flabel)
    {
        double average = 0.;
        for (const auto &u : fspace[l])
            average += vectorDistance(u, target);
        average /= fspace[l].size();
        if (min_average > average)
            min_average = average;
    }
    return min_average;
}

bool Rejection::noveltyDetection(std::vector<double> target,
                        double threshold)
{
    double min_average = minAverage(this->fspace, target);
    if (min_average < threshold)
        return false;
    else
        return true; // outlier
}

int Rejection::knn(std::vector<double> target, int k)
{
    // first k'th minimum <label, distance>
    std::vector<std::pair<int, double>> min_distances;

    for (const int &l : *this->flabel)
    {
        for (const auto &u : this->fspace[l])
        {
            double distance = vectorDistance(u, target);
            // fullfill first k'th <label, distance>
            if (min_distances.size() < k)
                min_distances.push_back(std::make_pair(l, distance));
            else 
            {
                double max = -std::numeric_limits<float>::infinity();
                int max_index;
                // get max value and index
                for(int i= 0 ; i < min_distances.size(); i++)
                {
                    if(max < std::get<1>(min_distances[i]))
                    {
                        max = std::get<1>(min_distances[i]);
                        max_index = i;
                    }
                }
                // remove max value and push new minimum value
                if (distance < max)
                {
                    min_distances.erase(min_distances.begin() + max_index);
                    min_distances.push_back(std::make_pair(l, distance));
                }
            }
        }
    }

    // count each label
    std::map<int, int> classified_label;
    for(const auto& m: min_distances)
        classified_label[std::get<0>(m)]++;

    // save maximum counted label into result
    int result;
    unsigned currentMax = 0;
    for(auto it = classified_label.cbegin(); it != classified_label.cend(); ++it) {
        if (it->second > currentMax) {
            result = it->first;
            currentMax = it->second;
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

    for (int i = 1; i <= ref_label; i++)
        (*this->flabel)[i - 1] = i;
    for (int i = 1; i <= test_label; i++)
        (*this->tlabel)[i - 1] = i;
}

void Rejection::showPrecisionRecall_noveltyDetection()
{
    double ct = 0;
    double TP, FP, TN, FN;
    TP = FP = TN = FN = 0;

    for (const int &l : (*this->tlabel))
    {
        for (const auto &u : tspace[l])
        {
            if (l != 501)
            {
                if (noveltyDetection(u, 0.93278782))
                    FP++;
                else
                    TN++;
            }
            else
            {
                if (noveltyDetection(u, 0.93278782))
                    TP++;
                else
                    FN++;
            }
            ct++;
            printf("Testing noveltyDetection progress: %2.1f%%\r", 100 * (ct / 3000));
        }
    }

    std::cout << std::endl;
    std::cout << "TN: " << TN << std::endl;
    std::cout << "FP: " << FP << std::endl;
    std::cout << "TP: " << TP << std::endl;
    std::cout << "FN: " << FN << std::endl;
    std::cout << "TP + FN = " << TP + FN << std::endl;
    std::cout << std::endl;
    std::cout << "noveltyDetection Precision\t" << TP / (TP + FP) * 100 << '%' << std::endl;
    std::cout << "noveltyDetection Recall\t" << TP / (TP + FN) * 100 << '%' << std::endl;
    std::cout << "noveltyDetection accuray\t" << 100 * ((TP + TN) / (TP + TN + FP + FN)) << '%' << std::endl;
}

void Rejection::showPrecisionRecall_KNN()
{
    double accuracy = 0;
    double ct = 0;
    for (const int &l : *this->tlabel)
    {
        for (const auto &u : tspace[l])
        {
            if (l != 501)
            {
                if (l == knn(u, 5))
                    accuracy++;
                ct++;
                printf("Testing knn progress: %2.1f%%\r", 100 * (ct / 2500));
            }
        }
    }
    std::cout << std::endl;
    std::cout << "KNN accuracy: " << 100 * (accuracy / 2500) << "%" << std::endl;
}
