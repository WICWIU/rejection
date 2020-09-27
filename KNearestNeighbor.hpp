#include <string>
#include <vector>
#include <map>

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

#include "Classifier.hpp"

typedef std::map<int, std::vector<std::vector<double>>> FEATURE_SPACE;

class KNearestNeighbor : public Classifier
{
private:
    FEATURE_SPACE fspace;
    FEATURE_SPACE tspace;
    int ref_label;
    int test_label;
    size_t fspace_size;
    size_t tspace_size;
    std::vector<int> *flabel;
    std::vector<int> *tlabel;

    double distance(std::vector<double> v, std::vector<double> u);

public:
    ~KNearestNeighbor();
    KNearestNeighbor(std::string ref_filepath,
                     std::string test_filepath,
                     double threshold);
    virtual int classify(std::vector<double> target, int k);
};

int KNearestNeighbor::classify(std::vector<double> target, int k)
{
    // first k'th minimum <label, distance>
    std::vector<std::pair<int, double>> min_distances;

    for (const int &label : flabel)
    {
        for (const auto &feature : fspace[label])
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

double KNearestNeighbor::distance(std::vector<double> v, std::vector<double> u)
{
    assert(v.size() == u.size());
    double accm = 0.;
    for (int i = 0; i < v.size(); i++)
        accm += pow(v[i] - u[i], 2);
    return sqrt(accm);
}

KNearestNeighbor(std::string ref_filepath,
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

~KNearestNeighbor()
{
    delete this->flabel;
    delete this->tlabel;
}
