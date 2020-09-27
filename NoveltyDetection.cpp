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

#include "Rejection.cpp"

typedef std::map<int, std::vector<std::vector<double>>> FEATURE_SPACE;

class NoveltyDetection : public Rejection
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
    double minAverage(std::vector<double> targe);

public:
    ~NoveltyDetection();
    NoveltyDetection(FEATURE_SPACE fspace1, FEATURE_SPACE tspace1, int ref_label1, int test_label1, size_t fspace_size1, size_t tspace_size1, std::vector<int> *flabel, std::vector<int> *tlabel1, double threshold1);
    virtual bool NoveltyDetection::reject(std::vector<double> target, double threshold);
};

double NoveltyDetection::distance(std::vector<double> v, std::vector<double> u)
{
    assert(v.size() == u.size());
    double accm = 0.;
    for (int i = 0; i < v.size(); i++)
        accm += pow(v[i] - u[i], 2);
    return sqrt(accm);
}

double NoveltyDetection::minAverage(std::vector<double> target)
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

bool NoveltyDetection::reject(std::vector<double> target, double threshold)
{
    if (minAverage(target) < threshold)
        return false;
    else
        return true; // outlier
}

NoveltyDetection(FEATURE_SPACE fspace, FEATURE_SPACE tspace, int ref_label, int test_label, size_t fspace_size, size_t tspace_size, std::vector<int> *flabel, std::vector<int> *tlabel, double threshold)
{
    this->fspace = fspace;
    this->tspace = tspace;
    this->ref_label = ref_label;
    this->test_label = test_label;
    this->flabel = new std::vector<int>(ref_label);
    this->tlabel = new std::vector<int>(test_label);

    this->tspace_size = tspace_size;
    this->fspace_size = fspace_size;
};

~NoveltyDetection()
{
    delete this->flabel;
    delete this->tlabel;
}
