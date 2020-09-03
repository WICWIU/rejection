#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <stdexcept>
#include <sstream>

typedef std::map<int, std::vector<std::vector<double>>> FEATURE_SPACE;

FEATURE_SPACE readCSV(std::string filename)
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

double vectorDistance(std::vector<double> v, std::vector<double> u)
{
    assert(v.size() == u.size());
    double accm = 0.;
    for (int i = 0; i < v.size(); i++)
        accm += pow(v[i] - u[i], 2);
    return sqrt(accm);
}

std::pair<int, double> minAverage(FEATURE_SPACE fspace,
                                       std::vector<double> v,
                                       std::vector<int> label)
{
    double min_average = std::numeric_limits<double>::infinity();
    int min_label;
    for (const int &l : label)
    {
        double average = 0.;
        for (const auto &u : fspace[l])
        {
            average += vectorDistance(u, v);
        }
        average /= v.size();
        if (min_average > average)
        {
            min_label = l;
            min_average = average;
        }
    }
    return std::make_pair(min_label, min_average);
}

int noveltyDetection(FEATURE_SPACE fspace,
                      std::vector<double> v,
                      std::vector<int> label,
                      double threshold)
{
    std::pair<int, double> min = minAverage(fspace, v, label);
    double min_average = min.second;
    int min_label = min.first;
    if (min_average < threshold)
        return min_label;
    else
        return -1; // outlier
}

// TEST FUNCTIONS
void test_readCSV();
void test_noveltyDetection();
// END of TEST FUNCTIONS

int main()
{
    //test_readCSV();
    test_noveltyDetection();

    return 0;
}

void test_readCSV()
{
    FEATURE_SPACE fspace = readCSV("csv/baseline_500_ref.csv");
    // vector of label 1
    std::cout << fspace[1].size() << std::endl;
    // first element of vector of label 1
    std::cout << fspace[1][0].size() << std::endl;
    // first element of first element of vector of label 1
    std::cout << fspace[1][0][0] << std::endl;
    // second element of first element of vector of label 1
    std::cout << fspace[1][0][1] << std::endl;
    // third element of first element of vector of label 1
    std::cout << fspace[1][0][2] << std::endl;
}

void test_noveltyDetection()
{
    FEATURE_SPACE fspace = readCSV("csv/baseline_500_ref.csv");
    std::vector<int> label(500);
    for (int i = 1; i <= 500; i++)
        label[i - 1] = i;
    double accuray = 0;
    double ct = 0;
    for(const int& l: label)
    {
        for(const auto& u: fspace[l])
        {
            if (l == noveltyDetection(fspace, u, label, 0.4))
                accuray++;
            ct++;
        }
        printf("Testing noveltyDetection accuray... %3.1f\%\r", 100 * (ct / 2500));
    }
    std::cout << std::endl;
    std::cout << "noveltyDetection accuray\t" << 100 * (accuray / 2500) << '%' << std::endl;
}
