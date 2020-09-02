#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <stdexcept>
#include <sstream>

typedef std::map<int, std::vector<std::vector<double>>> csv_t;

csv_t read_csv(std::string filename)
{
    std::ifstream csv_file(filename);
    if (!csv_file.is_open())
        throw std::runtime_error("Could not open file");
    csv_t result;
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

int main()
{
    csv_t ref = read_csv("baseline_500_ref.csv");
    // vector of label 1
    std::cout << ref[1].size() << std::endl;
    // first element of vector of label 1
    std::cout << ref[1][0].size() << std::endl;
    // first element of first element of vector of label 1
    std::cout << ref[1][0][0] << std::endl;
    // second element of first element of vector of label 1
    std::cout << ref[1][0][1] << std::endl;
    // third element of first element of vector of label 1
    std::cout << ref[1][0][2] << std::endl;

    return 0;
}
