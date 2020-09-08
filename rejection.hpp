#include <string>
#include <vector>
#include <map>

typedef std::map<int, std::vector<std::vector<double>>> FEATURE_SPACE;

class Rejection
{
    FEATURE_SPACE fspace;
    FEATURE_SPACE tspace;
    int ref_label;
    int test_label;
    std::vector<int> * flabel;
    std::vector<int> * tlabel;
    double threshold;

    FEATURE_SPACE readCSV(std::string filename);
    double vectorDistance(std::vector<double> v, std::vector<double> u);
    double minAverage(FEATURE_SPACE fspace,
                      std::vector<double> targe);
    bool noveltyDetection(std::vector<double> target,
                          double threshold);
    int knn(std::vector<double> target, int k);
public:
    ~Rejection();
    Rejection(std::string ref_filepath, 
              std::string test_filepath,
              double threshold);
    void showPrecisionRecall_noveltyDetection();
    void showPrecisionRecall_KNN();
};