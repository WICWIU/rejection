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
    size_t fspace_size;
    size_t tspace_size;
    std::vector<int> * flabel;
    std::vector<int> * tlabel;
    double threshold;

    FEATURE_SPACE readCSV(std::string filename);
    double distance(std::vector<double> v, std::vector<double> u);
    double minAverage(std::vector<double> targe);
    bool noveltyDetection(std::vector<double> target, double threshold);
    int knn(std::vector<double> target, int k);
public:
    ~Rejection();
    Rejection(std::string ref_filepath, 
              std::string test_filepath,
              double threshold);
    void showPrecisionRecall_noveltyDetection();
    void showPrecisionRecall_KNN(int k);
};