#include <vector>

class Classifier
{
public:
    virtual ~Classifier() {}
    virtual int classify(std::vector<double> target, int k) = 0;
};