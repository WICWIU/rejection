class Rejection
{
public:
    virtual ~Rejection() {}
    virtual bool reject(std::vector<double> target, double threshold) = 0;
};