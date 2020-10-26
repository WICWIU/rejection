#include "rejection.hpp"

int main()
{
    Rejection baseline_500(
        "csv/baseline_500_ref.csv",
        "csv/baseline_500_test.csv",
        0.93278782);
    baseline_500.showAccuracy_KNN(5);
    baseline_500.showPrecisionRecall_noveltyDetection();

    Rejection baseline(0.93278782);
    FEATURE_SPACE fspace = baseline.readCSV("csv/baseline_500_ref.csv");
    FEATURE_SPACE tspace = baseline.readCSV("csv/baseline_500_test.csv");

    baseline.SetRefInfo(fspace);
    baseline.SetTestInfo(tspace);
    baseline.showAccuracy_KNN(5);
    baseline.showPrecisionRecall_noveltyDetection();

    return 0;
}
