# rejection (novelty detection)

outlier 인지 판단하는 알고리즘.

두 클래스가 있을 때 Max, Avg, Centroid 의 방법 중 하나를 택하여 모든 클래스 간의 거리를 구하고 가장 짧은 거리에 대하여 Threshold 를 정해서 Threshold 보다 높으면 outlier 이고 낮으면 그 클래스로 분류한다.

## Average rejection

1. 각 라벨에 대한 거리를 구하고 평균을 구한다.

2. 모든 거리 평균에서 최솟값을 구한다.

3. 최솟값이 Threshole 보다 작으면 해당 클래스로 분류하고 높으면 우선 `-1` 을 반환하여 outlier 임을 표시하기로 한다.

## Testing

`test_noveltyDetection` 함수를 통해 테스팅을 해본다. 이 함수는 `csv/baseline_500_ref.csv` 을 읽고 각 좌표가 해당 클래스에 속한다고 판단하는지 검증한다.

```shell
$ ./rejection
Testing noveltyDetection accuray... 100.0%
noveltyDetection accuray        98.96%
```

정확도가 98.96% 가 나왔다.

이제 `csv/baseline_500_test.csv` 를 통하여 본격적인 테스트를 해본다.

# KNN

두 클래스가 있다고 가정할 때 분류하고 싶은 데이터에 대하여 가장 가까운 3개의 데이터를 구하고, 그 중 가장 많은 데이터의 클래스로 분류하는 것이다.

5 개 정도가 적당했다.

