//#include "knn.h"
//#include <map>
//#include <iostream>
//#include <math.h>
//#include <opencv2/opencv.hpp>
//
//
//using namespace std;
//using namespace cv;
//
//struct featureNode {
//    int index;
//    std::string label;
//    float value;
//
//    featureNode(int index, std::string label, float value) {
//        this->index = index;
//        this->label = label;
//        this->value = value;
//    }
//};
//
//bool valueComparator(const featureNode& a, const featureNode& b) {
//    return a.value < b.value;
//}
//
//std::vector<float> min_max(std::vector<std::vector<float>> trainedFeatures) {
//    std::vector<float> minMax(8, 0);
//    std::vector<float> f1Vector;
//    std::vector<float> f2Vector;
//    std::vector<float> f3Vector;
//    std::vector<float> f4Vector;
//    for (int i = 0; i < trainedFeatures.size(); i++) {
//        f1Vector.push_back(trainedFeatures[i][0]);
//        f2Vector.push_back(trainedFeatures[i][1]);
//        f3Vector.push_back(trainedFeatures[i][2]);
//        f4Vector.push_back(trainedFeatures[i][3]);
//    }
//    minMax[0] = *std::min_element(f1Vector.begin(), f1Vector.end());
//    minMax[1] = *std::max_element(f1Vector.begin(), f1Vector.end());
//    minMax[2] = *std::min_element(f2Vector.begin(), f2Vector.end());
//    minMax[3] = *std::max_element(f2Vector.begin(), f2Vector.end());
//    minMax[4] = *std::min_element(f3Vector.begin(), f3Vector.end());
//    minMax[5] = *std::max_element(f3Vector.begin(), f3Vector.end());
//    minMax[6] = *std::min_element(f4Vector.begin(), f4Vector.end());
//    minMax[7] = *std::max_element(f4Vector.begin(), f4Vector.end());
//    return minMax;
//}
//
//
//std::string temp(std::vector<float> targetFeatures, std::vector<char*> trainedLabels, std::vector<std::vector<float>> trainedNestedFeatures, int k) {
//    if (k > trainedNestedFeatures.size()) {
//        return "NA";
//    }
//    float distance = 0;
//    float summationOfDifferenceOfSquares = 0;
//    std::vector<featureNode> featureList;
//    std::vector<float> minMaxArray = min_max(trainedNestedFeatures);
//    for (int objectIter = 0; objectIter < trainedNestedFeatures.size(); objectIter++) {
//        summationOfDifferenceOfSquares = 0;
//        for (int featureIter = 0; featureIter < trainedNestedFeatures[objectIter].size(); featureIter++) {
//            float feature = trainedNestedFeatures[objectIter][featureIter];
//            feature = (feature - minMaxArray[featureIter * 2]) / (minMaxArray[featureIter * 2 + 1] - minMaxArray[featureIter * 2]);
//            float targetFeature = (targetFeatures[featureIter] - minMaxArray[featureIter * 2]) / (minMaxArray[featureIter * 2 + 1] - minMaxArray[featureIter * 2]);
//            summationOfDifferenceOfSquares += (feature - targetFeature) * (feature - targetFeature);
//        }
//        // Calculates Euclidean distance.
//        distance = pow(summationOfDifferenceOfSquares, 0.5);
//        // std::cout << "Distance for " << trainedLabels[objectIter] << " is " << distance << std::endl;
//        featureNode f(objectIter, trainedLabels[objectIter], distance);
//        featureList.push_back(f);
//    }
//
//    std::sort(featureList.begin(), featureList.end(), valueComparator);
//    std::map<std::string, int> kNNLabelFrequency;
//    for (int i = 0; i < k; i++) {
//        if (kNNLabelFrequency.count(featureList[i].label) == 0) {
//            kNNLabelFrequency[featureList[i].label] = 1;
//        }
//        else {
//            kNNLabelFrequency[featureList[i].label] += 1;
//        }
//    }
//
//    int maxFrequency = -1;
//    std::string labelWithMaxFrequency = "";
//    for (auto it = kNNLabelFrequency.begin(); it != kNNLabelFrequency.end(); ++it) {
//        if (it->second > maxFrequency) {
//            maxFrequency = it->second;
//            labelWithMaxFrequency = it->first;
//        }
//    }
//    return labelWithMaxFrequency;
//}