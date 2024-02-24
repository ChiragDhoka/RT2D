#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct region_features {
    double ratio;
    double percent_filled;
};

/* TASK 1 */
Mat thresholding(cv::Mat& src, int thresholding);

/* TASK 2 */
Mat customDilate(const Mat& src, const vector<vector<int>>& element);
Mat customErode(const Mat& src, const vector<vector<int>>& element);
vector<vector<int>> createStructuringElement(int rows, int cols);
Mat morphological_operation(Mat src, Mat& dst);

/* TASK 3 */
Mat segment(Mat src, Mat& dst, Mat& colored_dst, Mat& labels, Mat& stats, Mat& centroids);

/* TASK 4 */
int compute_features(Mat src, Mat& dst, vector<float>& features);

//std::vector<float> feature_computation(cv::Mat& src, cv::Mat& src_regions, cv::Mat& dst, cv::Mat stats, int nLabels);

float euclideanDistance(vector<float> f1, vector<float> f2); 
string classify(std::vector<float>& features);

float scaledEuclideanDis(std::vector<float>& feature1, std::vector<float>& feature2, std::vector<float>& deviations);

int standardDeviation(std::vector<std::vector<float>>& data, std::vector<float>& deviations);
