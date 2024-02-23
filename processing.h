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
vector<float> compute_features(Mat src, Mat& dst, vector<region_features>& features);

//std::vector<float> feature_computation(cv::Mat& src, cv::Mat& src_regions, cv::Mat& dst, cv::Mat stats, int nLabels);