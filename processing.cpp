//processing.cpp
#include <iostream>
#include <vector>
#include <stdio.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/dnn.hpp"
#include "csv_util.h"
#include "knn.cpp"

using namespace std;
using namespace cv;

/*Stores the features of a region */
struct region_features {
    double ratio;
    double percent_filled;
    vector<double> huMoments;
};

/*------------------ TASK 1 ------------------------------------------------------------------*/
/*Separate the object from the back ground according to a threshold */
Mat thresholding(Mat& src, int threshold) {

    Mat Thresh_Image, grayscale;
    Thresh_Image = Mat(src.size(), CV_8UC1);

    cvtColor(src, grayscale, COLOR_BGR2GRAY);

    int i = 0;
    while (i < grayscale.rows) {
        int j = 0;
        while (j < grayscale.cols) {
            if (grayscale.at<uchar>(i, j) > threshold) {
                Thresh_Image.at<uchar>(i, j) = 0;
            }
            else {
                Thresh_Image.at<uchar>(i, j) = 255;
            }
            j++;
        }
        i++;
    }

    return Thresh_Image;
}

/*------------------ TASK 2 ------------------------------------------------------------------*/
/* Function for cleaning up the binary image, uses morphological filtering to first shrink any unexpected noise,
   then grows back to clean up holes in the image. Uses erosion followed by dilation to remove noise, then
   dilation followed by erosion to remove the holes caused by the reflections.
   Parameters: src, a binary mat object that will be cleaned up
   Returns: A cleaned up binary Mat image
*/
vector<vector<int>> createStructuringElement(int rows, int cols) {
    vector<vector<int>> element(rows, vector<int>(cols, 1));
    return element;
}

// Custom erosion function
Mat customErode(const Mat& src, const vector<vector<int>>& element) {
    Mat dst = src.clone();
    int elementRows = element.size();
    int elementCols = element[0].size();
    int originX = elementRows / 2;
    int originY = elementCols / 2;

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            bool erodePixel = false;
            for (int x = 0; x < elementRows; ++x) {
                for (int y = 0; y < elementCols; ++y) {
                    int relX = i + x - originX;
                    int relY = j + y - originY;
                    if (relX >= 0 && relX < src.rows && relY >= 0 && relY < src.cols) {
                        if (element[x][y] == 1 && src.at<uchar>(relX, relY) == 0) {
                            erodePixel = true;
                            break;
                        }
                    }
                }
                if (erodePixel) break;
            }
            dst.at<uchar>(i, j) = erodePixel ? 0 : 255;
        }
    }
    return dst;
}

// Custom dilation function
Mat customDilate(const Mat& src, const vector<vector<int>>& element) {
    Mat dst = src.clone();
    int elementRows = element.size();
    int elementCols = element[0].size();
    int originX = elementRows / 2;
    int originY = elementCols / 2;

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            for (int x = 0; x < elementRows; ++x) {
                for (int y = 0; y < elementCols; ++y) {
                    int relX = i + x - originX;
                    int relY = j + y - originY;
                    if (relX >= 0 && relX < src.rows && relY >= 0 && relY < src.cols) {
                        if (element[x][y] == 1 && src.at<uchar>(relX, relY) == 255) {
                            dst.at<uchar>(i, j) = 255;
                            break;
                        }
                    }
                }
            }
        }
    }
    return dst;
}

Mat morphological_operation(Mat src, Mat& dst) {

    Mat dilated = src.clone();
    auto structElem = createStructuringElement(3, 3);

    // Perform custom erosion and dilation
    Mat dilatedImage = customDilate(dilated, structElem);
    Mat erode = dilatedImage.clone();

    Mat erodedImage = customErode(erode, structElem);

    dst = erodedImage;

    return dst;
}


/*------------------ TASK 3 ------------------------------------------------------------------*/
/*Function for connected component analysis, creates segmented, region-colored version of the src image
  Parameters: a src image to be sampled from, then Mat data types for labels, stats, and centroid calculation.
  Returns: a segmented, region colored version of the src image
*/

Mat segment(Mat src, Mat& dst, Mat& colored_dst, Mat& labels, Mat& stats, Mat& centroids) {


    int num = connectedComponentsWithStats(src, labels, stats, centroids, 8);

    //std::cout << num << std::endl;

    // number of colors will equal to number of regions
    vector<Vec3b> colors(num);
    vector<Vec3b> intensity(num);
    // set background to black
    colors[0] = Vec3b(0, 0, 0);
    intensity[0] = Vec3b(0, 0, 0);

    int area = 0;

    for (int i = 1; i < num; i++) {

        colors[i] = Vec3b(255 * i % 256, 170 * i % 256, 200 * i % 256);
        intensity[i] = Vec3b(255, 255, 255);

        // keep only the largest region
        if (stats.at<int>(i, CC_STAT_AREA) > area) {
            area = stats.at<int>(i, CC_STAT_AREA);
        }
        else {
            colors[i] = Vec3b(0, 0, 0);
            intensity[i] = Vec3b(0, 0, 0);
        }
    }
    // assign the colors to regions
    Mat colored_img = Mat::zeros(src.size(), CV_8UC3);
    Mat intensity_img = Mat::zeros(src.size(), CV_8UC3);
    for (int i = 0; i < colored_img.rows; i++) {
        for (int j = 0; j < colored_img.cols; j++)
        {
            int label = labels.at<int>(i, j);
            colored_img.at<Vec3b>(i, j) = colors[label];
            intensity_img.at<Vec3b>(i, j) = intensity[label];
        }
    }

    cvtColor(intensity_img, src, COLOR_BGR2GRAY);
    num = connectedComponentsWithStats(src, labels, stats, centroids, 8);
    dst = intensity_img.clone();
    colored_dst = colored_img.clone();

    return dst;
}


/*------------------ TASK 4 ------------------------------------------------------------------*/
/*Computes a set of features for a specified region given a region map and a region ID. */
/* Computes huMoment feature vector, computes and displays oriented bounding box
   Paramters: Mat src: binary image to be sampled from
          Mat src_regions: colored_region image to sample for dst image
          Mat dst: destination image, segmented, colored regions, bound box and first Hu Moment display
          Mat stats: stats for bound box coordinates
          int nLabels, number of regions in given frame
   Returns:   Hu momoment feature vector of floating point numbers
*/
int compute_features(Mat src, Mat& dst, vector<float>& features) {
    dst = src.clone();

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    Mat gray_pic;
    cvtColor(src, gray_pic, COLOR_BGR2GRAY);
    findContours(gray_pic, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());

    for (size_t i = 0; i < contours.size(); i++) {
        region_features tmp;

        // Calculating Moments for each contour
        Moments moment = moments(contours[i], false);

        // Calculating Hu Moments
        double hu[7];
        HuMoments(moment, hu);

        // Log transform Hu Moments for normalization and add them to the vector
        for (int i = 0; i < 7; i++) {

            hu[i] = -1 * copysign(1.0, hu[i]) * log10(abs(hu[i]));
            features.push_back(hu[i]);
        }

        // Store the transformed Hu Moments in the struct as well
        tmp.huMoments.assign(hu, hu + 7);

        // Calculating centroid
        Point2f centroid(moment.m10 / moment.m00, moment.m01 / moment.m00);

        // Calculate minimum area rectangle and its properties
        RotatedRect minAreaRec = minAreaRect(contours[i]);
        Point2f rect_points[4];
        minAreaRec.points(rect_points);
        for (int j = 0; j < 4; j++) {
            line(dst, rect_points[j], rect_points[(j + 1) % 4], Scalar(0, 255, 0), 2, LINE_8); // Drawing green rotated bounding box
        }

        // Assuming width > height for major and minor axis calculation
        float width = max(minAreaRec.size.width, minAreaRec.size.height);
        float height = min(minAreaRec.size.width, minAreaRec.size.height);

        // Calculate and draw axis line for each region
        double angle = minAreaRec.angle;
        if (minAreaRec.size.width < minAreaRec.size.height) angle += 90.0; // Adjust angle if height is the major axis
        double length = width; // Length of the axis line is consistent
        Point2f endPoint(centroid.x + length * cos(angle * CV_PI / 180.0), centroid.y + length * sin(angle * CV_PI / 180.0));
        line(dst, centroid, endPoint, Scalar(255, 0, 0), 2, LINE_8); // Drawing red axis line

        // Store ratio and percent filled in the struct
        float ratio = width / height;
        float percent_filled = moment.m00 / (width * height);

        features.push_back(ratio);
        features.push_back(percent_filled);


        // Annotate features on the dst image
        stringstream ss;
        ss << "Region " << i + 1 << ": Ratio=" << ratio << ", Percent Filled=" << percent_filled;
        putText(dst, ss.str(), Point(10, 30 + static_cast<int>(i) * 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);

    }
    return 0; 
}



//Calculate the scaled Euclidean Distance of two features: |x1-x2|/st_dev
float scaledEuclideanDis(std::vector<float>& feature1, std::vector<float>& feature2, std::vector<float>& deviations) {
    float distance = 0.0;
    for (int i = 0; i < feature1.size(); i++) {
        distance += std::sqrt((feature1[i] - feature2[i]) * (feature1[i] - feature2[i]) / deviations[i]);//here is the square root
    }
    return distance;
}

//Calculate the standard deveation for each entry of the features in the database(for sacle), the result is not square rooted until next function.
int standardDeviation(std::vector<std::vector<float>>& data, std::vector<float>& deviations) {
    std::vector<float> sums = std::vector<float>(data[0].size(), 0.0); //sum of each entry
    std::vector<float> avgs = std::vector<float>(data[0].size(), 0.0); //average of each entry
    std::vector<float> sumSqure = std::vector<float>(data[0].size(), 0.0); //sum of suqared difference between x_i and x_avg
    deviations = std::vector<float>(data[0].size(), 0.0); //final result(deviations not square rooted yet)

    //first loop for the sum of each entry
    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[0].size(); j++) {
            sums[j] += data[i][j];
        }
    }

    //calculate the avgs
    for (int i = 0; i < sums.size(); i++) {
        avgs[i] = sums[i] / data.size(); //average
    }

    //second loop, for the sum of  suqared difference of  each entry
    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[0].size(); j++) {
            sumSqure[j] += (data[i][j] - avgs[j]) * (data[i][j] - avgs[j]);
        }
    }

    //the deviations
    for (int i = 0; i < sumSqure.size(); i++) {
        deviations[i] = sumSqure[i] / (data.size() - 1);
    }

    return 0;
}

/*
* Simple classifier using Scaled euclidean distance
* vector<vector<double>>  &features: Number of features read from csv
* returns predicted label
*/
string classify(vector<float>& features) {

    //char fileName[256] = "D:/My source/Spring2024/PRCV/c++/recognition/RT2D/database.csv";
    char fileName[256] = "C:/Users/visar/Desktop/OneDrive - Northeastern University/PRCV/RT3D/RT3D/database.csv";
    
    //std::vector<std::string> labels;
    std::vector<char*> labels;
    std::vector<std::vector<float>> nfeatures;
    read_image_data_csv(fileName, labels, nfeatures, 0);
    
    double min_dist = std::numeric_limits<double>::infinity();
    string min_label;
    std::vector<float> deviations;
    standardDeviation(nfeatures, deviations);
    for (int i = 0; i < nfeatures.size(); i++) {

        double dist = scaledEuclideanDis(nfeatures[i], features,deviations);
        if (dist < min_dist) {
           
            min_dist = dist;
            min_label = labels[i];
        }
        /*cout << dist;*/
    }
    return min_label;
}

/*-------------------------------------------------- DNN-----------------------------------------*/

/*
  cv::Mat src        thresholded and cleaned up image in 8UC1 format
  cv::Mat ebmedding  holds the embedding vector after the function returns
  cv::Rect bbox      the axis-oriented bounding box around the region to be identified
  cv::dnn::Net net   the pre-trained network
  int debug          1: show the image given to the network and print the embedding, 0: don't show extra info
 */
int getEmbedding(cv::Mat& src, cv::Mat& embedding, cv::Rect& bbox, cv::dnn::Net& net,  int debug) {
    const int ORNet_size = 128;
    cv::Mat padImg;
    cv::Mat blob;

    cv::Mat roiImg = src(bbox);
    int top = bbox.height > 128 ? 10 : (128 - bbox.height) / 2 + 10;
    int left = bbox.width > 128 ? 10 : (128 - bbox.width) / 2 + 10;
    int bottom = top;
    int right = left;

    cv::copyMakeBorder(roiImg, padImg, top, bottom, left, right, cv::BORDER_CONSTANT, 0);
    cv::resize(padImg, padImg, cv::Size(400, 400));

    cv::dnn::blobFromImage(src, // input image
        blob, // output array
        (1.0 / 255.0) / 0.5, // scale factor
        cv::Size(ORNet_size, ORNet_size), // resize the image to this
        128,   // subtract mean prior to scaling
        false, // input is a single channel image
        true,  // center crop after scaling short side to size
        CV_32F); // output depth/type

    net.setInput(blob);
    embedding = net.forward("onnx_node!/fc1/Gemm");


    if (debug) {
        cv::imshow("pad image", padImg);
        std::cout << embedding << std::endl;
        cv::waitKey(0);
    }

    return(0);
}

float euclideanDistance(vector<float>& f1, vector<float>& f2) {

    float sum = 0;
    for (int i = 0; i < f1.size(); i++)
        sum += pow((f1[i] - f2[i]), 2);
    return sqrt(sum);
}

/*
* Simple classifier using euclidean distance
* vector<vector<double>>  &features: Number of features read from csv
* returns predicted label
*/
string classifyDNN(vector<float>& features) {

    //char fileName[256] = "D:/My source/Spring2024/PRCV/c++/recognition/RT2D/database.csv";
    char fileName[256] = "C:/Users/visar/Desktop/OneDrive - Northeastern University/PRCV/RT3D/RT3D/database.csv";

    //std::vector<std::string> labels;
    std::vector<char*> labels;
    std::vector<std::vector<float>> nfeatures;
    read_image_data_csv(fileName, labels, nfeatures, 0);

    double min_dist = std::numeric_limits<double>::infinity();
    string min_label;
    std::vector<float> deviations;
    //standardDeviation(nfeatures, deviations);
    for (int i = 0; i < nfeatures.size(); i++) {

        double dist = euclideanDistance(nfeatures[i], features);
        if (dist < min_dist) {

            min_dist = dist;
            min_label = labels[i];
        }
        /*cout << dist;*/
    }
    
    return min_label;
}













/*--------------------------KNN------------------------------------------------------*/

//class Data {
//public:
//    float distance;
//    std::string label;
//    std::vector<float> features;
//    Data() {
//        distance = 0;
//    }
//};
//
//float euclideanDistance(std::vector<float>& f1, std::vector<float>& f2) {
//    double sum = 0;
//    for (int i = 0; i < f1.size(); i++)
//        sum += pow((f1[i] - f2[i]), 2);
//    return sqrt(sum);
//}
//
//bool cmp(Data& a, Data& b) {
//    return a.distance < b.distance;
//}
//
//double manhattanDistance(std::vector<double>& f1, std::vector<double>& f2) {
//    double sum = 0;
//    for (int i = 0; i < f1.size(); i++)
//        sum += abs(f1[i] - f2[i]);
//    return sum;
//}
//
//void fillDistances(std::vector<float>& query, std::vector<float>& nfeatures, char*& labels, std::vector<Data>& data) {
//    for (int i = 0; i < labels.size(); i++) {
//        Data data_point;
//        data_point.label = labels[i];
//        data_point.features = nfeatures[i];
//        data_point.distance = euclideanDistance(data_point.features, query[0]);
//        data.push_back(data_point);
//    }
//}
//
//string KNN(std::vector<float>& feature, int k) {
//
//    //char fileName[256] = "D:/My source/Spring2024/PRCV/c++/recognition/RT2D/database.csv";
//    char fileName[256] = "C:/Users/visar/Desktop/OneDrive - Northeastern University/PRCV/RT3D/RT3D/database.csv";
//
//    //std::vector<std::string> labels;
//    std::vector<char*> labels;
//    std::vector<std::vector<float>> nfeatures;
//    read_image_data_csv(fileName, labels, nfeatures, 0);
//
//    vector<Data> data;
//
//    //filling the distances between all points and test
//    fillDistances(feature, nfeatures, labels, data);
//
//    //sorting so that we can get the k nearest
//    sort(data.begin(), data.end(), cmp);
//
//    if (data[0].distance > 10)
//        return "unknown";
//
//    map<string, int> count;
//    int max = -1;
//    string mode_label;
//
//    for (int i = 0; i < k; i++) {
//        count[data[i].label] += 1;
//        if (count[data[i].label] > max) {
//            max = count[data[i].label];
//            mode_label = data[i].label;
//        }
//        //cout << "label: " << data[i].label << " distance: " << data[i].distance << endl;
//    }
//
//    return mode_label;
//}