//processing.cpp
#include <iostream>
#include <vector>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/*Stores the features of a region */
struct region_features {
    double ratio;
    double percent_filled;
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

    Mat erode = src.clone();
    auto structElem = createStructuringElement(3, 3);

    // Perform custom erosion and dilation
    Mat erodedImage = customErode(erode, structElem);
    Mat dilated = erodedImage.clone();
    Mat dilatedImage = customDilate(dilated, structElem);

    dst = dilatedImage;

    return dst;
}


/*------------------ TASK 3 ------------------------------------------------------------------*/
/*Function for connected component analysis, creates segmented, region-colored version of the src image
  Parameters: a src image to be sampled from, then Mat data types for labels, stats, and centroid calculation.
  Returns: a segmented, region colored version of the src image
*/
Mat segment(Mat src, Mat& dst, Mat& colored_dst, Mat& labels, Mat& stats, Mat& centroids) {

    std::cout << "originalFrame channels: " << src.channels() << std::endl;

    //Mat gray_pic;
    //cvtColor(src, gray_pic, COLOR_BGR2GRAY);
    //std::cout << "originalFrame channels: " << gray_pic.channels() << std::endl;

    int num = connectedComponentsWithStats(src, labels, stats, centroids, 8);

    std::cout << num << std::endl;

    // number of colors will equal to number of regions
    vector<Vec3b> colors(num);
    vector<Vec3b> intensity(num);
    // set background to black
    colors[0] = Vec3b(0, 0, 0);
    intensity[0] = Vec3b(0, 0, 0);
    int area = 0;
    for (int i = 1; i < num; i++) {
        colors[i] = Vec3b(120*i, 140*i, 256*i);
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
int compute_features(Mat src, Mat& dst, vector<region_features>& features) {

    dst = src.clone();

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    Mat gray_pic;
    cvtColor(src, gray_pic, COLOR_BGR2GRAY);
    findContours(gray_pic, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());

    Moments moment;
    Point2f center;
    Point2f vertex[4];
    for (int i = 0; i < contours.size(); i++) {
        region_features tmp;

        // calculate moment, center, vertices of bounding box
        moment = moments(contours[i]);
        center = Point2f(moment.m10 / moment.m00,
            moment.m01 / moment.m00);
        RotatedRect rect = minAreaRect(contours[i]);
        double angle = 0.5 * atan((2 * moment.m11) / (moment.m20 - moment.m02));
        int len = max(rect.size.height, rect.size.width);
        int x1 = (moment.m10 / moment.m00) + len / 2 * cos(angle);
        int y1 = (moment.m01 / moment.m00) - len / 2 * sin(angle);
        int x2 = (moment.m10 / moment.m00) - len / 2 * cos(angle);
        int y2 = (moment.m01 / moment.m00) + len / 2 * sin(angle);
        line(dst, Point2f(x1, y1), Point2f(x2, y2), Scalar(0, 0, 255), 2, LINE_8);

        vertex[4];
        rect.points(vertex);
        line(dst, vertex[0], vertex[1], Scalar(0, 0, 255), 2, LINE_8);
        line(dst, vertex[1], vertex[2], Scalar(0, 0, 255), 2, LINE_8);
        line(dst, vertex[2], vertex[3], Scalar(0, 0, 255), 2, LINE_8);
        line(dst, vertex[3], vertex[0], Scalar(0, 0, 255), 2, LINE_8);

        tmp.ratio = max(rect.size.height, rect.size.width) / min(rect.size.height, rect.size.width);
        tmp.percent_filled = moment.m00 / (rect.size.height * rect.size.width);
        features.push_back(tmp);
    }

    return 0;
}

//int compute_features(Mat src, Mat& dst, vector<region_features>& features) {
//    dst = src.clone();
//
//    // Find connected components and retrieve statistics
//    Mat labels, stats, centroids;
//    int num_regions = connectedComponentsWithStats(src, labels, stats, centroids);
//
//    for (int i = 1; i < num_regions; ++i) {  // Start from 1 to skip background (0)
//        region_features tmp;
//
//        // Extract properties of the current region
//        int area = stats.at<int>(i, CC_STAT_AREA);
//        double aspect_ratio = static_cast<double>(stats.at<int>(i, CC_STAT_WIDTH)) / stats.at<int>(i, CC_STAT_HEIGHT);
//        double percent_filled = static_cast<double>(area) / (stats.at<int>(i, CC_STAT_WIDTH) * stats.at<int>(i, CC_STAT_HEIGHT));
//
//        // Store computed features
//        tmp.ratio = aspect_ratio;
//        tmp.percent_filled = percent_filled;
//        features.push_back(tmp);
//    }
//
//    return 0;
//}