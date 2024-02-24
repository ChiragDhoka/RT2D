//processing.cpp
#include <iostream>
#include <vector>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "csv_util.h"

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

    //Mat gray_pic;
    //cvtColor(src, gray_pic, COLOR_BGR2GRAY);
    //std::cout << "originalFrame channels: " << gray_pic.channels() << std::endl;

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
        
        colors[i] = Vec3b(255*i % 256, 170*i % 256, 200*i % 256);
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
        
    }

    //// Optionally, annotate features on the dst image
    //for (size_t i = 0; i < features.size(); ++i) {
    //    stringstream ss;
    //    ss << "Region " << i + 1 << ": Ratio=" << features[i].ratio << ", Percent Filled=" << features[i].percent_filled;
    //    putText(dst, ss.str(), Point(10, 30 + static_cast<int>(i) * 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    //}

    return 0; // Return the flat list of all Hu Moments
}

/* Computes huMoment feature vector, computes and displays oriented bounding box
   Paramters: Mat src: binary image to be sampled from
          Mat src_regions: colored_region image to sample for dst image
          Mat dst: destination image, segmented, colored regions, bound box and first Hu Moment display
          Mat stats: stats for bound box coordinates
          int nLabels, number of regions in given frame
   Returns:   Hu momoment feature vector of floating point numbers
*/
//std::vector<float> feature_computation(cv::Mat& src, cv::Mat& src_regions, cv::Mat& dst, cv::Mat stats, int nLabels) {
//
//    // Necessary to get nLabels and stats, will refactor out later
//    cv::Mat labels;
//    cv::Mat centroids;
//
//    // Output of frame with bounding box
//    dst = cv::Mat::zeros(src.size(), CV_8UC3);
//    dst = src_regions;
//
//    int min_size = 400;
//
//
//    // Calculate moments
//    cv::Moments moments = cv::moments(src, false);
//    // Calculate Hu Moments
//    double huMoments[7];
//    cv::HuMoments(moments, huMoments);
//    // Resulting hu moments have a HUGE range, use a log transform to bring them to same range
//    for (int i = 0; i < 7; i++) {
//        huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]));
//    }
//
//
//    //Use openCV's connectedComponentsWithStats
//    // returns the a 4 tuple of the total number of unique labels
//    //         a mask named labels that has the same spacial dimensions as our input image
//    //         stats: statistics on each connected component, including bound box coords and area
//    //         centroids: x,y coords of each connected component
//    nLabels = cv::connectedComponentsWithStats(src, labels, stats, centroids);
//
//    // Obtain bound box coords for each object
//    // Draw bound box for each object
//    for (int label = 1; label < nLabels; label++) {
//        if (stats.at<int>(label, cv::CC_STAT_AREA) >= min_size) {
//            std::vector<cv::Point> points;
//            // Store the point for the top left vertex
//            cv::Point top_left = cv::Point(stats.at<int>(label, cv::CC_STAT_LEFT), stats.at<int>(label, cv::CC_STAT_TOP));
//            points.push_back(top_left);
//            // Store the point for the top right vertex
//            cv::Point top_right = cv::Point(stats.at<int>(label, cv::CC_STAT_WIDTH) + stats.at<int>(label, cv::CC_STAT_LEFT), stats.at<int>(label, cv::CC_STAT_TOP));
//            points.push_back(top_right);
//            // Store the point for the bottom left vertex
//            cv::Point bottom_left = cv::Point(stats.at<int>(label, cv::CC_STAT_LEFT), stats.at<int>(label, cv::CC_STAT_TOP) + stats.at<int>(label, cv::CC_STAT_HEIGHT));
//            points.push_back(bottom_left);
//            // Store the point for the bottom right vertex
//            cv::Point bottom_right = cv::Point(stats.at<int>(label, cv::CC_STAT_WIDTH) + stats.at<int>(label, cv::CC_STAT_LEFT), stats.at<int>(label, cv::CC_STAT_HEIGHT) + stats.at<int>(label, cv::CC_STAT_TOP));
//
//            // Create the box based on the coordinates
//            cv::RotatedRect box = cv::minAreaRect(cv::Mat(points));
//
//            // Draw the bound box
//            cv::Point2f vertices[4];
//            box.points(vertices);
//            for (int i = 0; i < 4; ++i) {
//                cv::line(dst, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
//            }
//
//
//            // Display first region HuMoment for each region underneath the bounding box
//            std::string firstHu = std::to_string(huMoments[0]);
//            std::string text = "hu[0] =" + firstHu;
//            cv::Point textCoords = (cv::Point(stats.at<int>(label, cv::CC_STAT_LEFT), stats.at<int>(label, cv::CC_STAT_TOP) + stats.at<int>(label, cv::CC_STAT_HEIGHT) + 40));
//            //cv::Font font = cv::FONT_HERSHEY_SIMPLEX;
//            int fontScale = 1;
//            cv::Vec3b textColor = cv::Vec3b(255, 0, 255);
//            int thickness = 2;
//
//            // Draw the text to the dest image
//            cv::putText(dst, text, textCoords, cv::FONT_HERSHEY_SIMPLEX, fontScale, textColor, thickness, cv::LINE_AA);
//        }
//    }
//
//    // Convert the Hu Moment array to a vector for ease of use
//    int n = sizeof(huMoments) / sizeof(huMoments[0]);
//    std::vector<float> huVector(huMoments, huMoments + n);
//    return huVector;
//}

