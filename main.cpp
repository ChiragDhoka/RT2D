//main.cpp
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "processing.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {

    int threshold = 75;
    vector<region_features> features;

    Mat originalFrame, thresholdingFrame, cleanUpFrame, segmentedFrame, colorSegmentedFrame, featureImageFrame;

    // Store connectcomponents() parameters 
    Mat labels, stats, centroids;

    //test on images 
    string image = "D:/My source/Spring2024/PRCV/Project 1 images/img2P3.png";

    Mat imageMat = imread(image);

    // Open the video device
    VideoCapture* capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return -1;
    }
    else {
        printf("Stream Started!\n");
    }

    // Get properties of the image
    Size refS((int)capdev->get(CAP_PROP_FRAME_WIDTH), (int)capdev->get(CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    // Create a window to display the video
    namedWindow("Video", 1);

    while (true) {
        imshow("Target Image", imageMat);

        *capdev >> originalFrame;
        if (originalFrame.empty()) {
            cerr << "Frame is empty" << endl;
            break;
        }
        imshow("Video", originalFrame);

        // Process image with thresholding and cleanup
        thresholdingFrame = thresholding(imageMat, threshold);
        cleanUpFrame = morphological_operation(thresholdingFrame, cleanUpFrame);
        segmentedFrame = segment(cleanUpFrame, segmentedFrame, colorSegmentedFrame, labels, stats, centroids);

        imshow("After Thresholding", thresholdingFrame);
        imshow("Clean Image", cleanUpFrame);
        imshow("Segmented image", segmentedFrame);
        imshow("Colored Segmented image", colorSegmentedFrame);

        features.clear();
        compute_features(segmentedFrame, featureImageFrame, features);
        // Print region features on the screen
        for (int i = 0; i < features.size(); ++i) {
            region_features region = features[i];
            // Print ratio and percent_filled values on the screen
            std::stringstream ss;
            ss << "Region " << i << ": Ratio=" << region.ratio << ", Percent Filled=" << region.percent_filled;
            cv::putText(featureImageFrame, ss.str(), cv::Point(10, 30 + i * 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }

        // Display the image with text
        imshow("Boxes and axis", featureImageFrame);


        // Exit loop if 'q' is pressed
        char c = (char)waitKey(25);
        if (c == 'q' || c == 27 || c == 'Q') { // 27 is ASCII for ESC
            break;
        }
    }

    // Clean up
    capdev->release();
    destroyAllWindows();

    return 0;
}