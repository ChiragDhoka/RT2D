//main.cpp
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "processing.h"
#include "csv_util.h"

using namespace std;
using namespace cv;

//Path to CSV File
char CSV_FILE[256] = "C:/Users/visar/Desktop/OneDrive - Northeastern University/PRCV/RT3D/RT3D/database.csv";

//path to Image File
char IMAGE[256] = "C:/Users/visar/Desktop/OneDrive - Northeastern University/PRCV/RT3D/RT3D/Training_Data/Remote4.jpeg";

//Declaring all the frames
Mat originalFrame, thresholdingFrame, cleanUpFrame, segmentedFrame, colorSegmentedFrame, featureImageFrame, imageMat;

// Store connectcomponents() parameters 
Mat labels, stats, centroids;
int image_nLabels;

//Vector to store images
vector<region_features> features;

int main(int argc, char* argv[]) {
    
    //Thresholding Parameter
    int threshold = 110;

    imageMat = imread(IMAGE);

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

        vector<float> feature;

        imshow("Target Image", imageMat);

        *capdev >> originalFrame;
        if (originalFrame.empty()) {
            cerr << "Frame is empty" << endl;
            break;
        }
        //imshow("Video", originalFrame);

        feature.clear();

        // Process image with thresholding and cleanup
        thresholdingFrame = thresholding(imageMat, threshold);
        cleanUpFrame = morphological_operation(thresholdingFrame, cleanUpFrame);
        segmentedFrame = segment(cleanUpFrame, segmentedFrame, colorSegmentedFrame, labels, stats, centroids);
        compute_features(segmentedFrame, featureImageFrame, feature);

        //imshow("After Thresholding", thresholdingFrame);
        imshow("Clean Image", cleanUpFrame);
        imshow("Segmented image", segmentedFrame);
        imshow("Boxes and axis", featureImageFrame);

        // Exit loop if 'q' is pressed
        char c = (char)waitKey(5);
        if (c == 'n' || c == 'N') {
            cout << "N pressed: " << endl;
            char label[20];
            cout << "Enter the Label for the object : " << endl;
            cin >> label;

            append_image_data_csv(CSV_FILE, label, feature);
        }
        if (c == 'q' || c == 27 || c == 'Q') { // 27 is ASCII for ESC
            break;
        }
    }

    // Clean up
    capdev->release();
    destroyAllWindows();

    return 0;
}
