//main.cpp
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "processing.h"
#include "csv_util.h"

using namespace std;
using namespace cv;

//std::vector<std::vector<float>> convert_features_to_float(const std::vector<double>& features) {
//    std::vector<std::vector<float>> float_features;
//    for (const auto& feature : features) {
//        std::vector<float> float_feature;
//        float_feature.push_back(feature.ratio); // Assuming ratio is a float
//        float_feature.push_back(feature.percent_filled); // Assuming percent_filled is a float
//        // Add other feature values as needed
//        float_features.push_back(float_feature);
//    }
//    return float_features;
//}

//struct region_features {
//    float ratio;
//    float percent_filled;
//};

int main(int argc, char* argv[]) {

    int threshold = 110;
    vector<region_features> features;
    char CSV_FILE[256] = "C:/Users/visar/Desktop/OneDrive - Northeastern University/PRCV/RT3D/RT3D/database.csv";
    std::vector<char*> filenames;

    Mat originalFrame, thresholdingFrame, cleanUpFrame, segmentedFrame, colorSegmentedFrame, featureImageFrame;

    // Store connectcomponents() parameters 
    Mat labels, stats, centroids;
    int image_nLabels;

    //test on images 
    string IMAGE_FILE = "C:/Users/visar/Desktop/img1p3.png";
    //string CSV_FILE = "C:/Users/visar/Desktop/OneDrive - Northeastern University/PRCV/RT3D/RT3D/database.csv";

    Mat imageMat = imread(IMAGE_FILE);

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
        imshow("Video", originalFrame);

        // Process image with thresholding and cleanup
        thresholdingFrame = thresholding(originalFrame, threshold);
        cleanUpFrame = morphological_operation(thresholdingFrame, cleanUpFrame);
        segmentedFrame = segment(cleanUpFrame, segmentedFrame, colorSegmentedFrame, labels, stats, centroids);
        feature.clear();
        int temp = compute_features(segmentedFrame, featureImageFrame, feature);

        imshow("After Thresholding", thresholdingFrame);
        //imshow("Clean Image", cleanUpFrame);
        imshow("Segmented image", segmentedFrame);
        //imshow("Colored Segmented image", colorSegmentedFrame);
        
        // Store the ID and feature vector of the image
        struct ImageData *currentImage = (struct ImageData*)malloc(sizeof (struct ImageData*));

        // Store the huMoments vector as the database image's feature vector

        imshow("Boxes and axis", featureImageFrame);

        // Exit loop if 'q' is pressed
        char c = (char)waitKey(5);
        vector<ImageData*> csvData;

        if (c == 'n' || c == 'N') {
            cout << "N pressed: " << endl;
            char label[20];
            cout << "Enter the Label for the object : " << endl;
            cin >> label;

            //ImageData *img = new ImageData();
            //img->label = label;
            //img->featureVector = allHuMoments;
            ////currentImage->label = label;
            ////currentImage->featureVector = allHuMoments;

            //
            ////append_image_data_csv(CSV_FILE, "label", currentImage->featureVector, 0);

            ////append_image_data_csv(CSV_FILE, "huMoments", allHuMoments, 0);
            //
            //// Append features to CSV file      
            //std::vector<std::vector<float>> features_data = convert_features_to_float(feature);
            //img->webCamData = features_data[0];

            //csvData.push_back(img);
            //append_image_data_csv(CSV_FILE, label, img, 0);
            append_image_data_csv(CSV_FILE, label, feature, 0);
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
