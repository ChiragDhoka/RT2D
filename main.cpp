//main.cpp
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <map>
#include "processing.h"
#include "csv_util.h"
//#include "knn.h"

using namespace std;
using namespace cv;

bool trainingModeFlag = false;
bool recognizeModeFlag = false;
bool dnnModeFlag = false;

//Path to CSV File
char CSV_FILE[256] = "C:/Users/visar/Desktop/OneDrive - Northeastern University/PRCV/RT3D/RT3D/database.csv";
char CSV_DNN[256] = "C:/Users/visar/Desktop/OneDrive - Northeastern University/PRCV/RT3D/RT3D/database_DNN.csv";
//path to Image File
char IMAGE[256] = "C:/Users/visar/Desktop/OneDrive - Northeastern University/PRCV/RT3D/RT3D/Training_Data/Box1.jpeg";

//Declaring all the frames
Mat originalFrame, thresholdingFrame, cleanUpFrame, segmentedFrame, colorSegmentedFrame, featureImageFrame, imageMat;

// Store connectcomponents() parameters 
Mat labels, stats, centroids;
int image_nLabels;

//Vector to store images
vector<region_features> features;

std::vector<float> matToVector(const cv::Mat& mat) {
    std::vector<float> vec;
    // Ensure that the input matrix is not empty
    if (!mat.empty()) {
        // Iterate through the matrix elements
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                // Push the element into the vector
                vec.push_back(mat.at<float>(i, j));
            }
        }
    }
    return vec;
}

int main(int argc, char* argv[]) {

    //Thresholding Parameter
    int threshold = 50;

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
        vector<float> embeddingFeature;

        imshow("Target Image", imageMat);

        *capdev >> originalFrame;
        if (originalFrame.empty()) {
            cerr << "Frame is empty" << endl;
            break;
        }
        //imshow("Video", originalFrame);

        feature.clear();

        // Process image with thresholding and cleanup
        thresholdingFrame = thresholding(originalFrame, threshold);
        cleanUpFrame = morphological_operation(thresholdingFrame, cleanUpFrame);
        segmentedFrame = segment(cleanUpFrame, segmentedFrame, colorSegmentedFrame, labels, stats, centroids);
        compute_features(segmentedFrame, featureImageFrame, feature);

        //imshow("After Thresholding", thresholdingFrame);
        imshow("Clean Image", cleanUpFrame);
        imshow("Segmented image", segmentedFrame);
        imshow("Boxes and axis", featureImageFrame);

        // Exit loop if 'q' is pressed
        char key = waitKey(5);
        if (key == 'n' || key == 'N') {
            trainingModeFlag = true;
        }
        else if (key == 'r' || key == 'R') {
            recognizeModeFlag = true;
        }
        else if (key == 'k' || key == 'K') {
            dnnModeFlag = true;
        }
        if (key == 'q' || key == 27 || key == 'Q') { // 27 is ASCII for ESC
            break;
        }

        if (trainingModeFlag) {
            cout << "Training Mode " << endl;
            char label[20];
            cout << "Enter the Label for the object : " << endl;
            cin >> label;

            append_image_data_csv(CSV_FILE, label, feature,0);

            cout << "Exit Trainging Mode!" << endl;
            trainingModeFlag = false;

        }
        else if (recognizeModeFlag) {
            cout << "Recognize Mode " << endl;
            string temp1 = classify(feature);
            
            cout << "The Object is: " << temp1 << endl;

            cout << "Exiting Recognize Mode!" << endl;
            recognizeModeFlag = false;
        }
        else if (dnnModeFlag) {
            cout << "KNN Mode" << endl;

            string mod_filename = "C:/Users/visar/Desktop/OneDrive - Northeastern University/PRCV/RT3D/RT3D/or2d-normmodel-007.onnx";
            // read the network
            cv::dnn::Net net = cv::dnn::readNet(mod_filename);
            printf("Network read successfully\n");

            /// print the names of the layers
            std::vector<cv::String> names = net.getLayerNames();

            for (int i = 0; i < names.size(); i++) {
                printf("Layer %d: '%s'\n", i, names[i].c_str());
            }

            // read image and convert it to greyscale
            cv::Mat src = cv::imread(IMAGE);
            cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);

            // the getEmbedding function wants a rectangle that contains the object to be recognized
            cv::Rect bbox(0, 0, src.cols, src.rows);

            // get the embedding
            cv::Mat embedding;
            getEmbedding(src, embedding, bbox, net, 1);  // change the 1 to a 0 to turn off debugging

            embeddingFeature = matToVector(embedding);
            cout << embeddingFeature[0] << endl;
            append_image_data_csv(CSV_DNN, IMAGE, embeddingFeature, 0);

            //string temp2 = classifyDNN(embeddingFeature);

            //cout << "The Object is: " << temp2 << endl;

            //cout << "Exiting KNN Mode" << endl;

            dnnModeFlag = false;
        }
    }

    // Clean up
    capdev->release();
    destroyAllWindows();

    return 0;
}
