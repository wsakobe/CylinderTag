#pragma once
#ifndef CORNER_DETECTOR_H
#define CORNER_DETECTOR_H

#include "config.h"

using namespace std;
using namespace cv;

struct featureInfo{
    vector<Point2f> corners;
    int ID = -1;
    Point2f feature_center;
    double feature_angle;
};

struct MarkerInfo{
    int markerID = -1;
    vector<int> featureID;
    vector<int> featurePos;
    vector<array<Point2f, 8>> cornerLists;
};

class corner_detector{
public:
    void adaptiveThreshold(const Mat& src, Mat& dst, int thresholdWindow = 5);
    
    void connectedComponentLabeling(const Mat& src, vector<vector<Point>> quadArea, int method = 0);
    
    void edgeExtraction(const Mat& img, vector<vector<Point>> quadArea, vector<Point2f> corners_init, int KMeansIter = 5);
    
    bool quadJudgment(vector<Point2f> corners_init, int areaPixelNumber, int threshold = 0.5);

    void edgeSubPix(const Mat& src, vector<Point2f> corners_init, vector<Point2f> corners_refined, int subPixWindow);

    bool parallelogramJudgment(vector<Point2f> corners);

    void featureRecovery(vector<vector<Point2f>> corners_refined, vector<featureInfo> features);

    void featureExtraction(const Mat& img, vector<vector<Point2f>> feature_src, vector<featureInfo> feature_dst);

    void markerOrganization(vector<featureInfo> feature, vector<MarkerInfo> markers);

    void markerDecoder(vector<MarkerInfo> markers_src, vector<MarkerInfo> markers_dst);

private:
    // Adaptive threshold use
    Mat img_part, min_part, max_part;
    int row_count = 0, col_count = 0, row_count_final = 1, col_count_final = 1;
    double maxVal, minVal;

    // CCL use
    Mat img_labeled, stats, centroids;
    int nccomp_area = 0;
};

#endif