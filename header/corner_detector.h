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

struct corners_pre{
    Point2f intersect;
    float dis_to_centroid;
    float angle_to_centroid;
};

class corner_detector{
public:
    void adaptiveThreshold(const Mat& src, Mat& dst, int thresholdWindow = 5);
    
    void connectedComponentLabeling(const Mat& src, vector<vector<Point>>& quadArea, int method = 0);
    
    void edgeExtraction(const Mat& img, vector<vector<Point>>& quadArea, vector<Point2f>& corners_init, int KMeansIter = 5);
    
    bool quadJudgment(vector<corners_pre> corners, int areaPixelNumber);

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

    // Edge extraction use
    Moments moment;
    vector<float> edge_angle;
    array<vector<float>, 4> edge_angle_cluster, line_func;
    vector<Point> edge_point;
    array<vector<Point>, 4> edge_point_cluster;
    vector<int> kmeans_label;
    float Gmax = -1, Gmin = 1, edge_angle_all = 0;
    double measure;
    vector<corners_pre> corners_p;    
    corners_pre c;
    Point2f area_center;
    bool flag_line_number, flag_illegal_corner;

    // Quad judgment
    float quad_area, RAC;
    float threshold_RAC = 10000;

    // Edge refinement use


    // Para Judgment use
    Point2f corner_center;
    float diff_percentage = 0.02, dist_to_center[4];
    float coeff_1, coeff_2;
};

#endif