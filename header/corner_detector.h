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
    float feature_angle;
};

struct MarkerInfo{
    int markerID = -1;
    vector<int> featurePos, feature_ID;
    vector<array<Point2f, 8>> cornerLists;
    vector<Point2f> feature_center;
    vector<float> edge_length;
};

struct corners_pre{
    Point2f intersect;
    float dis_to_centroid;
    float angle_to_centroid;
};

struct pos_with_ID{
    vector<int> pos;
    int ID = -1;
};

class corner_detector{
public:
    void adaptiveThreshold(const Mat& src, Mat& dst, int thresholdWindow = 5);
    
    void connectedComponentLabeling(const Mat& src, vector<vector<Point>>& quadArea, int method = 0);
    
    void edgeExtraction(const Mat& img, vector<vector<Point>>& quadArea, vector<vector<Point2f>>& corners_init, int KMeansIter = 5);
    
    bool quadJudgment(vector<corners_pre>& corners, int areaPixelNumber);

    void edgeSubPix(const Mat& src, vector<vector<Point2f>>& corners_init, vector<vector<Point2f>>& corners_refined, int subPixWindow);

    bool parallelogramJudgment(vector<Point2f> corners);

    void featureRecovery(vector<vector<Point2f>>& corners_refined, vector<featureInfo>& features);
    
    featureInfo featureOrganization(vector<Point2f> quad1, vector<Point2f> quad2, Point2f quad1_center, Point2f quad2_center, float feature_angle);

    void featureExtraction(const Mat& img, vector<featureInfo> feature_src, vector<featureInfo> feature_dst);

    void markerOrganization(vector<featureInfo> feature, vector<MarkerInfo> markers);

    void markerDecoder(vector<MarkerInfo> markers_src, vector<MarkerInfo> markers_dst, Mat1i& state);

private:
    // Adaptive threshold use
    Mat img_part, min_part, max_part;
    int row_count = 0, col_count = 0, row_count_final = 1, col_count_final = 1;
    double maxVal, minVal;

    // CCL use
    Mat img_labeled, stats, centroids;
    int nccomp_area = 0;

    // Edge extraction use
    long long sum_x, sum_y;
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
    float threshold_RAC = 0.5;
    vector<Point2f> corners_pass;

    // Edge refinement use


    // Para Judgment use
    Point2f corner_center;
    float diff_percentage = 0.02, dist_to_center[4];
    float coeff_1, coeff_2;

    // Feature recovery
    vector<Point2f> corner_centers;
    vector<array<float, 4>> corner_dist;
    vector<float> corner_angles_1, corner_angles_2;
    float feature_angle, feature_half_length, threshold_angle = 5, dist1_short, dist1_long, dist2_short, dist2_long;
    bool isVisited[1000], tag1, tag2;

    // Feature organization
    featureInfo Fea;
    float middle_pos, angle_quad1[4], angle_quad2[4];

    // Feature extraction
    float distance_2points(Point2f point1, Point2f point2);
    float cross_ratio_1, cross_ratio_2, length_1[4], length_2[4];
    bool label_area, label_instruct;

    // Marker organization
    int union_find(int input);
    float area(featureInfo feature);
    int father[100];
    float area_ratio = 0.1, threshold_vertical = 0.5, center_angle;
    Point2f vector_center, vector_longedge;
    int pose[4] = {0, 1, 4, 5}, cnt;
    vector<int> father_database;
    vector<vector<int>> marker_ID;

    // Marker decoder
    float marker_angle, edge_last, edge_now, dist_center;
    int code[50];
    pos_with_ID Pos_ID;
    pos_with_ID match_dictionary(int *code, Mat1i& state);
    bool find_father;
};

#endif