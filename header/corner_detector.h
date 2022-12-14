#pragma once
#ifndef CORNER_DETECTOR_H
#define CORNER_DETECTOR_H

#include "config.h"

using namespace std;
using namespace cv;
using namespace ceres;

struct featureInfo{
    vector<Point2f> corners;
    int ID = -1;
    Point2f feature_center;
    float feature_angle, cross_ratio;
    bool firstDarker;
};

struct MarkerInfo{
    int markerID = -1;
    vector<int> featurePos, feature_ID;
    vector<vector<Point2f>> cornerLists;
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
    corner_detector();
    ~corner_detector();

    void adaptiveThreshold(const Mat& src, Mat& dst, int thresholdWindow = 5);
    
    void connectedComponentLabeling(const Mat& src, vector<vector<Point>>& quadArea, int method = 0);
    
    void edgeExtraction(const Mat& img, vector<vector<Point>>& quadArea, vector<vector<Point2f>>& corners_init, vector<double>& meanG, int KMeansIter = 5);
    
    bool quadJudgment(vector<corners_pre>& corners, int areaPixelNumber);

    bool parallelogramJudgment(vector<Point2f> corners);

    void featureRecovery(vector<vector<Point2f>>& corners_refined, vector<featureInfo>& features, vector<double>& meanG);

    void edgeSubPix(const Mat& src, vector<featureInfo>& features, vector<featureInfo>& features_refined, int subPixWindow);

    void buildProblem(Problem* problem, vector<Point> inlier_points, vector<float> inlier_pixels);

    featureInfo featureOrganization(vector<Point2f> quad1, vector<Point2f> quad2, Point2f quad1_center, Point2f quad2_center, float feature_angle, bool darker);

    void featureExtraction(const Mat& img, vector<featureInfo>& feature_src, vector<featureInfo>& feature_dst);

    void markerOrganization(vector<featureInfo> feature, vector<MarkerInfo>& markers);

    void markerDecoder(vector<MarkerInfo> markers_src, vector<MarkerInfo> markers_dst, Mat1i& state, int featureSize);

private:
    // Adaptive threshold use
    Mat img_part, min_part, max_part;
    int row_count, col_count, row_count_final, col_count_final;
    double maxVal, minVal;

    // CCL use
    Mat img_labeled, stats, centroids;
    bool illegal[1000];
    int nccomp_area = 0;

    // Edge extraction use
    int x_min, x_max, y_min, y_max; // mask size
    long long sum_x, sum_y;
    
    vector<Point> edge_point; 
    Point starter;
    int x_bias[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    int y_bias[8] = {-1, -1, 0, 1, 1, 1, 0, -1};

    vector<float> dist2center, dist2line;
    int init, cnt_boundary;
    float normal_line[2];
    float d_line, dist_expand, threshold_line = 1.5, threshold_expand = 0.8, cost;
    Point2f area_center;
    vector<int> span, span_temp;
    vector<int> b;

    vector<int> expand_line(vector<Point> edge_point, int init, int end);
    bool find_edge_left, find_edge_right;
    int left, right;
    
    array<vector<float>, 4> edge_angle_cluster, line_func;
    array<vector<Point>, 4> edge_point_cluster;

    vector<corners_pre> corners_p;    
    corners_pre c;

    bool flag_line_number, flag_illegal_corner;

    // Quad judgment
    float quad_area, RAC;
    float threshold_RAC = 0.5;
    vector<Point2f> corners_pass;

    // Edge refinement use
    Problem problem;
    vector<Point2f> contours;
    vector<Point> inlier_points;
    vector<float> inlier_pixels;
    float width, pixel_high_low[2], mean_pixel[2], dist;
    double line_function[3]; 
    int count[2], direction;
    
    // Para Judgment use
    Point2f corner_center;
    float diff_percentage = 0.02, dist_to_center[4];
    float coeff_1, coeff_2;

    // Feature recovery
    vector<Point2f> corner_centers;
    vector<array<float, 4>> corner_dist;
    vector<float> corner_angles_1, corner_angles_2;
    float feature_angle, feature_half_length, threshold_angle = 5, dist1_short, dist1_long, dist2_short, dist2_long;
    bool isVisited[1000], tag1, tag2, tag_darker;

    // Feature organization
    featureInfo Fea;
    float middle_pos, angle_quad1[4], angle_quad2[4];

    // Feature extraction
    float distance_2points(Point2f point1, Point2f point2);
    float cross_ratio_1, cross_ratio_2, cross_ratio, length_1[4], length_2[4];
    bool label_area, label_instruct;
    float ID_cr_correspond[9] = {1.45, 1.54, 1.63, 1.72, 1.8, 1.72, 1.63, 1.54, 1.45};
    bool instruct[9] = {1, 1, 1, 1, 1, 0, 0, 0, 0}, tag_length;

    // Marker organization
    int union_find(int input);
    float area(featureInfo feature);
    int father[100];
    float area_ratio = 0.5, threshold_vertical = 0.5, center_angle;
    Point2f vector_center, vector_longedge;
    int pose[4] = {0, 1, 4, 5}, cnt;
    vector<int> father_database;
    vector<vector<int>> marker_ID;

    // Marker decoder
    float marker_angle, edge_last, edge_now, dist_center;
    int code[50], pos_now, max_coverage, second_coverage, coverage_now, direc;
    Point max_coverage_pos;
    pos_with_ID Pos_ID;
    pos_with_ID match_dictionary(int *code, Mat1i& state, int length);
    bool find_father;
};

#endif