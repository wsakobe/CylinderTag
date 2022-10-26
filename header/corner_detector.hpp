#pragma once

#include"config.hpp"

using namespace std;
using namespace cv;

class corner_detector{
public:
    void adaptiveThreshold(Mat& src, Mat& dst, int thresholdWindow = 5);
    void connectedComponentLabeling(Mat& src, vector<vector<Point>> quadArea, int method = 0);
    void edgeExtraction(Mat& grad_src);
}