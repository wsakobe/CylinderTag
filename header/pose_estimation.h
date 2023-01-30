#pragma once
#ifndef POSE_ESTIMATION_H
#define POSE_ESTIMATION_H

#include "config.h"
#include "corner_detector.h"

using namespace std;
using namespace cv;

struct CamInfo {
    Mat Intrinsic, distCoeffs;
};

struct ModelInfo {
    int MarkerID;
    Point3f axis, base;
    vector<Point3f> corners;
};

class PoseEstimator {
public:
    // PnP solver
    void PnPSolver(MarkerInfo markers, vector<ModelInfo> reconstruct_model, CamInfo camera, Mat& rvec, Mat& tvec);

    // Dense pose estimator
    void DenseSolver(const Mat& img, vector<ModelInfo> reconstruct_model, Mat& rvec, Mat& tvec);

private:
    int ID;
    vector<Point2f> image_points;
    vector<Point3f> model_points;

};

#endif