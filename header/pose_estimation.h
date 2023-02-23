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

struct PoseInfo {
    int markerID;
    Mat rvec, tvec;
};

class PoseEstimator {
public:
    // PnP solver
    void PnPSolver(MarkerInfo markers, vector<ModelInfo> reconstruct_model, CamInfo camera, PoseInfo& pose);
    void PoseBA(vector<Point2f> imagePoints, vector<Point3f> worldPoints, PoseInfo pose, CamInfo camera);

    // Dense pose estimator
    void DenseSolver(const Mat& img, vector<ModelInfo> reconstruct_model, PoseInfo& pose);

private:
    int ID;
    vector<Point2f> image_points;
    vector<Point3f> model_points;

    Problem problem;
    double rot[3], trans[3];

    void buildProblem(Problem* problem, vector<Point2f> imagePoints, vector<Point3f> worldPoints, CamInfo camera, PoseInfo pose);
};

#endif