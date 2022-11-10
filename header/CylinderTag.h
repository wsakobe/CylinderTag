#pragma once
#ifndef CYLINDERTAG_H
#define CYLINDERTAG_H

#include "config.h"
#include "corner_detector.h"
#include "corner_localize.h"

using namespace std;
using namespace cv;

class CylinderTag{
public:
    // Load state matrix of CylinderTag from file
	CylinderTag(const string& path);

	// Manual input the state matrix (M x N, each element from 0 to 8) of CylinderTag, M represents the size of the dictionary, N represents the coloum of each marker
	CylinderTag(const Mat1i& set_state);

    // Marker Detector
    void detect(const Mat& img, vector<MarkerInfo>& cornerList, int adaptiveThresh = 5, const bool cornerSubPix = false, int cornerSubPixDist = 3);

    // Load reconstructed model
    void loadModel(const string& path, vector<ModelInfo> reconstruct_model);

    // Marker Localization
    void estimatePose(vector<MarkerInfo> cornerList, vector<ModelInfo> reconstruct_model, Mat& rvec, Mat& tvec, bool useDensePoseRefine = false);

    // Draw axis on markers
    void drawAxis(const Mat& img, Mat& rvec, Mat& tvec);

private:
    corner_detector detector;
    
    vector<vector<Point>> quadAreas_labeled;
    vector<vector<Point2f>> corners;
    vector<vector<Point2f>> corners_refined;
    vector<featureInfo> features;
    vector<MarkerInfo> markers;
    
    Mat1i state;

	void load_from_file(const string path); 
	void load_from_set(const Mat1i& set_state);
    void check_dictionary(const Mat1i& state);
    
};

#endif