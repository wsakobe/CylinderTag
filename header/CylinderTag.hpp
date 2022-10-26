#pragma once

#include"config.hpp"
#include"corner_detector.hpp"

using namespace std;
using namespace cv;

class CylinderTag{
public:
    // Load state matrix of CylinderTag from file
	CylinderTag(const string& path);

	// Manual input the state matrix (M x N, each element from 0 to 8) of CylinderTag, M represents the size of the dictionary, N represents the coloum of each marker
	CylinderTag(const Mat1i& set_state);

    // Marker Detector
    void detect(const Mat& img, vector<Point2f> cornerList, int adaptiveThresh = 5, const bool cornerSubPix = false, int cornerSubPixDist = 3);

    // Load reconstructed model
    void loadModel(const string& path, vector<Point3f> reconstruct_model);

    // Marker Localization
    void estimatePose(vector<Point2f> cornerList, vector<Point3f> reconstruct_model, Mat& rvec, Mat& tvec);

    // Draw axis on markers
    void drawAxis(const Mat& img, Mat& rvec, Mat& tvec);
private:
    corner_detector detector;
    
    // 从文件中读取 info
	void load_from_file(const string path); 

	// 手动填写 info
	void load_from_set(const Mat1i& set_state);
    
}