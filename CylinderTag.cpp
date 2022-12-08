#include "header/CylinderTag.h"
#include <fstream>

using namespace std;
using namespace cv;

CylinderTag::CylinderTag(const string& path)
{
	load_from_file(path);
}

CylinderTag::CylinderTag(const Mat1i& set_state)
{
	load_from_set(set_state);
}

void CylinderTag::load_from_file(const string path)
{
	ifstream input_file(path);
	if (!input_file.is_open())
	{
		throw __FUNCTION__ + string(", ") + "could not open the file\n";
	}
	
	int marker_num, marker_col, feature_size;
	input_file >> marker_num >> marker_col >> feature_size;
	
	this->featureSize = feature_size;
	this->state = Mat1i(marker_num, marker_col);
	for (int& i : this->state)
	{
		input_file >> i;
	}
	try
	{
		check_dictionary(this->state);
	}
	catch (const string s)
	{
		throw s + __FUNCTION__ + string(", ") + "illegal marker info\n";
	}
}

void CylinderTag::load_from_set(const Mat1i& set_state)
{
	try
	{
		check_dictionary(set_state);
	}
	catch (const string s)
	{
		throw s + __FUNCTION__ + string(", ") + "illegal marker info\n";
	}
	this->state = set_state;
}

void CylinderTag::check_dictionary(const Mat1i& input_state)
{
		for (int i : input_state)
		{
			if (!(i >= 0 && i <= 9))
				throw __FUNCTION__ + string(", ") + "the number in state matrix must between 0 to 8\n";
		}
		
		return;
}

void CylinderTag::detect(const Mat& img, vector<MarkerInfo>& cornerList, int adaptiveThresh, const bool cornerSubPix, int cornerSubPixDist){
	// Display
    Mat imgMark(img.rows, img.cols, CV_32FC3);
    cvtColor(img, imgMark, COLOR_GRAY2RGB);

	//time record
	clock_t start[10];  
    double duration[10]; 
	start[0] = clock();
    
	//Refresh
	quadAreas_labeled.clear();
	meanG.clear();
	corners.clear();
	features.clear();
	markers.clear();

	Mat img_resize, img_float;
    resize(img, img_resize, Size(img.cols / 2, img.rows / 2), 0.5, 0.5, INTER_CUBIC);
    img_resize.convertTo(img_resize, CV_32FC1, 1.0 / 255);

	start[1] = clock();
	duration[0] = (double)(start[1] - start[0]) / CLOCKS_PER_SEC; 

    Mat img_binary(img_resize.rows, img_resize.cols, CV_8UC1);
    detector.adaptiveThreshold(img_resize, img_binary, adaptiveThresh);
	//imshow("binary", img_binary);
	//waitKey(1);
	start[2] = clock();
	duration[1] = (double)(start[2] - start[1]) / CLOCKS_PER_SEC; 

	detector.connectedComponentLabeling(img_binary, quadAreas_labeled);

	start[3] = clock();
	duration[2] = (double)(start[3] - start[2]) / CLOCKS_PER_SEC; 

    detector.edgeExtraction(img_resize, quadAreas_labeled, corners, meanG);
	if (corners.empty()) {
		cout << "No corner detected!" << endl;
		return;
	}

	start[4] = clock();
	duration[3] = (double)(start[4] - start[3]) / CLOCKS_PER_SEC; 

	detector.featureRecovery(corners, features, meanG);
	if (features.empty()) {
		cout << "No feature detected!" << endl;
		return;
	}

	if (cornerSubPix) {
		img.convertTo(img_float, CV_32FC1, 1.0 / 255);
		detector.edgeSubPix(img_float, features, features, cornerSubPixDist);
	}
		
	start[5] = clock();
	duration[4] = (double)(start[5] - start[4]) / CLOCKS_PER_SEC;

    
	detector.featureExtraction(img, features, features);

	start[6] = clock();
	duration[5] = (double)(start[6] - start[5]) / CLOCKS_PER_SEC; 

	for (int i = 0; i < features.size(); i++){
        line(imgMark, features[i].corners[0], features[i].corners[1], Scalar(0, 255, 255), 2.5);
        line(imgMark, features[i].corners[1], features[i].corners[2], Scalar(0, 255, 255), 2.5);
        line(imgMark, features[i].corners[2], features[i].corners[7], Scalar(0, 255, 255), 2.5);
        line(imgMark, features[i].corners[7], features[i].corners[4], Scalar(0, 255, 255), 2.5);
        line(imgMark, features[i].corners[4], features[i].corners[5], Scalar(0, 255, 255), 2.5);
        line(imgMark, features[i].corners[5], features[i].corners[6], Scalar(0, 255, 255), 2.5);
        line(imgMark, features[i].corners[6], features[i].corners[3], Scalar(0, 255, 255), 2.5);
        line(imgMark, features[i].corners[3], features[i].corners[0], Scalar(0, 255, 255), 2.5);
        circle(imgMark, features[i].feature_center, 1.5, Scalar(107, 90, 219));
		ostringstream oss;
		oss << std::setprecision(3) << features[i].cross_ratio;
		putText(imgMark, oss.str(), features[i].corners[0], FONT_ITALIC, 0.6, Scalar(20, 200, 255), 2);
    }
	for (int i = 0; i < corners.size(); i++) {
		for (int j = 0; j < corners[i].size(); j++)
			circle(imgMark, corners[i][j], 5, Scalar(75, 92, 196), -1);
	}
    
	imshow("Feature Organization", imgMark);
    waitKey(1);
    
	detector.markerOrganization(features, markers);
	detector.markerDecoder(markers, markers, this->state, this->featureSize);

	start[7] = clock();
	duration[6] = (double)(start[7] - start[6]) / CLOCKS_PER_SEC; 
	cout << duration[0] * 1000 << " " << duration[1] * 1000 << " " << duration[2] * 1000 << " " << duration[3] * 1000 << " " << duration[4] * 1000 << " " << duration[5] * 1000 << " " << duration[6] * 1000 << " " << endl;
	double ttime = duration[0] + duration[1] + duration[2] + duration[3] + duration[4] + duration[5] + duration[6];
	cout << "Total time: " << ttime * 1000 << endl;
}

void CylinderTag::loadModel(const string& path, vector<ModelInfo> reconstruct_model){

}

void CylinderTag::estimatePose(vector<MarkerInfo> cornerList, vector<ModelInfo> reconstruct_model, Mat& rvec, Mat& tvec, bool useDensePoseRefine){

}

void CylinderTag::drawAxis(const Mat& img, Mat& rvec, Mat& tvec){

}