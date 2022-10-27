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
	// 打开文件
	// ifstream input_file(path);
	// if (!input_file.is_open())
	// {
	// 	throw __FUNCTION__ + string(", ") + "could not open the file\n";
	// }
	// // 读 marker
	// int hexM, hexN, triM, triN;
	// input_file >> hexM >> hexN;
	// triM = hexM + 1;
	// triN = 2 * (hexN + 1);
	// this->state = Mat1i(triM, triN);
	// for (int& i : this->state)
	// {
	// 	input_file >> i;
	// }
	// try
	// {
	// 	check_dictionary(this->state);
	// }
	// catch (const string s)
	// {
	// 	throw s + __FUNCTION__ + string(", ") + "illegal marker info\n";
	// }
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
			if (!(i >= 0 && i <= 8))
				throw __FUNCTION__ + string(", ") + "the number in state matrix must between 0 to 8\n";
		}
		// **检查唯一性

		return;
}

void CylinderTag::detect(const Mat& img, vector<MarkerInfo> cornerList, int adaptiveThresh, const bool cornerSubPix, int cornerSubPixDist){
    Mat img_resize;
    resize(img, img_resize, Size(img.cols / 2, img.rows / 2));
    img_resize.convertTo(img_resize, CV_64FC1, 1.0 / 255);

    Mat img_binary(img_resize.rows, img_resize.cols, CV_8UC1);
    detector.adaptiveThreshold(img_resize, img_binary, adaptiveThresh);
    detector.connectedComponentLabeling(img_binary, quadAreas_labeled);
    detector.edgeExtraction(img_resize, quadAreas_labeled, corners);

    if (cornerSubPix){
       // detector.edgeSubPix(img, corners, corners, cornerSubPixDist);
    }
}

void CylinderTag::loadModel(const string& path, vector<ModelInfo> reconstruct_model){

}

void CylinderTag::estimatePose(vector<MarkerInfo> cornerList, vector<ModelInfo> reconstruct_model, Mat& rvec, Mat& tvec, bool useDensePoseRefine){

}

void CylinderTag::drawAxis(const Mat& img, Mat& rvec, Mat& tvec){

}