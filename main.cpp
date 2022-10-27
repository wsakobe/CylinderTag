#include "header/CylinderTag.h"

using namespace std;
using namespace cv;

Mat frame, img_gray;
vector<MarkerInfo> marker_corners;
vector<ModelInfo> marker_model;
Mat rvec, tvec;

void read_from_image(const string& path);
void read_from_video(const string& path);
void read_online();

int main(int argc, char** argv){
	read_from_image(".\\Data\\22.bmp");
	//read_from_video(".\\Data\\result4.avi");
		
	system("pause");
	return 0;
}

void read_from_image(const string& path){
	frame = imread(path);

	CylinderTag marker("CTag_4f20c.marker");
	//marker.loadModel("CTag.model", marker_model);

	cvtColor(frame, img_gray, COLOR_BGR2GRAY);
	marker.detect(img_gray, marker_corners, 5, true, 3);
	//marker.estimatePose(marker_corners, marker_model, rvec, tvec, true);
	//marker.drawAxis(frame, rvec, tvec);
}

void read_from_video(const string& path){
	VideoCapture capture; 
	frame = capture.open(path);	

	CylinderTag marker("CTag_4f20c.marker");
	marker.loadModel("CTag.model", marker_model);

	while (capture.read(frame))
	{
		cvtColor(frame, img_gray, COLOR_BGR2GRAY);
		marker.detect(img_gray, marker_corners, 5, true, 3);
		marker.estimatePose(marker_corners, marker_model, rvec, tvec, true);
		marker.drawAxis(frame, rvec, tvec);
	}
}