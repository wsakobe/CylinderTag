#include "header/CylinderTag.h"

using namespace std;
using namespace cv;

Mat frame, img_gray;
vector<MarkerInfo> markers;
vector<ModelInfo> marker_model;
CamInfo camera;
vector<PoseInfo> pose;

void read_from_image(const string& path);
void read_from_video(const string& path);

int main(int argc, char** argv){
	google::InitGoogleLogging(argv[0]);
	
	//read_from_image("test.bmp");
	read_from_video("test.avi"); 

	waitKey(0);
	destroyAllWindows();
	system("Pause");

	return 0;
}

void read_from_image(const string& path){
	frame = imread(path);

	CylinderTag marker("CTag_2f12c.marker");
	marker.loadModel("CTag_2f12c.model", marker_model);
	marker.loadCamera("cameraParams.yml", camera);

	if (frame.channels() == 3) {
		cvtColor(frame, img_gray, COLOR_BGR2GRAY);
	}

	marker.detect(img_gray, markers, 5, true, 5);
	marker.estimatePose(img_gray, markers, marker_model, camera, pose, false);
	marker.drawAxis(img_gray, markers, marker_model, pose, camera, 30);
}

void read_from_video(const string& path){
	VideoCapture capture; 
	frame = capture.open(path );	

	CylinderTag marker("CTag_2f12c.marker");
	marker.loadModel("CTag_2f12c.model", marker_model);
	marker.loadCamera("cameraParams.yml", camera);

	while (capture.read(frame))
	{	
		cvtColor(frame, img_gray, COLOR_BGR2GRAY);
		markers.clear();
		pose.clear();
		marker.detect(img_gray, markers, 5, true, 5);
		marker.estimatePose(img_gray, markers, marker_model, camera, pose, false);
		marker.drawAxis(img_gray, markers, marker_model, pose, camera, 30);
	}
}
