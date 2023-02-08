#include "header/CylinderTag.h"

using namespace std;
using namespace cv;

Mat frame, img_gray;
vector<MarkerInfo> markers;
vector<ModelInfo> marker_model;
CamInfo camera;
vector<Mat> rvec, tvec;

void read_from_image(const string& path, int num);
void read_from_video(const string& path);
void read_online();

int main(int argc, char** argv){
	google::InitGoogleLogging(argv[0]);
	
	for (int i = 1; i <= 21; i++) {
		string filepath = ".\\Data\\r\\";
		filepath = filepath + to_string(i) + ".bmp";
		read_from_image(filepath, i);
	}
	//read_from_image(".\\Data\\l3.bmp", 1);
	//read_from_video(".\\Data\\v2.avi");
		
	system("pause");
	return 0;
}

void read_from_image(const string& path, int num){
	frame = imread(path);

	CylinderTag marker("CTag_3f12c.marker");
	marker.loadModel("CTag_3f12c.model", marker_model);
	marker.loadCamera("cameraParams.yml", camera);

	cvtColor(frame, img_gray, COLOR_BGR2GRAY);
	marker.detect(img_gray, markers, 5, true, 5);
	//marker.estimatePose(img_gray, markers, marker_model, camera, rvec, tvec, false);
	//marker.drawAxis(img_gray, markers, marker_model, rvec, tvec, camera, 8);

	//Output
	string fname = ".\\Recon\\r";
	fname = fname + to_string(num) + ".txt";
	ofstream Files;
	Files.open(fname, ios::ate);
	for (int i = 0; i < markers.size(); i++) {
		for (int j = 0; j < markers[i].cornerLists.size(); j++) {
			for (int k = 0; k < 8; k++)
				Files << markers[i].featurePos[j] * 8 + k << " " << markers[i].cornerLists[j][k].x << " " << markers[i].cornerLists[j][k].y << endl;
		}
	}
}

void read_from_video(const string& path){
	VideoCapture capture; 
	frame = capture.open(path);	

	CylinderTag marker("CTag_3f12c.marker");
	marker.loadModel("CTag_3f12c.model", marker_model);
	marker.loadCamera("cameraParams.yml", camera);

	int cnt = 0;
	while (capture.read(frame))
	{
		for (int i = 0; i < 0; i++)
			capture.read(frame);
		cout << cnt++ << endl;
		cvtColor(frame, img_gray, COLOR_BGR2GRAY);
		marker.detect(img_gray, markers, 5, true, 5);
		//marker.estimatePose(img_gray, markers, marker_model, camera, rvec, tvec, false);
		//marker.drawAxis(img_gray, markers, marker_model, rvec, tvec, camera, 5);
	}
}

void read_online() {

}