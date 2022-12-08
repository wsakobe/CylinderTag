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
	google::InitGoogleLogging(argv[0]);
	
	// int a = 100;
	// clock_t start, finish;  
    // double duration;   
	// while (a--){
	// 	start = clock();
	// 	read_from_image(".\\Data\\n1.bmp");
	// 	finish = clock();   
    // 	duration = (double)(finish - start) / CLOCKS_PER_SEC; /*CLOCKS_PER_SEC，它用来表示一秒钟会有多少个时钟计时单元*/ 
	// 	//cout << duration * 1000 << endl;
	// }
		
	read_from_video(".\\Data\\vid1.avi");
		
	system("pause");
	return 0;
}

void read_from_image(const string& path){
	frame = imread(path);

	CylinderTag marker("CTag_3f15c.marker");
	//marker.loadModel("CTag.model", marker_model);

	cvtColor(frame, img_gray, COLOR_BGR2GRAY);
	marker.detect(img_gray, marker_corners, 5, true, 5);
	//marker.estimatePose(marker_corners, marker_model, rvec, tvec, true);
	//marker.drawAxis(frame, rvec, tvec);
}

void read_from_video(const string& path){
	VideoCapture capture; 
	frame = capture.open(path);	

	CylinderTag marker("CTag_3f15c.marker");
	//marker.loadModel("CTag.model", marker_model);

	while (capture.read(frame))
	{
		for (int i = 0; i < 0; i++)
			capture.read(frame);
		
		cvtColor(frame, img_gray, COLOR_BGR2GRAY);
		marker.detect(img_gray, marker_corners, 5, true, 5);
		//marker.estimatePose(marker_corners, marker_model, rvec, tvec, true);
		//marker.drawAxis(frame, rvec, tvec);
	}
}

void read_online() {

}