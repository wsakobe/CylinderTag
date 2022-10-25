#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cstdlib>
#include <vector>
#include <iostream> 

using namespace std;
using namespace cv;

Mat frame, img_gray;

void main() {
	frame = imread(".\\Data\\test3.jpg");
	string filename = ".\\Data\\result4.avi";
	VideoCapture capture; 

	frame = capture.open(filename);
	while (capture.read(frame))
	{
		cvtColor(frame, img_gray, COLOR_BGR2GRAY);
		adaptiveThreshold(img_gray, img_gray, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, -10);
		vector<KeyPoint> corner_candidates;
		vector<Point2f> pt;
		FAST(img_gray, corner_candidates, 250, true, FastFeatureDetector::TYPE_9_16);
		KeyPoint::convert(corner_candidates, pt);
		cout << pt.size() << endl;
		//for (int i = 0; i < pt.size(); i++)
		//	circle(img_gray, pt[i], 3, Scalar(255, 0, 0), -1);
		imshow("FAST", img_gray);
		waitKey(0);
	}
	
	destroyAllWindows();
}