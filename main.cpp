#include <opencv2/opencv.hpp>
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

int main() {
	frame = imread(".\\Data\\test3.jpg");
	string filename = ".\\Data\\result4.avi";
	VideoCapture capture; 

	frame = capture.open(filename);
	while (capture.read(frame))
	{
		cvtColor(frame, img_gray, COLOR_BGR2GRAY);
		imshow("FAST", img_gray);
		waitKey(0);
	}
	
	system("pause");
	return 0;
}