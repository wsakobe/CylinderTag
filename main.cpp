#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

Mat img, img_gray;

void main() {
	VideoCapture capture;
	img = capture.open("test3.jpg");
	cvtColor(img, img_gray, COLOR_BGR2GRAY);

}