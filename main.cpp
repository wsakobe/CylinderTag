#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

Mat img, img_gray;

void main() {
	img = imread(".\\Data\\test3.jpg");
	cvtColor(img, img_gray, COLOR_BGR2GRAY);
	resize(img_gray, img_gray, Size(2000, 1500));

	//int start_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	vector<Point> corner_candidates;
	goodFeaturesToTrack(img_gray, corner_candidates, 150, 0.2, 10, noArray(), 11, true);
	for (int i = 0; i < corner_candidates.size(); i++)
		circle(img_gray, corner_candidates[i], 3, Scalar(255, 0, 0), -1);
	imshow("Harris", img_gray);
	waitKey(0);

	Mat img_blur;
	Mat Gx, Gy, Gxx, Gyy, Gxy, G_score;
	int kernal_size = 5, sigma = 3;

	GaussianBlur(img_gray, img_blur, Size(kernal_size, kernal_size), sigma);
	Scharr(img_blur, Gx, CV_32FC1, 1, 0);
	Scharr(img_blur, Gy, CV_32FC1, 0, 1);

	Scharr(Gx, Gxx, CV_32FC1, 1, 0);
	Scharr(Gy, Gyy, CV_32FC1, 0, 1);
	Scharr(Gx, Gxy, CV_32FC1, 0, 1);

	G_score = Gxy.mul(Gxy) - Gxx.mul(Gyy);

	destroyAllWindows();
}