#include "header/CylinderTag.h"

using namespace std;
using namespace cv;

Mat frame, img_gray;
vector<MarkerInfo> markers;
vector<ModelInfo> marker_model;
CamInfo camera;
vector<PoseInfo> pose;

void read_from_image(const string& path, int num);
void read_from_video(const string& path);
void read_online();

int cnt = 0;

int main(int argc, char** argv){
	google::InitGoogleLogging(argv[0]);
	
	//Reconstruction
	/*
	for (int i = 1; i <= 20; i++) {
		string filepath = "F:/CylinderTag/Experiment/Pose_experiments/Reconstruction/CylinderTag/l/ (";
		filepath = filepath + to_string(i) + ").bmp";
		read_from_image(filepath, i);
	}*/
	
	//read_from_image("F:/CylinderTag/Experiment/ArUco/aruco-markers-master/pose_estimation/src/test.bmp", 1);
	read_from_video("F:/CylinderTag/Experiment/detection_rate_with_angle/det_42.avi"); 
	//read_from_video("F:/CylinderTag/Experiment/Pose_experiments/Distances/test_videos/distance_70.avi"); 
	//read_online();

	waitKey(0);
	destroyAllWindows();
	system("pause");
	return 0;
}

void read_from_image(const string& path, int num){
	frame = imread(path);

	CylinderTag marker("CTag_2f12c.marker");
	marker.loadModel("CTag_2f12c.model", marker_model);
	marker.loadCamera("cameraParams.yml", camera);

	if (frame.channels() == 3) {
		cvtColor(frame, img_gray, COLOR_BGR2GRAY);
	}

	marker.detect(img_gray, markers, 5, true, 5);
	for (int i = 0; i < markers.size(); i++)
		cout << "ID " << i << ": " << markers[i].markerID + 1 << endl;
	if (markers.size() == 0) {
		cout << "No tag detected" << endl;
		cnt++;
	}
	//marker.estimatePose(img_gray, markers, marker_model, camera, pose, false);
	//marker.drawAxis(img_gray, markers, marker_model, pose, camera, 8);
	
	//Output
	string fname = "F:/CylinderTag/Experiment/Pose_experiments/Reconstruction/CylinderTag/l/";
	fname = fname + to_string(num) + ".txt";
	ofstream Files;
	Files.open(fname, ios::ate);
	for (int i = 0; i < markers.size(); i++) {
		for (int j = 0; j < markers[i].cornerLists.size(); j++) {
			for (int k = 0; k < 2; k++)
				Files << markers[i].featurePos[j] * 8 + k << " " << markers[i].cornerLists[j][k].x << " " << markers[i].cornerLists[j][k].y << endl;
			for (int k = 4; k < 6; k++)
				Files << markers[i].featurePos[j] * 8 + k << " " << markers[i].cornerLists[j][k].x << " " << markers[i].cornerLists[j][k].y << endl;
			cout << markers[i].feature_ID_right[j] << endl;
			if (abs(markers[i].feature_ID_left[j] - markers[i].feature_ID_right[j]) < 3 && markers[i].feature_ID_right[j] != -1) {
				for (int k = 2; k < 4; k++)
					Files << markers[i].featurePos[j] * 8 + k << " " << markers[i].cornerLists[j][k].x << " " << markers[i].cornerLists[j][k].y << endl;
				for (int k = 6; k < 8; k++)
					Files << markers[i].featurePos[j] * 8 + k << " " << markers[i].cornerLists[j][k].x << " " << markers[i].cornerLists[j][k].y << endl;
			}
		}
	}

}

void read_from_video(const string& path){
	VideoCapture capture; 
	frame = capture.open(path );	

	CylinderTag marker("CTag_2f12c.marker");
	marker.loadModel("CTag_2f12c.model", marker_model);
	marker.loadCamera("cameraParams.yml", camera);

	/*
	char fname[256];
	sprintf(fname, "F:/CylinderTag/Experiment/Pose_experiments/Distances/Rot/CylinderTag/70.txt");
	ofstream Files;
	Files.open(fname, ios::out | ios::trunc);
	Files.close();

	sprintf(fname, "F:/CylinderTag/Experiment/Pose_experiments/Distances/Trans/CylinderTag/70.txt");
	Files.open(fname, ios::out | ios::trunc);
	Files.close();
	*/
	int TP = 0, P_ALL = 0, FN = 0;
	for (int i = 0; i < 0; i++)
		capture.read(frame);
	while (capture.read(frame))
	{		
		cvtColor(frame, img_gray, COLOR_BGR2GRAY);
		markers.clear();
		pose.clear();
		marker.detect(img_gray, markers, 5, true, 3);
		for (int i = 0; i < markers.size(); i++) {
			cout << "ID " << i << ": " << markers[i].markerID + 1 << endl;
			if (markers[i].markerID == 21) {
				TP++;
			}
			else {
				FN++;
			}
		}
		cout << TP << "/" << ++P_ALL << endl;
		
		marker.estimatePose(img_gray, markers, marker_model, camera, pose, false);
		if (!pose.empty()) {
			marker.drawAxis(img_gray, markers, marker_model, pose, camera, 8);

			//cout << pose[0].rvec << endl << pose[0].tvec << endl << endl;
			Mat R;
			Rodrigues(pose[0].rvec, R);
			/*
			//output
			sprintf(fname, "F:/CylinderTag/Experiment/Pose_experiments/Distances/Rot/CylinderTag/70.txt");
			Files.open(fname, ios::app);
			Files << format(pose[0].rvec, cv::Formatter::FMT_CSV) << endl;
			Files.close();

			sprintf(fname, "F:/CylinderTag/Experiment/Pose_experiments/Distances/Trans/CylinderTag/70.txt");
			Files.open(fname, ios::app);
			Files << format(pose[0].tvec, cv::Formatter::FMT_CSV) << endl;
			Files.close();*/
		}
	}
	std::cout << "Precision: " << (float)(TP * 1.0 / P_ALL) * 100 << "%" << std::endl;
	std::cout << "Recall: " << (float)(1.0 * TP / (TP + FN)) * 100 << "%" << std::endl;
	waitKey(0);
	destroyAllWindows();
}

void read_online() {
	VideoCapture cap(0);
	//cap.set(CAP_PROP_FRAME_WIDTH, 1080);
	//cap.set(CAP_PROP_FRAME_HEIGHT, 1920);

	CylinderTag marker("CTag_2f12c.marker");
	marker.loadModel("CTag_2f12c.model", marker_model);
	marker.loadCamera("cameraParams.yml", camera);

	while (waitKey(10) != 27) {
		cap >> frame;
		imshow("Input", frame);
		waitKey(1);
		cvtColor(frame, img_gray, COLOR_BGR2GRAY);
		marker.detect(img_gray, markers, 5, true, 5);
	}
	destroyAllWindows();
}