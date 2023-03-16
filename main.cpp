#include "header/CylinderTag.h"
#include "mvcameracontrol.h"

using namespace std;
using namespace cv;

Mat frame, img_gray;
vector<MarkerInfo> markers;
vector<ModelInfo> marker_model;
CamInfo camera;
vector<PoseInfo> pose;

//HikVision Camera Preparation
int nRet[5] = { MV_OK };
void* handle[5] = { NULL };
unsigned char* pData[5];
unsigned int g_nPayloadSize = 0;
MV_FRAME_OUT_INFO_EX* imageInfo;
MV_CC_DEVICE_INFO_LIST stDeviceList;
MV_FRAME_OUT stImageInfo[5] = { { 0 } };

void read_from_image(const string& path, int num);
void read_from_video(const string& path);
void read_online();
bool initCamera();
int RGB2BGR(unsigned char* pRgbData, unsigned int nWidth, unsigned int nHeight);
cv::Mat Convert2Mat(MV_FRAME_OUT& pstImage);

int cnt = 0;

int main(int argc, char** argv){
	google::InitGoogleLogging(argv[0]);
	
	//Reconstruction
	/*
	for (int i = 1; i <= 18; i++) {
		string filepath = "F:/CylinderTag/CylinderTag/Recon/coil/r/ (";
		filepath = filepath + to_string(i) + ").bmp";
		read_from_image(filepath, i);
	}*/
	
	//read_from_image("F:/CylinderTag/CylinderTag/test.bmp", 1);
	//read_from_video("F:/CylinderTag/Experiment/detection_rate_with_angle/det_90.avi"); 
	read_online();

	system("Pause");
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
	string fname = "F:/CylinderTag/CylinderTag/Recon/coil/r/";
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
			if (abs(markers[i].feature_ID_left[j] - markers[i].feature_ID_right[j]) < 4 && markers[i].feature_ID_right[j] != -1) {
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
	sprintf(fname, "F:/CylinderTag/Experiment/detection_rate_with_angle/pose_inform/yaw/Rot/-90.txt");
	ofstream Files;
	Files.open(fname, ios::out | ios::trunc);
	Files.close();

	sprintf(fname, "F:/CylinderTag/Experiment/detection_rate_with_angle/pose_inform/yaw/Trans/-90.txt");
	Files.open(fname, ios::out | ios::trunc);
	Files.close();
	*/
	int TP = 0, P_ALL = 0, FN = 0;
	for (int i = 0; i < 0; i++)
		capture.read(frame);
	while (capture.read(frame))
	{		
		TickMeter meter;
		meter.start();
		cvtColor(frame, img_gray, COLOR_BGR2GRAY);
		markers.clear();
		pose.clear();
		marker.detect(img_gray, markers, 5, true, 5);
		meter.stop();
		cout << std::fixed << std::setprecision(3) << meter.getTimeSec() * 1000 << endl;
		for (int i = 0; i < markers.size(); i++) {
			//cout << "ID " << i << ": " << markers[i].markerID + 1 << endl;
			if (markers[i].markerID == 21) {
				TP++;
			}
			else {
				FN++;
			}
		}
		
		//cout << TP << "/" << ++P_ALL << endl;
		/*
		marker.estimatePose(img_gray, markers, marker_model, camera, pose, false);
		if (!pose.empty()) {
			marker.drawAxis(img_gray, markers, marker_model, pose, camera, 8);
			//cout << pose[0].rvec << endl << pose[0].tvec << endl << endl;
			
			Mat R;
			Rodrigues(pose[0].rvec, R);

			//output 
			sprintf(fname, "F:/CylinderTag/Experiment/detection_rate_with_angle/pose_inform/yaw/Rot/-90.txt");
			Files.open(fname, ios::app);
			Files << format(R, cv::Formatter::FMT_CSV) << endl;
			Files.close();

			sprintf(fname, "F:/CylinderTag/Experiment/detection_rate_with_angle/pose_inform/yaw/Trans/-90.txt");
			Files.open(fname, ios::app);
			Files << format(pose[0].tvec, cv::Formatter::FMT_CSV) << endl;
			Files.close();
		}*/
	}
	std::cout << "Precision: " << (float)(TP * 1.0 / P_ALL) * 100 << "%" << std::endl;
	std::cout << "Recall: " << (float)(1.0 * TP / (TP + FN)) * 100 << "%" << std::endl;
	waitKey(0);
	destroyAllWindows();
}

void read_online() {
	if (!initCamera()) { return; }

	CylinderTag marker("CTag_2f12c.marker");
	marker.loadModel("CTag_2f12c.model", marker_model);
	marker.loadCamera("cameraParams.yml", camera);
	
	int t = 100;
	char fname[256];
	ofstream Files;
	
	while (t--) {
		nRet[0] = MV_CC_GetImageBuffer(handle[0], &stImageInfo[0], 50);
		if (nRet[0] == MV_OK)
		{
			frame = Convert2Mat(stImageInfo[0]);
			imwrite("./photo/1.bmp", frame);
			if (frame.channels() == 3) {
				cvtColor(frame, frame, COLOR_BGR2GRAY);
			}
			marker.detect(frame, markers, 5, true, 5);
			marker.estimatePose(frame, markers, marker_model, camera, pose, false);
			marker.drawAxis(frame, markers, marker_model, pose, camera, 8);
			
			if (!pose.empty()) {
				Mat rvec, tvec, R;
				rvec = pose[0].rvec;
				tvec = pose[0].tvec;
				rvec.convertTo(rvec, CV_32FC1);
				tvec.convertTo(tvec, CV_32FC1);

				//output 
				sprintf(fname, "./photo/rot.txt");
				Files.open(fname, ios::app);
				Files << format(rvec, cv::Formatter::FMT_CSV) << endl;
				Files.close();

				sprintf(fname, "./photo/trans.txt");
				Files.open(fname, ios::app);
				Files << format(tvec, cv::Formatter::FMT_CSV) << endl;
				Files.close();
								
				Rodrigues(rvec, R);
				Mat end_effector(3, 1, CV_32FC1);
				end_effector.at<float>(0, 0) = 0;
				end_effector.at<float>(1, 0) = 80;
				end_effector.at<float>(2, 0) = 297;
				sprintf(fname, "./photo/end.txt");
				Files.open(fname, ios::app);
				Files << format(R * end_effector + tvec, cv::Formatter::FMT_CSV) << endl;
				Files.close();
			}			
		}
		nRet[0] = MV_CC_FreeImageBuffer(handle[0], &stImageInfo[0]);
		if (nRet[0] != MV_OK)
		{
			printf("Free Image Buffer fail! nRet [0x%x]\n", nRet[0]);
		}
		waitKey(20);
	}

	for (int i = 0; i < stDeviceList.nDeviceNum; i++) {
		nRet[i] = MV_CC_CloseDevice(handle[i]);
		if (MV_OK != nRet[i])
		{
			printf("Close Device fail! nRet [0x%x]\n", nRet[i]);
			break;
		}
	}
	// Destroy handle
	for (int i = 0; i < stDeviceList.nDeviceNum; i++) {
		nRet[i] = MV_CC_DestroyHandle(handle[i]);
		if (MV_OK != nRet[i])
		{
			printf("Destroy Handle fail! nRet [0x%x]\n", nRet[i]);
			break;
		}
	}
	printf("Device successfully closed.\n");

	destroyAllWindows();
}

int RGB2BGR(unsigned char* pRgbData, unsigned int nWidth, unsigned int nHeight)
{
	if (NULL == pRgbData)
	{
		return MV_E_PARAMETER;
	}

	for (unsigned int j = 0; j < nHeight; j++)
	{
		for (unsigned int i = 0; i < nWidth; i++)
		{
			unsigned char red = pRgbData[j * (nWidth * 3) + i * 3];
			pRgbData[j * (nWidth * 3) + i * 3] = pRgbData[j * (nWidth * 3) + i * 3 + 2];
			pRgbData[j * (nWidth * 3) + i * 3 + 2] = red;
		}
	}

	return MV_OK;
}

cv::Mat Convert2Mat(MV_FRAME_OUT& pstImage)   // convert data stream in Mat format
{
	cv::Mat srcImage;
	if (pstImage.stFrameInfo.enPixelType == PixelType_Gvsp_Mono8)
	{
		srcImage = cv::Mat(pstImage.stFrameInfo.nHeight, pstImage.stFrameInfo.nWidth, CV_8UC1, pstImage.pBufAddr);
	}
	else if (pstImage.stFrameInfo.enPixelType == PixelType_Gvsp_RGB8_Packed)
	{
		srcImage = cv::Mat(pstImage.stFrameInfo.nHeight, pstImage.stFrameInfo.nWidth, CV_8UC3, pstImage.pBufAddr);
	}

	return srcImage;
}

bool initCamera() {
	memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));

	nRet[0] = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
	if (MV_OK != nRet[0])
	{
		printf("Enum Devices fail! nRet [0x%x]\n", nRet[0]);
		return false;
	}

	if (stDeviceList.nDeviceNum == 0) {
		printf("There is no device available.\n");
		return false;
	}

	if (stDeviceList.nDeviceNum == 1) {
		printf("There is one HikCamera offline. Please check.\n");
		return false;
	}

	for (int i = 0; i < stDeviceList.nDeviceNum; i++) {
		nRet[i] = MV_CC_CreateHandle(&handle[i], stDeviceList.pDeviceInfo[i]);
		if (MV_OK != nRet[i])
		{
			printf("Create Handle fail! nRet [0x%x]\n", nRet[i]);
			return false;
		}
	}

	for (int i = 0; i < stDeviceList.nDeviceNum; i++) {
		nRet[i] = MV_CC_OpenDevice(handle[i]);
		if (MV_OK != nRet[i])
		{
			printf("Open Device fail! nRet [0x%x]\n", nRet[i]);
			return false;
		}
	}

	for (int i = 0; i < stDeviceList.nDeviceNum; i++) {
		nRet[i] = MV_CC_SetEnumValue(handle[i], "TriggerMode", 0);
		if (MV_OK != nRet[i])
		{
			printf("Set Enum Value fail! nRet [0x%x]\n", nRet[i]);
			return false;
		}
	}

	// Get payload size
	MVCC_INTVALUE stParam;
	memset(&stParam, 0, sizeof(MVCC_INTVALUE));
	nRet[0] = MV_CC_GetIntValue(handle[0], "PayloadSize", &stParam);
	g_nPayloadSize = stParam.nCurValue;

	for (int i = 0; i < stDeviceList.nDeviceNum; i++) {
		nRet[i] = MV_CC_StartGrabbing(handle[i]);
		if (MV_OK != nRet[i])
		{
			printf("Start Grabbing fail! nRet [0x%x]\n", nRet[i]);
			return false;
		}
	}
	for (int i = 0; i < stDeviceList.nDeviceNum; i++) {
		memset(&stImageInfo[i], 0, sizeof(MV_FRAME_OUT_INFO_EX));
		pData[i] = (unsigned char*)malloc(sizeof(unsigned char) * (g_nPayloadSize));
	}

	cout << "Stereo camera loaded successfully!" << endl;

	return true;
}