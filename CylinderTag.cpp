#include "header/CylinderTag.h"

using namespace std;
using namespace cv;

CylinderTag::CylinderTag(const string& path)
{
	load_from_file(path);
}

CylinderTag::CylinderTag(const Mat1i& set_state)
{
	load_from_set(set_state);
}

void CylinderTag::load_from_file(const string path)
{
	ifstream input_file(path);
	if (!input_file.is_open())
	{
		throw __FUNCTION__ + string(", ") + "could not open the file\n";
	}
	
	int marker_num, marker_col, feature_size;
	input_file >> marker_num >> marker_col >> feature_size;
	
	this->featureSize = feature_size;
	this->state = Mat1i(marker_num, marker_col);
	for (int& i : this->state)
	{
		input_file >> i;
	}
	try
	{
		check_dictionary(this->state);
	}
	catch (const string s)
	{
		throw s + __FUNCTION__ + string(", ") + "illegal marker info\n";
	}
}

void CylinderTag::load_from_set(const Mat1i& set_state)
{
	try
	{
		check_dictionary(set_state);
	}
	catch (const string s)
	{
		throw s + __FUNCTION__ + string(", ") + "illegal marker info\n";
	}
	this->state = set_state;
}

void CylinderTag::check_dictionary(const Mat1i& input_state)
{
		for (int i : input_state)
		{
			if (!(i > 0 && i <= 63))
				throw __FUNCTION__ + string(", ") + "the number in state matrix must between 1 to 63\n";
		}
		
		return;
}

void CylinderTag::detect(const Mat& img, vector<MarkerInfo>& markers_info, int adaptiveThresh, const bool cornerSubPix, int cornerSubPixDist){
	// Display
    Mat imgMark(img.rows, img.cols, CV_32FC3);
    cvtColor(img, imgMark, COLOR_GRAY2RGB);

	//time record
	clock_t start[10];  
    double duration[10]; 
	start[0] = clock();
    
	//Refresh
	quadAreas_labeled.clear();
	corners.clear();
	features.clear();
	markers.clear();

	Mat img_resize, img_float;
    resize(img, img_resize, Size(img.cols / 2, img.rows / 2), 0.5, 0.5, INTER_CUBIC);
    img_resize.convertTo(img_resize, CV_32FC1, 1.0 / 255);

	start[1] = clock();
	duration[0] = (double)(start[1] - start[0]) / CLOCKS_PER_SEC; 

    Mat img_binary(img_resize.rows, img_resize.cols, CV_8UC1);
    detector.adaptiveThreshold(img_resize, img_binary, adaptiveThresh);
	//imshow("binary", img_binary);
	//waitKey(0);
	start[2] = clock();
	duration[1] = (double)(start[2] - start[1]) / CLOCKS_PER_SEC; 

	detector.connectedComponentLabeling(img_binary, quadAreas_labeled);

	start[3] = clock();
	duration[2] = (double)(start[3] - start[2]) / CLOCKS_PER_SEC; 

    detector.edgeExtraction(img_resize, quadAreas_labeled, corners);
	if (corners.empty()) {
		cout << "No corner detected!" << endl;
		return;
	}

	start[4] = clock();
	duration[3] = (double)(start[4] - start[3]) / CLOCKS_PER_SEC; 

	detector.featureRecovery(corners, features);
	if (features.empty()) {
		cout << "No feature detected!" << endl;
		return;
	}

	detector.cornerObtain(img, features);

	if (cornerSubPix) {
		img.convertTo(img_float, CV_32FC1, 1.0 / 255);
		detector.edgeSubPix(img_float, features, features, cornerSubPixDist);
	}
		
	start[5] = clock();
	duration[4] = (double)(start[5] - start[4]) / CLOCKS_PER_SEC;
	    
	detector.featureExtraction(img, features, features);

	start[6] = clock();
	duration[5] = (double)(start[6] - start[5]) / CLOCKS_PER_SEC; 
	
	for (int i = 0; i < features.size(); i++){
        line(imgMark, features[i].corners[0], features[i].corners[1], Scalar(0, 255, 255), 2.5);
        line(imgMark, features[i].corners[1], features[i].corners[2], Scalar(0, 255, 255), 2.5);
        line(imgMark, features[i].corners[2], features[i].corners[7], Scalar(0, 255, 255), 2.5);
        line(imgMark, features[i].corners[7], features[i].corners[4], Scalar(0, 255, 255), 2.5);
        line(imgMark, features[i].corners[4], features[i].corners[5], Scalar(0, 255, 255), 2.5);
        line(imgMark, features[i].corners[5], features[i].corners[6], Scalar(0, 255, 255), 2.5);
        line(imgMark, features[i].corners[6], features[i].corners[3], Scalar(0, 255, 255), 2.5);
        line(imgMark, features[i].corners[3], features[i].corners[0], Scalar(0, 255, 255), 2.5);
		for (int k = 0; k < 8; k++)
			circle(imgMark, features[i].corners[k], 3, Scalar(75, 92, 196), -1);
		ostringstream oss;
		oss << std::setprecision(4) << features[i].ID_left;
		putText(imgMark, oss.str(), features[i].corners[0], FONT_ITALIC, 0.6, Scalar(250, 250, 250), 2);
		oss.str("");
		oss << std::setprecision(4) << features[i].ID_right;
		putText(imgMark, oss.str(), features[i].corners[4], FONT_ITALIC, 0.6, Scalar(250, 250, 90), 2);
    }
	imshow("Feature Organization", imgMark);
    waitKey(0);
    
	detector.markerOrganization(features, markers);
	detector.markerDecoder(markers, markers, this->state, this->featureSize);
	markers_info = markers;

	//Plot
	//imgMark = img.clone();
	//cvtColor(img, imgMark, COLOR_GRAY2RGB);
	//for (int i = 0; i < markers.size(); i++) {
	//	for (int j = 0; j < markers[i].cornerLists.size(); j++) {
	//		line(imgMark, markers[i].cornerLists[j][0], markers[i].cornerLists[j][1], Scalar(0, 255, 255), 2.5);
	//		line(imgMark, markers[i].cornerLists[j][1], markers[i].cornerLists[j][2], Scalar(0, 255, 255), 2.5);
	//		line(imgMark, markers[i].cornerLists[j][2], markers[i].cornerLists[j][7], Scalar(0, 255, 255), 2.5);
	//		line(imgMark, markers[i].cornerLists[j][7], markers[i].cornerLists[j][4], Scalar(0, 255, 255), 2.5);
	//		line(imgMark, markers[i].cornerLists[j][4], markers[i].cornerLists[j][5], Scalar(0, 255, 255), 2.5);
	//		line(imgMark, markers[i].cornerLists[j][5], markers[i].cornerLists[j][6], Scalar(0, 255, 255), 2.5);
	//		line(imgMark, markers[i].cornerLists[j][6], markers[i].cornerLists[j][3], Scalar(0, 255, 255), 2.5);
	//		line(imgMark, markers[i].cornerLists[j][3], markers[i].cornerLists[j][0], Scalar(0, 255, 255), 2.5);
	//		for (int k = 0; k < 8; k++)
	//			circle(imgMark, markers[i].cornerLists[j][k], 1.5, Scalar(107, 90, 219));
	//		ostringstream oss;
	//		oss << markers[i].featurePos[j];
	//		putText(imgMark, oss.str(), markers[i].cornerLists[j][0], FONT_ITALIC, 0.6, Scalar(20, 200, 255), 2);
	//	}
	//}
	//imshow("Output", imgMark);
	//waitKey(0);
	
	start[7] = clock();
	duration[6] = (double)(start[7] - start[6]) / CLOCKS_PER_SEC; 
	cout << duration[0] * 1000 << " " << duration[1] * 1000 << " " << duration[2] * 1000 << " " << duration[3] * 1000 << " " << duration[4] * 1000 << " " << duration[5] * 1000 << " " << duration[6] * 1000 << " " << endl;
	double ttime = duration[0] + duration[1] + duration[2] + duration[3] + duration[4] + duration[5] + duration[6];
	cout << "Total time: " << ttime * 1000 << endl;
}

void CylinderTag::loadModel(const string& path, vector<ModelInfo>& reconstruct_model){
	ifstream input_file(path);
	if (!input_file.is_open())
	{
		throw __FUNCTION__ + string(", ") + "could not open the model file\n";
	}

	int model_num, model_size, corner_id;
	input_file >> model_num >> model_size;
	reconstruct_model.resize(model_num);
	Point3f world_point;
	for (int i = 0; i < model_num; i++) {
		reconstruct_model[i].MarkerID = i;
		input_file >> reconstruct_model[i].base.x;
		input_file >> reconstruct_model[i].base.y;
		input_file >> reconstruct_model[i].base.z;
		input_file >> reconstruct_model[i].axis.x;
		input_file >> reconstruct_model[i].axis.y;
		input_file >> reconstruct_model[i].axis.z;
		reconstruct_model[i].corners.resize(model_size * 8);
		for (int j = 0; j < 8 * model_size; j++) {
			input_file >> corner_id;
			input_file >> reconstruct_model[i].corners[corner_id].x;
			input_file >> reconstruct_model[i].corners[corner_id].y;
			input_file >> reconstruct_model[i].corners[corner_id].z;
		}		
	}
	
}

void CylinderTag::loadCamera(const string& path, CamInfo& camera){
	FileStorage fs(path, FileStorage::READ);
	fs["cameraMatrix"] >> camera.Intrinsic;
	fs["distCoeffs"] >> camera.distCoeffs;
}

void CylinderTag::estimatePose(const Mat& img, vector<MarkerInfo> markers, vector<ModelInfo> reconstruct_model, CamInfo camera, vector<Mat>& rvec, vector<Mat>& tvec, bool useDensePoseRefine){
	rvec.resize(markers.size());
	tvec.resize(markers.size());
	for (int i = 0; i < markers.size(); i++) {
		estimator.PnPSolver(markers[i], reconstruct_model, camera, rvec[i], tvec[i]);
		if (useDensePoseRefine)
			estimator.DenseSolver(img, reconstruct_model, rvec[i], tvec[i]);
	}	
}

void CylinderTag::drawAxis(const Mat& img, vector<MarkerInfo> markers, vector<ModelInfo> reconstruct_model, vector<Mat>& rvec, vector<Mat>& tvec, CamInfo camera, int length = 5){
	// Display
	Mat imgMark(img.rows, img.cols, CV_32FC3);
	cvtColor(img, imgMark, COLOR_GRAY2RGB);

	vector<Point2f> imagePoints, image_points;
	vector<Point3f> model_points;
	for (int i = 0; i < markers.size(); i++) {
		model_points.clear();
		image_points.clear();
		int ID = markers[i].markerID;
		for (int j = 0; j < markers[i].cornerLists.size(); j++) {
			for (int k = 0; k < 8; k++) {
				model_points.push_back(reconstruct_model[ID].corners[markers[i].featurePos[j] * 8 + k]);
				image_points.push_back(markers[i].cornerLists[j][k]);
			}
		}
		model_points.push_back(reconstruct_model[ID].base);
		model_points.push_back(reconstruct_model[ID].base + reconstruct_model[ID].axis * length);
		model_points.push_back(reconstruct_model[ID].base + Point3f(0.0372, 0.0372, 0.9986) * length);
		model_points.push_back(reconstruct_model[ID].base + Point3f(0.9980, -0.0520, -0.0353) * length);

		imagePoints.clear();
		projectPoints(model_points, rvec[i], tvec[i], camera.Intrinsic, camera.distCoeffs, imagePoints);
		for (int i = 0; i < imagePoints.size() - 4; i++) {
			circle(imgMark, imagePoints[i], 5, Scalar(255, 234, 32), -1);
		}

		arrowedLine(imgMark, imagePoints[imagePoints.size() - 4], imagePoints[imagePoints.size() - 3], Scalar(255, 0, 0), 3);
		arrowedLine(imgMark, imagePoints[imagePoints.size() - 4], imagePoints[imagePoints.size() - 2], Scalar(0, 255, 0), 3);
		arrowedLine(imgMark, imagePoints[imagePoints.size() - 4], imagePoints[imagePoints.size() - 1], Scalar(0, 0, 255), 3);
		circle(imgMark, imagePoints[imagePoints.size() - 4], 8, Scalar(247, 235, 235), -1);

		float reprojection_error = 0;
		for (int i = 0; i < imagePoints.size() - 4; i++) {
			reprojection_error += sqrt((imagePoints[i].x - image_points[i].x) * (imagePoints[i].x - image_points[i].x) + (imagePoints[i].y - image_points[i].y) * (imagePoints[i].y - image_points[i].y));
		}
		cout << "PnP RPE: " << reprojection_error / (imagePoints.size() - 4) << endl;
	}
	imshow("Plot", imgMark);
	waitKey(1);
}