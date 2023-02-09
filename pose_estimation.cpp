#include "header/pose_estimation.h"

using namespace std;
using namespace cv;


void PoseEstimator::PnPSolver(MarkerInfo markers, vector<ModelInfo> reconstruct_model, CamInfo camera, Mat& rvec, Mat& tvec)
{
	image_points.clear();
	model_points.clear();

	ID = markers.markerID;
		
	for (int j = 0; j < markers.cornerLists.size(); j++) {
		for (int k = 0; k < 8; k++) {
			image_points.push_back(markers.cornerLists[j][k]);
			model_points.push_back(reconstruct_model[ID].corners[markers.featurePos[j] * 8 + k]);
		}
	}
	solvePnPRansac(model_points, image_points, camera.Intrinsic, camera.distCoeffs, rvec, tvec, false, 100, 2, 0.99, noArray(), SOLVEPNP_EPNP);
	//solvePnPRefineLM(model_points, image_points, camera.Intrinsic, camera.distCoeffs, rvec, tvec, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 20, 0.5));
}

void PoseEstimator::DenseSolver(const Mat& img, vector<ModelInfo> reconstruct_model, Mat& rvec, Mat& tvec)
{
	// To be updated
}
