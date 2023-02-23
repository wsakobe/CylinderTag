#include "header/pose_estimation.h"

using namespace std;
using namespace cv;

struct SnavelyReprojectionError
{
	Point2d observed;
	CamInfo cam;
	Point3d point_ID;

	SnavelyReprojectionError(Point2d observation, CamInfo cam, Point3d point_ID) :observed(observation), cam(cam), point_ID(point_ID) {}

	template <typename T>
	bool operator()(const T* const rotation,
		const T* const translation,
		T* residuals)const {
		T predictions[2], pos_proj[3], pos_world[3];

		pos_world[0] = T(point_ID.x);
		pos_world[1] = T(point_ID.y);
		pos_world[2] = T(point_ID.z);
		AngleAxisRotatePoint(rotation, pos_world, pos_proj);

		pos_proj[0] += translation[0];
		pos_proj[1] += translation[1];
		pos_proj[2] += translation[2];

		const T fx = T(cam.Intrinsic.at<float>(0, 0));
		const T fy = T(cam.Intrinsic.at<float>(1, 1));
		const T cx = T(cam.Intrinsic.at<float>(0, 2));
		const T cy = T(cam.Intrinsic.at<float>(1, 2));

		predictions[0] = fx * (pos_proj[0] / pos_proj[2]) + cx;
		predictions[1] = fy * (pos_proj[1] / pos_proj[2]) + cy;

		residuals[0] = predictions[0] - T(observed.x);
		residuals[1] = predictions[1] - T(observed.y);

		return true;
	}

	static ceres::CostFunction* Create(Point2d observed, CamInfo cam, Point3d point_ID) {
		return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 3, 3>(
			new SnavelyReprojectionError(observed, cam, point_ID)));
	}

};

void PoseEstimator::PnPSolver(MarkerInfo markers, vector<ModelInfo> reconstruct_model, CamInfo camera, PoseInfo& pose)
{
	image_points.clear();
	model_points.clear();

	ID = markers.markerID;
	for (int j = 0; j < reconstruct_model.size(); j++) {
		if (reconstruct_model[j].MarkerID == ID) {
			ID = j;
			break;
		}
	}
		
	for (int j = 0; j < markers.cornerLists.size(); j++) {
		for (int k = 0; k < 8; k++) {
			image_points.push_back(markers.cornerLists[j][k]);
			model_points.push_back(reconstruct_model[ID].corners[markers.featurePos[j] * 8 + k]);
		}
	}
	solvePnPRansac(model_points, image_points, camera.Intrinsic, camera.distCoeffs, pose.rvec, pose.tvec, false, 100, 2, 0.99, noArray(), SOLVEPNP_EPNP);
	PoseBA(image_points, model_points, pose, camera);
}

void PoseEstimator::PoseBA(vector<Point2f> imagePoints, vector<Point3f> worldPoints, PoseInfo pose, CamInfo camera)
{
	rot[0] = pose.rvec.at<double>(0, 0);
	rot[1] = pose.rvec.at<double>(1, 0);
	rot[2] = pose.rvec.at<double>(2, 0);
	trans[0] = pose.tvec.at<double>(0, 0);
	trans[1] = pose.tvec.at<double>(1, 0);
	trans[2] = pose.tvec.at<double>(2, 0);

	undistortPoints(imagePoints, imagePoints, camera.Intrinsic, camera.distCoeffs, noArray(), camera.Intrinsic);
	Problem problem;
	buildProblem(&problem, imagePoints, worldPoints, camera, pose);

	Solver::Options options;
	options.linear_solver_type = DENSE_SCHUR;
	options.gradient_tolerance = 1e-15;
	options.function_tolerance = 1e-15;
	options.parameter_tolerance = 1e-10;
	Solver::Summary summary;

	Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << "\n";

	pose.rvec.at<double>(0, 0) = rot[0];
	pose.rvec.at<double>(1, 0) = rot[1];
	pose.rvec.at<double>(2, 0) = rot[2];
	pose.tvec.at<double>(0, 0) = trans[0];
	pose.tvec.at<double>(1, 0) = trans[1];
	pose.tvec.at<double>(2, 0) = trans[2];
}

void PoseEstimator::buildProblem(Problem* problem, vector<Point2f> imagePoints, vector<Point3f> worldPoints, CamInfo camera, PoseInfo pose) {
	rot[0] = pose.rvec.at<double>(0, 0);
	rot[1] = pose.rvec.at<double>(1, 0);
	rot[2] = pose.rvec.at<double>(2, 0);
	trans[0] = pose.tvec.at<double>(0, 0);
	trans[1] = pose.tvec.at<double>(1, 0);
	trans[2] = pose.tvec.at<double>(2, 0);

	for (int i = 0; i < imagePoints.size(); ++i) {
			CostFunction* cost_function;
			cost_function = SnavelyReprojectionError::Create((Point2d)imagePoints[i], camera, (Point3d)worldPoints[i]);
			problem->AddResidualBlock(cost_function, NULL, rot, trans);
	}
}

void PoseEstimator::DenseSolver(const Mat& img, vector<ModelInfo> reconstruct_model, PoseInfo& pose)
{
	// To be updated
}
