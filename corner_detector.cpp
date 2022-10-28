#include "header/corner_detector.h"

using namespace std;
using namespace cv;

void corner_detector::adaptiveThreshold(const Mat& src, Mat& dst, int thresholdWindow){
    int col_number = src.cols / thresholdWindow;
    int row_number = src.rows / thresholdWindow;
    Mat extreme_min(row_number, col_number, CV_32FC1);
    Mat extreme_max(row_number, col_number, CV_32FC1);
    Mat extreme_min_final(row_number, col_number, CV_32FC1);
    Mat extreme_max_final(row_number, col_number, CV_32FC1);

    for (int i = 0; i < src.rows; i += thresholdWindow){
        for (int j = 0; j < src.cols; j += thresholdWindow){
            Rect rect(j, i, thresholdWindow, thresholdWindow);
            img_part = src(rect);
            minMaxLoc(img_part, &minVal, &maxVal, NULL, NULL);
            extreme_min.at<float>(row_count, col_count) = (float)minVal;
            extreme_max.at<float>(row_count, col_count) = (float)maxVal;
            col_count++;
        }
        col_count = 0;
        row_count++;
    }
    for (int i = 0; i < row_number - 2; i++){
        for (int j = 0; j < col_number - 2; j++){
            Rect rect(j, i, 3, 3);
            min_part = extreme_min(rect);
            minMaxLoc(min_part, &minVal, NULL, NULL, NULL);
            max_part = extreme_max(rect);
            minMaxLoc(max_part, NULL, &maxVal, NULL, NULL);
            extreme_min_final.at<float>(row_count_final, col_count_final) = (float)minVal;
            extreme_max_final.at<float>(row_count_final, col_count_final) = (float)maxVal;
            col_count_final++;
        }   
        col_count_final = 1;
        row_count_final++;
    }
    
    for (int i = 0; i < src.rows; i++){
        for (int j = 0; j < src.cols; j++){
            if (src.at<float>(i, j) < min((float)0.2, (extreme_max_final.at<float>(i / thresholdWindow, j / thresholdWindow) + extreme_min_final.at<float>(i / thresholdWindow, j / thresholdWindow)) / 2)){
                dst.at<uchar>(i, j) = 255;
            }
            else{
                dst.at<uchar>(i, j) = 0;
            }
        }
    }
}

void corner_detector::connectedComponentLabeling(const Mat& src, vector<vector<Point>>& quadArea, int method){
    nccomp_area = connectedComponentsWithStats(src, img_labeled, stats, centroids, 8, 4, CCL_BBDT);
    quadArea.resize(nccomp_area);

    bool illegal[nccomp_area] = {false};
    for (int i = 0; i < nccomp_area; i++){
        if (stats.at<int>(i, cv::CC_STAT_AREA) < 30 || stats.at<int>(i, cv::CC_STAT_AREA) > round(0.002 * src.cols * src.rows)){
            illegal[i] = true;
        }
    }
    for (int i = 0; i < img_labeled.rows; i++){
        for (int j = 0; j < img_labeled.cols; j++){
            if (!illegal[img_labeled.at<int>(i, j)])
                quadArea[img_labeled.at<int>(i, j)].push_back(Point(j, i));
        }
    } 
    int count = 0;
    for (auto iter = quadArea.begin(); iter != quadArea.end(); ){
        if (illegal[count++]){
            iter = quadArea.erase(iter);
        }
        else{
            iter++;
        }
    }

    // //去除过小区域，初始化颜色表
    // vector<cv::Vec3b> colors(nccomp_area);
    // colors[0] = cv::Vec3b(0,0,0); // background pixels remain black.
    // for(int i = 1; i < nccomp_area; i++ ) {
    //     colors[i] = cv::Vec3b(rand()%256, rand()%256, rand()%256);
    //     //去除面积小于30的连通域
    //     if (stats.at<int>(i, cv::CC_STAT_AREA) < 30 || stats.at<int>(i, cv::CC_STAT_AREA) > round(0.002 * src.cols * src.rows))
    //             colors[i] = cv::Vec3b(0,0,0); // small regions are painted with black too.
    // }
    // //按照label值，对不同的连通域进行着色
    // Mat img_color = cv::Mat::zeros(src.size(), CV_8UC3);
    // for( int y = 0; y < img_color.rows; y++ )
    //     for( int x = 0; x < img_color.cols; x++ )
    //     {
    //             int label = img_labeled.at<int>(y, x);
    //             img_color.at<cv::Vec3b>(y, x) = colors[label];
    //     }
    // cv::imshow("CCL", img_color);
    // cv::waitKey(0);
}

bool cmp_dis(const corners_pre& a, corners_pre& b){
    return a.dis_to_centroid < b.dis_to_centroid;
}

bool cmp_ang(const corners_pre& a, corners_pre& b){
    return a.angle_to_centroid < b.angle_to_centroid;
}

void corner_detector::edgeExtraction(const Mat& img, vector<vector<Point>>& quadArea, vector<Point2f>& corners_init, int KMeansIter){
    // Display
    Mat imgMark(img.rows, img.cols, CV_32FC3);
    cvtColor(img, imgMark, COLOR_GRAY2RGB);

    Mat Gx, Gy;

    Sobel(img, Gx, -1, 1, 0);
    Sobel(img, Gy, -1, 0, 1);

    Mat Gangle(Gx.rows, Gx.cols, CV_32FC1);
    Mat Gpow(Gy.rows, Gy.cols, CV_32FC1);

    for (int i = 0; i < Gx.rows; i++)
        for (int j = 0; j < Gx.cols; j++){
            Gangle.at<float>(i, j) = atan2(Gy.at<float>(i, j), Gx.at<float>(i, j)) * 180 / CV_PI;
            Gpow.at<float>(i, j) = sqrt(Gy.at<float>(i, j) * Gy.at<float>(i, j) + Gx.at<float>(i, j) * Gx.at<float>(i, j));    
        }

    for (int i = 0; i < quadArea.size(); i++){
        flag_line_number = false;
        flag_illegal_corner = false;

        edge_angle.clear();
        edge_point.clear();
        for (int j = 0; j < 4; j++){
            edge_angle_cluster[j].clear();
            edge_point_cluster[j].clear();
        }
        corners_p.clear();

        Mat A(2, 2, CV_32FC1);
        Mat B(2, 1, CV_32FC1);
        Mat sol(2, 1, CV_32FC1);

        moment = moments(quadArea[i]);
        if (moment.m00 != 0){
            area_center.x = moment.m10 / moment.m00;
            area_center.y = moment.m01 / moment.m00;
        }
        
        for (int j = 0; j < 4; j++){
            edge_angle_cluster[j].clear();
        }
        Gmax = -1, Gmin = 1;
        for (int j = 0; j < quadArea[i].size(); j++){
            if (Gpow.at<float>(quadArea[i][j].y, quadArea[i][j].x) < Gmin) Gmin = Gpow.at<float>(quadArea[i][j].y, quadArea[i][j].x);
            if (Gpow.at<float>(quadArea[i][j].y, quadArea[i][j].x) > Gmax) Gmax = Gpow.at<float>(quadArea[i][j].y, quadArea[i][j].x);
        }
        for (int j = 0; j < quadArea[i].size(); j++){
            if (Gpow.at<float>(quadArea[i][j].y, quadArea[i][j].x) > (Gmax + Gmin) / 2){
                edge_angle.push_back(Gangle.at<float>(quadArea[i][j].y, quadArea[i][j].x));
                edge_point.push_back(quadArea[i][j]);
            }
        }
        measure = kmeans(edge_angle, 4, kmeans_label, TermCriteria(TermCriteria::EPS, 10, 0.5), 5, KMEANS_RANDOM_CENTERS);
        for (int j = 0; j < edge_angle.size(); j++){
            edge_angle_cluster[kmeans_label[j]].push_back(edge_angle[j]);
            edge_point_cluster[kmeans_label[j]].push_back(edge_point[j]);
        }
        for (int j = 0; j < 4; j++){
            // No enough points to fit a line
            if (edge_angle_cluster[j].size() < 3){ 
                flag_line_number = true;
                break;
            }
            fitLine(edge_point_cluster[j], line_func[j], DIST_L2, 0, 0.01, 0.01);
        }
        if (flag_line_number) continue;
        for (int j = 0; j < 3; j++){
            for (int k = j + 1; k < 4; k++){
                A.at<float>(0, 0) = line_func[j][1];
                A.at<float>(0, 1) = -line_func[j][0];
                A.at<float>(1, 0) = line_func[k][1];
                A.at<float>(1, 1) = -line_func[k][0];
                B.at<float>(0, 0) = line_func[j][1] * line_func[j][2] - line_func[j][0] * line_func[j][3];
                B.at<float>(1, 0) = line_func[k][1] * line_func[k][2] - line_func[k][0] * line_func[k][3];
                if (determinant(A) != 0){
                    solve(A, B, sol);
                    c.intersect.x = sol.at<float>(0, 0);
                    c.intersect.y = sol.at<float>(1, 0);
                    c.dis_to_centroid = sqrt((c.intersect.x - area_center.x) * (c.intersect.x - area_center.x) + (c.intersect.y - area_center.y) * (c.intersect.y - area_center.y));
                    c.angle_to_centroid = atan2(c.intersect.y - area_center.y, c.intersect.x - area_center.x);
                    if (c.dis_to_centroid < img.cols && c.dis_to_centroid < img.rows)
                        corners_p.push_back(c);
                }
            }
        }
        if (corners_p.size() < 4) continue;
        sort(corners_p.begin(), corners_p.end(), cmp_dis);
        corners_p.resize(4);
        sort(corners_p.begin(), corners_p.end(), cmp_ang);
        for (int j = 0; j < 4; j++){
            if (corners_p[j].intersect.x < 0 || corners_p[j].intersect.y < 0 || corners_p[j].intersect.x > img.cols || corners_p[j].intersect.y > img.rows || !quadJudgment(corners_p, quadArea[i].size())){
                flag_illegal_corner = true;
                break;
            }
        }
        if (flag_illegal_corner) continue;
        for (int j = 0; j < 4; j++){
            corners_init.push_back(corners_p[j].intersect);
        }
        for (int j = 0; j < quadArea[i].size(); j++){
            circle(imgMark, quadArea[i][j], 1, Scalar(0, 250, 0), -1);
        }
        for (int j = 0; j < corners_init.size(); j++){
            circle(imgMark, corners_init[j], 1, Scalar(120, 150, 0), -1);
        }
        
    }
    imshow("corners initial", imgMark);
    waitKey(0);
}

bool corner_detector::quadJudgment(vector<corners_pre> corners, int areaPixelNumber){
    quad_area = 0;
    for (int i = 0; i < 3; i++){
        quad_area += corners[i].intersect.x * corners[i + 1].intersect.y - corners[i].intersect.y * corners[i + 1].intersect.x; 
    }
    quad_area += corners[3].intersect.x * corners[0].intersect.y - corners[3].intersect.y * corners[0].intersect.x; 
    quad_area /= 2;
    RAC = abs(abs(quad_area) - areaPixelNumber) / areaPixelNumber;
    if (RAC > threshold_RAC) return false;
    return true;    
}

void corner_detector::edgeSubPix(const Mat& src, vector<Point2f> corners_init, vector<Point2f> corners_refined, int subPixWindow){
    corners_refined = corners_init;
}

bool corner_detector::parallelogramJudgment(vector<Point2f> corners){
    return true;
}

void corner_detector::featureRecovery(vector<vector<Point2f>> corners_refined, vector<featureInfo> features){}

void corner_detector::featureExtraction(const Mat& img, vector<vector<Point2f>> feature_src, vector<featureInfo> feature_dst){}

void corner_detector::markerOrganization(vector<featureInfo> feature, vector<MarkerInfo> markers){}

void corner_detector::markerDecoder(vector<MarkerInfo> markers_src, vector<MarkerInfo> markers_dst){}