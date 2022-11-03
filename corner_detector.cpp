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

void corner_detector::edgeExtraction(const Mat& img, vector<vector<Point>>& quadArea, vector<vector<Point2f>>& corners_init, int KMeansIter){
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
        sum_x = 0; 
        sum_y = 0;

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
        
        for (int j = 0; j < quadArea[i].size(); j++){
            sum_x += quadArea[i][j].x;
            sum_y += quadArea[i][j].y;
        }
        area_center.x = sum_x / quadArea[i].size();
        area_center.y = sum_y / quadArea[i].size();
        
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
                    c.angle_to_centroid = atan2(c.intersect.y - area_center.y, c.intersect.x - area_center.x) * 180 / CV_PI;
                    if (c.dis_to_centroid < img.cols && c.dis_to_centroid < img.rows)
                        corners_p.push_back(c);
                }
            }
        }
        if (corners_p.size() < 4) continue;
        sort(corners_p.begin(), corners_p.end(), cmp_dis);
        corners_p.resize(4);
        sort(corners_p.begin(), corners_p.end(), cmp_ang);
        
        if (!quadJudgment(corners_p, quadArea[i].size())) continue;
        for (int j = 0; j < 4; j++){
            if (corners_p[j].intersect.x < 0 || corners_p[j].intersect.y < 0 || corners_p[j].intersect.x > img.cols || corners_p[j].intersect.y > img.rows){
                flag_illegal_corner = true;
                break;
            }
        }
        if (flag_illegal_corner) continue;
        corners_pass.clear();
        for (int j = 0; j < 4; j++){
            corners_pass.push_back(corners_p[j].intersect);
        }
        corners_init.push_back(corners_pass);

        for (int j = 0; j < quadArea[i].size(); j++){
            circle(imgMark, quadArea[i][j], 1, Scalar(0, 250, 0), -1);
        }
        for (int j = 0; j < corners_p.size(); j++){
            circle(imgMark, corners_p[j].intersect, 3, Scalar(120, 150, 0), -1);
        }
    }    
    // imshow("corners initial", imgMark);
    // waitKey(0);
}

bool corner_detector::quadJudgment(vector<corners_pre>& corners, int areaPixelNumber){
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

void corner_detector::edgeSubPix(const Mat& src, vector<vector<Point2f>>& corners_init, vector<vector<Point2f>>& corners_refined, int subPixWindow){
    
}

bool corner_detector::parallelogramJudgment(vector<Point2f> corners){
    sum_x = 0, sum_y = 0;
    for (int j = 0; j < corners.size(); j++){
        sum_x += corners[j].x;
        sum_y += corners[j].y;
    }
    corner_center.x = sum_x / corners.size();
    corner_center.y = sum_y / corners.size();

    for (int i = 0; i < 4; i++){
        dist_to_center[i] = sqrt((corner_center.x - corners[i].x) * (corner_center.x - corners[i].x) + (corner_center.y - corners[i].y) * (corner_center.y - corners[i].y));
    }
    coeff_1 = abs(dist_to_center[0] - dist_to_center[2]) / (dist_to_center[0] + dist_to_center[2]);
    coeff_2 = abs(dist_to_center[1] - dist_to_center[3]) / (dist_to_center[1] + dist_to_center[3]);
    if (coeff_1 < diff_percentage && coeff_2 < diff_percentage) return true;
    return false;
}

void corner_detector::featureRecovery(vector<vector<Point2f>>& corners_refined, vector<featureInfo>& features){
    memset(isVisited, false, sizeof(isVisited));
    tag1 = false;
    tag2 = false;

    corner_dist.resize(corners_refined.size());
    corner_centers.clear();
    corner_angles_1.clear();
    corner_angles_2.clear();
    
    for (int i = 0; i < corners_refined.size() - 1; i++){
        sum_x = 0, sum_y = 0;
        for (int j = 0; j < corners_refined[i].size(); j++){
            sum_x += corners_refined[i][j].x;
            sum_y += corners_refined[i][j].y;
        }
        corner_centers.push_back((Point2f)(sum_x / corners_refined[i].size(), sum_y / corners_refined[i].size()));
        for (int j = 0; j < 4; j++){
            corner_dist[i][j] = sqrt((corners_refined[i][j].x - corners_refined[i][(j + 1) % 4].x) * (corners_refined[i][j].x - corners_refined[i][(j + 1) % 4].x) + (corners_refined[i][j].y - corners_refined[i][(j + 1) % 4].y) * (corners_refined[i][j].y - corners_refined[i][(j + 1) % 4].y));
        }
        corner_angles_1.push_back(atan2(corners_refined[i][0].y - corners_refined[i][1].y, corners_refined[i][0].x - corners_refined[i][1].x) * 180 / CV_PI);
        corner_angles_2.push_back(atan2(corners_refined[i][1].y - corners_refined[i][2].y, corners_refined[i][1].x - corners_refined[i][2].x) * 180 / CV_PI);
    }
    for (int i = 0; i < corners_refined.size() - 1; i++){
        if (isVisited[i]) continue;
        for (int j = i + 1; j < corners_refined.size(); j++){
            if (!isVisited[j]){
                feature_angle = atan2(corner_centers[i].y - corner_centers[j].y, corner_centers[i].x - corner_centers[j].x) * 180 / CV_PI;
                feature_half_length = sqrt((corner_centers[i].y - corner_centers[j].y) * (corner_centers[i].y - corner_centers[j].y) + (corner_centers[i].x - corner_centers[j].x) * (corner_centers[i].x - corner_centers[j].x));
                if (abs(feature_angle - corner_angles_1[i]) < threshold_angle || abs(abs(feature_angle - corner_angles_1[i]) - 180) < threshold_angle){
                    tag1 = true;
                    dist1_long = (corner_dist[i][0] + corner_dist[i][2]) / 2;
                    dist1_short = (corner_dist[i][1] + corner_dist[i][3]) / 2;
                }
                if (abs(feature_angle - corner_angles_2[i]) < threshold_angle || abs(abs(feature_angle - corner_angles_2[i]) - 180) < threshold_angle){
                    tag1 = true;
                    dist1_short = (corner_dist[i][0] + corner_dist[i][2]) / 2;
                    dist1_long = (corner_dist[i][1] + corner_dist[i][3]) / 2;
                }
                if (abs(feature_angle - corner_angles_1[j]) < threshold_angle || abs(abs(feature_angle - corner_angles_1[j]) - 180) < threshold_angle){
                    tag2 = true;
                    dist2_long = (corner_dist[j][0] + corner_dist[j][2]) / 2;
                    dist2_short = (corner_dist[j][1] + corner_dist[j][3]) / 2;
                }
                if (abs(feature_angle - corner_angles_2[j]) < threshold_angle || abs(abs(feature_angle - corner_angles_2[j]) - 180) < threshold_angle){
                    tag2 = true;
                    dist2_short = (corner_dist[j][0] + corner_dist[j][2]) / 2;
                    dist2_long = (corner_dist[j][1] + corner_dist[j][3]) / 2;
                }

                if (tag1 && tag2) && (dist1_long > dist1_short || dist2_long > dist2_short)
                    && (feature_half_length - (dist1_long + dist2_long) / 2 < 1.5 * (dist1_short + dist2_short))
                    && (abs(dist1_short - dist2_short) < min(dist1_short, dist2_short) * 0.2)
                    && (dist1_long + dist2_long > 2 * (dist1_short + dist2_short))
                {
                    isVisited[i] = true;
                    isVisited[j] = true;
                    features.push_back(featureOrganization(corners_refined[i], corners_refined[j], corner_centers[i], corner_centers[j], feature_angle));
                }
            }
        }
    }
}

featureInfo corner_detector::featureOrganization(vector<Point2f> quad1, vector<Point2f> quad2, Point2f quad1_center, Point2f quad2_center, float feature_angle){
    Fea.ID = -1;
    Fea.feature_angle = feature_angle;
    middle_pos = 
}

void corner_detector::featureExtraction(const Mat& img, vector<featureInfo> feature_src, vector<featureInfo> feature_dst){}

void corner_detector::markerOrganization(vector<featureInfo> feature, vector<MarkerInfo> markers){}

void corner_detector::markerDecoder(vector<MarkerInfo> markers_src, vector<MarkerInfo> markers_dst){}