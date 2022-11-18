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
    // Mat img_color = cv::Mat::zeros(src.size(), CV_32FC3);
    // cvtColor(src, img_color, COLOR_GRAY2RGB);
    // for( int y = 0; y < img_color.rows; y++ )
    //     for( int x = 0; x < img_color.cols; x++ )
    //     {
    //             int label = img_labeled.at<int>(y, x);
    //             //img_color.at<cv::Vec3b>(y, x) = colors[label];
    //             if (label != 0)
    //                 circle(img_color, Point(x, y), 1, Scalar(colors[label]));
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

void corner_detector::edgeExtraction(const Mat& img, vector<vector<Point>>& quadArea, vector<vector<Point2f>>& corners_init, vector<double>& meanG, int KMeansIter){
    // Display
    Mat imgMark(img.rows, img.cols, CV_32FC3);
    cvtColor(img, imgMark, COLOR_GRAY2RGB);

    Mat Gx, Gy;

    clock_t start,finish;  
    double duration; 
    
    Sobel(img, Gx, -1, 1, 0, 3);
    Sobel(img, Gy, -1, 0, 1, 3);
    
    start = clock();

    Mat Gangle, Gpow;

    magnitude(Gx, Gy, Gpow);
    phase(Gx, Gy, Gangle, true);

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
        Gmax = -1;
        for (int j = 0; j < quadArea[i].size(); j++){
            if (Gpow.at<float>(quadArea[i][j].y, quadArea[i][j].x) > Gmax) Gmax = Gpow.at<float>(quadArea[i][j].y, quadArea[i][j].x);
        }
        double mean_G = 0.0;
        int num = 0;
        for (int j = 0; j < quadArea[i].size(); j++){
            if (Gpow.at<float>(quadArea[i][j].y, quadArea[i][j].x) > Gmax / 3){
                mean_G += Gpow.at<float>(quadArea[i][j].y, quadArea[i][j].x);
                num++;
                edge_angle.push_back(Gangle.at<float>(quadArea[i][j].y, quadArea[i][j].x));
                edge_point.push_back(quadArea[i][j]);
            }
        }
        mean_G /= num;
        //for (int j = 0; j < edge_point.size(); j++)
        //   circle(imgMark, edge_point[j], 1, Scalar(rand()%256, rand()%256, rand()%256));
        putText(imgMark, to_string(i), edge_point[0], FONT_ITALIC, 0.5, Scalar(120, 120, 120));

        measure = kmeans(edge_angle, 4, kmeans_label, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 50, 1.0), KMeansIter, KMEANS_RANDOM_CENTERS);
        for (int j = 0; j < edge_angle.size(); j++){
            edge_angle_cluster[kmeans_label[j]].push_back(edge_angle[j]);
            edge_point_cluster[kmeans_label[j]].push_back(edge_point[j]);
        }
        for (int j = 0; j < 4; j++){
            // No enough points to fit a line
            if (edge_angle_cluster[j].size() < 2){ 
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
        meanG.push_back(mean_G);

        // for (int j = 0; j < quadArea[i].size(); j++){
        //     circle(imgMark, quadArea[i][j], 1, Scalar(0, 250, 0), -1);
        // }
        for (int j = 0; j < corners_p.size(); j++){
            circle(imgMark, corners_p[j].intersect, 3, Scalar(120, 150, 0), -1);
        }
    }    
      
    imshow("corners initial", imgMark);
    waitKey(0);
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
    corners_refined = corners_init;
    for (int i = 0; i < corners_init.size(); i++){
        for (int j = 0; j < corners_init[i].size(); j++){
            corners_refined[i][j] *= 2; 
        }
    }
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

void corner_detector::featureRecovery(vector<vector<Point2f>>& corners_refined, vector<featureInfo>& features, vector<double>& meanG){
    memset(isVisited, false, sizeof(isVisited));
    
    corner_dist.resize(corners_refined.size());
    corner_centers.clear();
    corner_angles_1.clear();
    corner_angles_2.clear();
    
    for (int i = 0; i < corners_refined.size(); i++){
        sum_x = 0, sum_y = 0;
        for (int j = 0; j < corners_refined[i].size(); j++){
            sum_x += corners_refined[i][j].x;
            sum_y += corners_refined[i][j].y;
        }
        Point2f center_ = Point2f(sum_x * 1.0 / corners_refined[i].size(), sum_y * 1.0 / corners_refined[i].size());
        corner_centers.push_back(center_);
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
                tag1 = false;
                tag2 = false;
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

                if ((tag1 && tag2) && (dist1_long > dist1_short || dist2_long > dist2_short)
                    && (feature_half_length - (dist1_long + dist2_long) / 2 < 2 * (dist1_short + dist2_short))
                    && (abs(dist1_short - dist2_short) < min(dist1_short, dist2_short) * 0.2)
                    && (dist1_long + dist2_long > 2 * (dist1_short + dist2_short)))
                {
                    isVisited[i] = true;
                    isVisited[j] = true;
                    tag_darker = meanG[i] > meanG[j] ? true : false;
                    features.push_back(featureOrganization(corners_refined[i], corners_refined[j], corner_centers[i], corner_centers[j], feature_angle, tag_darker));
                    j = corners_refined.size();
                }
            }
        }
    }
}

featureInfo corner_detector::featureOrganization(vector<Point2f> quad1, vector<Point2f> quad2, Point2f quad1_center, Point2f quad2_center, float feature_angle, bool darker){
    float angle_max = 0, angle_min = 360;
    int pos_quad1 = -1, pos_quad2 = -1;
    Fea.corners.clear();
    Fea.ID = -1;
    Fea.feature_angle = feature_angle;
    for (int i = 0; i < 4; i++){
        angle_quad1[i] = atan2(quad1_center.y - quad1[i].y, quad1_center.x - quad1[i].x) * 180 / CV_PI;
        angle_quad2[i] = atan2(quad2_center.y - quad2[i].y, quad2_center.x - quad2[i].x) * 180 / CV_PI;
    }    
    for (int i = 0; i < 4; i++){
        if (min(360 - abs(angle_quad1[(i + 2) % 4] - feature_angle), abs(angle_quad1[(i + 2) % 4] - feature_angle)) + min(360 - abs(angle_quad1[(i + 3) % 4] - feature_angle), abs(angle_quad1[(i + 3) % 4] - feature_angle)) < angle_min){
            angle_min = min(360 - abs(angle_quad1[(i + 2) % 4] - feature_angle), abs(angle_quad1[(i + 2) % 4] - feature_angle)) + min(360 - abs(angle_quad1[(i + 3) % 4] - feature_angle), abs(angle_quad1[(i + 3) % 4] - feature_angle));
            pos_quad1 = i;
        }
        if (min(360 - abs(angle_quad2[(i + 2) % 4] - feature_angle), abs(angle_quad2[(i + 2) % 4] - feature_angle)) + min(360 - abs(angle_quad2[(i + 3) % 4] - feature_angle), abs(angle_quad2[(i + 3) % 4] - feature_angle)) > angle_max){
            angle_max = min(360 - abs(angle_quad2[(i + 2) % 4] - feature_angle), abs(angle_quad2[(i + 2) % 4] - feature_angle)) + min(360 - abs(angle_quad2[(i + 3) % 4] - feature_angle), abs(angle_quad2[(i + 3) % 4] - feature_angle));
            pos_quad2 = i;
        }
    }
    for (int i = 0; i < 4; i++){
        Fea.corners.push_back(quad1[(i + pos_quad1) % 4]);
    }
    for (int i = 0; i < 4; i++){
        Fea.corners.push_back(quad2[(i + pos_quad2) % 4]);
    }
    Fea.feature_center = Point2f((Fea.corners[0].x + Fea.corners[1].x + Fea.corners[4].x + Fea.corners[5].x) / 4, (Fea.corners[0].y + Fea.corners[1].y + Fea.corners[4].y + Fea.corners[5].y) / 4);
    Fea.firstDarker = darker ? true : false;
    return Fea;
}

void corner_detector::featureExtraction(const Mat& img, vector<featureInfo> feature_src, vector<featureInfo> feature_dst){
    for (int i = 0; i < feature_src.size(); i++){
        length_1[0] = distance_2points(feature_src[i].corners[0], feature_src[i].corners[3]);
        length_1[1] = distance_2points(feature_src[i].corners[3], feature_src[i].corners[6]);
        length_1[2] = distance_2points(feature_src[i].corners[6], feature_src[i].corners[5]);
        length_1[3] = distance_2points(feature_src[i].corners[0], feature_src[i].corners[5]);
        length_2[0] = distance_2points(feature_src[i].corners[1], feature_src[i].corners[2]);
        length_2[1] = distance_2points(feature_src[i].corners[2], feature_src[i].corners[7]);
        length_2[2] = distance_2points(feature_src[i].corners[7], feature_src[i].corners[4]);
        length_2[3] = distance_2points(feature_src[i].corners[1], feature_src[i].corners[4]);

        cross_ratio_1 = (length_1[0] + length_1[1]) * (length_1[2] + length_1[1]) / ((length_1[1] * length_1[3]));
        cross_ratio_2 = (length_2[0] + length_2[1]) * (length_2[2] + length_2[1]) / ((length_2[1] * length_2[3]));
        cross_ratio = (cross_ratio_1 + cross_ratio_2) / 2;
    }
}

void corner_detector::markerOrganization(vector<featureInfo> feature, vector<MarkerInfo>& markers){
    for (int i = 0; i < feature.size(); i++){
        father[i] = i;
    }
    for (int i = 0; i < feature.size() - 1; i++){
        for (int j = i + 1; j < feature.size(); j++){
            vector_center = feature[i].feature_center - feature[j].feature_center;
            vector_longedge = feature[i].corners[0] - feature[i].corners[5];
            center_angle = (vector_center.x * vector_longedge.x + vector_center.y * vector_longedge.y) / sqrt((vector_center.x * vector_center.x + vector_center.y * vector_center.y) * (vector_longedge.x * vector_longedge.x + vector_longedge.y * vector_longedge.y));
            if ((abs(feature[i].feature_angle - feature[j].feature_angle) < threshold_angle) && (abs(area(feature[i]) - area(feature[j])) < area_ratio * min(area(feature[i]), area(feature[j]))) && (distance_2points(feature[i].feature_center, feature[j].feature_center) < area(feature[i]) + area(feature[j])) && (abs(center_angle) < threshold_vertical)){
                father[j] = union_find(i);
            }
        }
    }

    father_database.clear();
    cnt = 0;
    vector<vector<int>> marker_ID(feature.size());
    father_database.push_back(father[0]);
    marker_ID[cnt++].push_back(0);
    
    for (int i = 1; i < feature.size(); i++){
        find_father = false;
        for (int j = 0; j < father_database.size(); j++){
            if (father[i] == father_database[j]){
                marker_ID[j].push_back(i);    
                find_father = true;
                break;
            }
        }
        if (!find_father){
            father_database.push_back(father[i]);
            marker_ID[cnt++].push_back(i);
        }        
    }
    markers.clear();
    for (int i = 0; i < cnt; i++){
        MarkerInfo mark;
        for (int j = 0; j < marker_ID[i].size(); j++){
            mark.cornerLists.push_back(feature[marker_ID[i][j]].corners);
            mark.feature_center.push_back(feature[marker_ID[i][j]].feature_center);
            mark.edge_length.push_back(distance_2points(feature[marker_ID[i][j]].corners[0], feature[marker_ID[i][j]].corners[1]));
            mark.feature_ID.push_back(feature[marker_ID[i][j]].ID);
        } 
        markers.push_back(mark);
    }
}

bool cmp_x(Point2f& a, Point2f& b){
    return a.x < b.x;
}

bool cmp_y(Point2f& a, Point2f& b){
    return a.y < b.y;
}

void corner_detector::markerDecoder(vector<MarkerInfo> markers_src, vector<MarkerInfo> markers_dst, Mat1i& state){
    for (int i = 0; i < markers_src.size(); i++){
        marker_angle = atan2(markers_src[i].feature_center[0].y - markers_src[i].feature_center[1].y, markers_src[i].feature_center[0].x - markers_src[i].feature_center[1].x) * 180 / CV_PI;
        
        if (abs(marker_angle) > 45 && abs(marker_angle) < 135){
            sort(markers_src[i].feature_center.begin(), markers_src[i].feature_center.end(), cmp_x);
        }
        else{
            sort(markers_src[i].feature_center.begin(), markers_src[i].feature_center.end(), cmp_y);
        }

        memset(code, -1, sizeof(code));
        pos_now = 0;
        code[pos_now] = markers_src[i].feature_ID[0];

        for (int j = 1; j < markers_src[i].feature_center.size(); j++){
            float dist = distance_2points(markers_src[i].feature_center[j], markers_src[i].feature_center[j - 1]);
            float gap_dist = dist - (markers_src[i].edge_length[j] + markers_src[i].edge_length[j - 1]) / 2;
            int gap = round((gap_dist - (markers_src[i].edge_length[j] + markers_src[i].edge_length[j - 1]) / 4) / (3 * (markers_src[i].edge_length[j] + markers_src[i].edge_length[j - 1]) / 4));
            pos_now += gap + 1; 
            code[pos_now] = markers_src[i].feature_ID[j];
        }

        Pos_ID = match_dictionary(code, state, pos_now);
    }
}

float corner_detector::distance_2points(Point2f point1, Point2f point2){
    return sqrt((point1.x - point2.x) * (point1.x - point2.x) + (point1.y - point2.y) * (point1.y - point2.y));
}

int corner_detector::union_find(int x){
    return x == father[x] ? x : (father[x] = union_find(father[x]));
}

float corner_detector::area(featureInfo feature){
    float feature_area = 0;
    for (int i = 0; i < 4; i++){
        feature_area += feature.corners[pose[i]].x * feature.corners[pose[(i + 1) % 4]].y - feature.corners[pose[i]].y * feature.corners[pose[(i + 1) % 4]].x; 
    }
    feature_area /= 2;
    return abs(feature_area);
}

pos_with_ID corner_detector::match_dictionary(int *code, Mat1i& state, int length){
    pos_with_ID POS_ID;
    max_coverage = -1;
    second_coverage = -1;
    coverage_now = -1;
    for (int i = 0; i < state.rows; i++){
        for (int j = 0; j < state.cols; j++){
            coverage_now = 0;
            for (int k = 0; k < length; k++){
                if (state.at<int>(i, (j + k) % state.cols) == code[k]){
                    coverage_now++;
                }
            }
            if (coverage_now > max_coverage){
                max_coverage = coverage_now;
                max_coverage_pos = Point(i, j);
                direc = 1;
            }
            else if (coverage_now > second_coverage){
                second_coverage = coverage_now;
            }
        }
    }
    for (int i = 0; i < state.rows; i++){
        for (int j = 0; j < state.cols; j++){
            coverage_now = 0;
            for (int k = 0; k < length; k++){
                if (state.at<int>(i, (j - k + state.cols) % state.cols) == code[k]){
                    coverage_now++;
                }
            }
            if (coverage_now > max_coverage){
                max_coverage = coverage_now;
                max_coverage_pos = Point(i, j);
                direc = -1;
            }
            else if (coverage_now > second_coverage){
                second_coverage = coverage_now;
            }
        }
    }
    if (max_coverage > 0.8 * length && max_coverage > second_coverage){
        POS_ID.ID = max_coverage_pos.x;
        for (int i = 0; i < length; i++){
            if (code[i] != -1){
                POS_ID.pos.push_back(max_coverage_pos.y + direc * i);
            }
        }            
    }
    return POS_ID;
}
