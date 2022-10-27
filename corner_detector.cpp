#include "header/corner_detector.h"

using namespace std;
using namespace cv;

void corner_detector::adaptiveThreshold(const Mat& src, Mat& dst, int thresholdWindow){
    int col_number = src.cols / thresholdWindow;
    int row_number = src.rows / thresholdWindow;
    Mat extreme_min(row_number, col_number, CV_64FC1);
    Mat extreme_max(row_number, col_number, CV_64FC1);
    Mat extreme_min_final(row_number, col_number, CV_64FC1);
    Mat extreme_max_final(row_number, col_number, CV_64FC1);

    for (int i = 0; i < src.rows; i += thresholdWindow){
        for (int j = 0; j < src.cols; j += thresholdWindow){
            Rect rect(j, i, thresholdWindow, thresholdWindow);
            img_part = src(rect);
            minMaxLoc(img_part, &minVal, &maxVal, NULL, NULL);
            extreme_min.at<double>(row_count, col_count) = minVal;
            extreme_max.at<double>(row_count, col_count) = maxVal;
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
            extreme_min_final.at<double>(row_count_final, col_count_final) = minVal;
            extreme_max_final.at<double>(row_count_final, col_count_final) = maxVal;
            col_count_final++;
        }   
        col_count_final = 1;
        row_count_final++;
    }
    
    for (int i = 0; i < src.rows; i++){
        for (int j = 0; j < src.cols; j++){
            if (src.at<double>(i, j) < min(0.2, (extreme_max_final.at<double>(i / thresholdWindow, j / thresholdWindow) + extreme_min_final.at<double>(i / thresholdWindow, j / thresholdWindow)) / 2)){
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
        if (stats.at<int>(i, cv::CC_STAT_AREA) < 30 || stats.at<int>(i, cv::CC_STAT_AREA) > 1000){
            illegal[i] = true;
        }
    }
    for (int i = 0; i < img_labeled.rows; i++){
        for (int j = 0; j < img_labeled.cols; j++){
            if (!illegal[img_labeled.at<int>(i, j)])
                quadArea[img_labeled.at<int>(i, j)].push_back(Point(i, j));
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
    //     if (stats.at<int>(i, cv::CC_STAT_AREA) < 30 || stats.at<int>(i, cv::CC_STAT_AREA) > 1000)
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

void corner_detector::edgeExtraction(const Mat& img, vector<vector<Point>>& quadArea, vector<Point2f>& corners_init, int KMeansIter){
    Mat Gx, Gy;

    Sobel(img, Gx, -1, 1, 0);
    Sobel(img, Gy, -1, 0, 1);

    Mat Gangle(Gx.rows, Gx.cols, CV_32FC1);
    Mat Gpow(Gy.rows, Gy.cols, CV_32FC1);

    for (int i = 0; i < Gx.rows; i++)
        for (int j = 0; j < Gx.cols; j++){
            Gangle.at<float>(i, j) = atan2(Gy.at<double>(i, j), Gx.at<double>(i, j)) * 180 / CV_PI;
            Gpow.at<float>(i, j) = sqrt(Gy.at<double>(i, j) * Gy.at<double>(i, j) + Gx.at<double>(i, j) * Gx.at<double>(i, j));    
        }

    for (int i = 0; i < quadArea.size(); i++){
        edge_angle.clear();
        for (int j = 0; j < 4; j++){
            edge_angle_cluster[j].clear();
        }
        Gmax = -1, Gmin = 1;
        for (int j = 0; j < quadArea[i].size(); j++){
            if (Gpow.at<float>(quadArea[i][j].x, quadArea[i][j].y) < Gmin) Gmin = Gpow.at<float>(quadArea[i][j].x, quadArea[i][j].y);
            if (Gpow.at<float>(quadArea[i][j].x, quadArea[i][j].y) > Gmax) Gmax = Gpow.at<float>(quadArea[i][j].x, quadArea[i][j].y);
        }
        for (int j = 0; j < quadArea[i].size(); j++){
            if (Gpow.at<float>(quadArea[i][j].x, quadArea[i][j].y) > (Gmax + Gmin) / 2)
                edge_angle.push_back(Gangle.at<float>(quadArea[i][j].x, quadArea[i][j].y));
        }
        measure = kmeans(edge_angle, 4, kmeans_label, TermCriteria(TermCriteria::EPS, 10, 0.5), 5, KMEANS_RANDOM_CENTERS);
        for (int j = 0; j < edge_angle.size(); j++){
            edge_angle_cluster[kmeans_label[j]].push_back(edge_angle[j]);
        }
        for (int j = 0; j < 4; j++){
            // No enough points to fit a line
            if (edge_angle_cluster[j].size() < 3){ 
                break;
            }

            edge_angle_all = 0;
            for (int k = 0; k < edge_angle_cluster[j].size(); k++){
                edge_angle_all += edge_angle_cluster[j][k];
            }

            if (abs(edge_angle_all / edge_angle_cluster[j].size()) > 45 || abs(edge_angle_all / edge_angle_cluster[j].size()) < 135){
                fitLine(edge_angle_cluster[j], line_func[j], DIST_L2, 0, 0.01, 0.01);
            }

        }
    }
}

bool corner_detector::quadJudgment(vector<Point2f> corners_init, int areaPixelNumber, int threshold){
    return true;    
}

void corner_detector::edgeSubPix(const Mat& src, vector<Point2f> corners_init, vector<Point2f> corners_refined, int subPixWindow){}

bool corner_detector::parallelogramJudgment(vector<Point2f> corners){
    return true;
}

void corner_detector::featureRecovery(vector<vector<Point2f>> corners_refined, vector<featureInfo> features){}

void corner_detector::featureExtraction(const Mat& img, vector<vector<Point2f>> feature_src, vector<featureInfo> feature_dst){}

void corner_detector::markerOrganization(vector<featureInfo> feature, vector<MarkerInfo> markers){}

void corner_detector::markerDecoder(vector<MarkerInfo> markers_src, vector<MarkerInfo> markers_dst){}