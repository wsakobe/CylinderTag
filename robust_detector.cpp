// Experimental opt-in detector. The original detect() implementation remains
// in CylinderTag.cpp; its thresholding, pairing and decoder are not replaced.
#include "header/CylinderTag.h"

namespace {
float robustDistance(cv::Point2f a, cv::Point2f b) {
    return std::sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
}

template <typename T, typename Compare>
std::vector<std::size_t> sort_permutation(
    const std::vector<T> vec,
    Compare compare)
{
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(),
        [&](std::size_t i, std::size_t j) { return compare(vec[i], vec[j]); });
    return p;
}

template <typename T>
std::vector<T> apply_permutation(
    const std::vector<T>& vec,
    const std::vector<std::size_t>& p)
{
    std::vector<T> sorted_vec(vec.size());
    std::transform(p.begin(), p.end(), sorted_vec.begin(),
        [&](std::size_t i) { return vec[i]; });
    return sorted_vec;
}


}

void CylinderTag::detectRobust(const Mat& img, vector<MarkerInfo>& result, int adaptiveThresh, const bool refine, int radius) {
    if(img.empty()||img.type()!=CV_8UC1||adaptiveThresh<1||radius<1)
        throw std::invalid_argument("detect expects a nonempty 8-bit grayscale image and positive windows");
    vector<int> windows={adaptiveThresh};
    if(multiThreshold) windows={adaptiveThresh,2*adaptiveThresh+1,4*adaptiveThresh+1};
    vector<MarkerInfo> hypotheses;
    DetectionStats selected;
    vector<MarkerInfo> selectedCandidates;
    vector<pair<string,Mat>> selectedImages;
    double bestStrength=-1e9;
    for(int window : windows) {
        vector<MarkerInfo> found;
        detectRobustSingle(img,found,window,refine,radius);
        double strength=stats.features;
        for(const auto& marker : found)
            strength=std::max(strength,100.0*marker.supportColumns-marker.decodeError);
        if(strength>bestStrength) {
            bestStrength=strength; selected=stats;
            selectedCandidates=candidateMarkers; selectedImages=stageImages;
        }
        hypotheses.insert(hypotheses.end(),found.begin(),found.end());
    }
    // Keep the strongest spatial hypothesis across threshold windows, without
    // giving any preference to a particular ID (including the test strip ID).
    std::stable_sort(hypotheses.begin(),hypotheses.end(),[](const auto& a,const auto& b){
        if(a.supportColumns!=b.supportColumns) return a.supportColumns>b.supportColumns;
        return a.decodeError<b.decodeError;
    });
    auto bounds=[](const MarkerInfo& marker) {
        vector<Point2f> points;
        for(const auto& column : marker.cornerLists) points.insert(points.end(),column.begin(),column.end());
        return boundingRect(points);
    };
    result.clear();
    for(const auto& candidate : hypotheses) {
        const Rect box=bounds(candidate);
        bool duplicate=false;
        for(const auto& accepted : result) {
            const Rect other=bounds(accepted);
            const double overlap=(box&other).area();
            if(overlap>.3*std::min(box.area(),other.area())) {duplicate=true;break;}
        }
        if(!duplicate) result.push_back(candidate);
    }
    stats=selected;candidateMarkers=std::move(selectedCandidates);stageImages=std::move(selectedImages);
    if(debugCapture) {
        Mat overlay; cvtColor(img,overlay,COLOR_GRAY2BGR);
        for(const auto& marker : result) {
            const auto color=Scalar(20,220,20);
            for(const auto& column : marker.cornerLists) for(int half=0;half<2;++half) {
                vector<Point> p(column.begin()+half*4,column.begin()+half*4+4);
                polylines(overlay,p,true,color,2,LINE_AA);
            }
            const Rect box=bounds(marker);
            putText(overlay,"ID "+std::to_string(marker.markerID)+(marker.mirrored?" MIRROR":""),
                {box.x,std::max(15,box.y-7)},FONT_HERSHEY_SIMPLEX,.5,color,2,LINE_AA);
        }
        stageImages.push_back({"06_decoded",overlay});
    }
}

void CylinderTag::detectRobustSingle(const Mat& img, vector<MarkerInfo>& markers_info, int adaptiveThresh, const bool cornerSubPix, int cornerSubPixDist){
    stats = DetectionStats{};
    stats.selectedWindow=adaptiveThresh;
    stageImages.clear();
    if(debugCapture) stageImages.push_back({"01_gray",img.clone()});
    candidateMarkers.clear();
    markers_info.clear();

	//Refresh
	quadAreas_labeled.clear();
	corners.clear();
	features.clear();
	markers.clear();

	Mat img_resize, img_float;
    img.convertTo(img_resize, CV_32FC1, 1.0 / 255);

    Mat img_binary(img_resize.rows, img_resize.cols, CV_8UC1);
    detector.adaptiveThresholdRobust(img_resize, img_binary, adaptiveThresh);
    if(debugCapture) stageImages.push_back({"02_binary",img_binary.clone()});
	detector.connectedComponentLabeling(img_binary, quadAreas_labeled);
    stats.components = quadAreas_labeled.size();

    detector.contourQuads(img_binary, corners);
    if (cornerSubPix) detector.refineQuads(img_resize, corners, cornerSubPixDist);
    stats.quads = corners.size();
    if(debugCapture) {
        Mat overlay; cvtColor(img,overlay,COLOR_GRAY2BGR);
        for(const auto& quad : corners) {
            vector<Point> points(quad.begin(),quad.end());
            polylines(overlay,points,true,Scalar(20,220,20),1,LINE_AA);
        }
        stageImages.push_back({"03_quads",overlay});
    }
	if (corners.empty()) {
		return;
	}

	detector.featureRecoveryRobust(corners, features);
    for (auto& feature : features) {
        auto& p=feature.corners;
        const Point2f longitudinal=(p[0]+p[1]-p[4]-p[5])/2;
        const bool vertical=std::abs(longitudinal.y)>=std::abs(longitudinal.x);
        const auto along=[&](Point2f q){ return vertical?q.y:q.x; };
        const auto across=[&](Point2f q){ return vertical?q.x:q.y; };
        // Contour enumeration is not a top-to-bottom scan. Canonicalize the
        // two halves before decoding instead of depending on component order.
        if(along(p[0]+p[1])>along(p[4]+p[5]))
            for(int k=0;k<4;++k) std::swap(p[k],p[k+4]);
        if(across(p[0])>across(p[1])) {
            std::swap(p[0],p[1]); std::swap(p[2],p[3]);
            std::swap(p[4],p[5]); std::swap(p[6],p[7]);
        }
        feature.feature_center=(p[0]+p[1]+p[4]+p[5])/4;
        feature.feature_angle=std::atan2((p[0]+p[1]-p[4]-p[5]).y,
                                        (p[0]+p[1]-p[4]-p[5]).x)*180/CV_PI;
    }
    stats.features = features.size();
    if(debugCapture) {
        Mat overlay; cvtColor(img,overlay,COLOR_GRAY2BGR);
        for(size_t i=0;i<features.size();++i) {
            const auto& f=features[i];
            for(int half=0;half<2;++half) {
                vector<Point> points(f.corners.begin()+half*4,f.corners.begin()+half*4+4);
                polylines(overlay,points,true,half?Scalar(0,220,255):Scalar(255,220,0),2,LINE_AA);
            }
            putText(overlay,std::to_string(i),f.feature_center,FONT_HERSHEY_SIMPLEX,.4,Scalar(0,0,255),1,LINE_AA);
        }
        stageImages.push_back({"04_pairs",overlay});
    }
	if (features.size() < this->featureSize) {
		return;
	}


	detector.markerOrganizationRobust(features, markers);
    candidateMarkers = markers;
    stats.groups = markers.size();
    if(debugCapture) {
        Mat overlay; cvtColor(img,overlay,COLOR_GRAY2BGR);
        for(size_t i=0;i<markers.size();++i) {
            const auto color=Scalar(70+(i*73)%180,70+(i*47)%180,70+(i*113)%180);
            for(size_t k=0;k<markers[i].cornerLists.size();++k) {
                const auto& p=markers[i].cornerLists[k];
                for(int half=0;half<2;++half) {
                    vector<Point> points(p.begin()+half*4,p.begin()+half*4+4);
                    polylines(overlay,points,true,color,2,LINE_AA);
                }
                putText(overlay,std::to_string(markers[i].feature_ID[k]),markers[i].feature_center[k],FONT_HERSHEY_SIMPLEX,.4,Scalar(0,0,255),1,LINE_AA);
            }
        }
        stageImages.push_back({"05_groups",overlay});
    }
    for (const auto& candidate : markers)
        for (int code : candidate.feature_ID) stats.validColumns += code >= 0;

	detector.markerDecoderRobust(markers, markers, this->state, this->featureSize);
	markers_info = markers;

}

void corner_detector::adaptiveThresholdRobust(const Mat& src, Mat& dst, int thresholdWindow){
    int col_number = src.cols / thresholdWindow + (src.cols % thresholdWindow != 0 ? 1 : 0);
    int row_number = src.rows / thresholdWindow + (src.rows % thresholdWindow != 0 ? 1 : 0);
    Mat extreme_min(row_number, col_number, CV_32FC1);
    Mat extreme_max(row_number, col_number, CV_32FC1);
    Mat extreme_min_final(row_number, col_number, CV_32FC1);
    Mat extreme_max_final(row_number, col_number, CV_32FC1);

    RAC = 0;
    row_count = 0;
    col_count = 0;
    row_count_final = 1;
    col_count_final = 1;

    for (int i = 0; i < src.rows; i += thresholdWindow){
        for (int j = 0; j < src.cols; j += thresholdWindow){
            Rect rect(j, i, min(thresholdWindow, src.cols - j), min(thresholdWindow, src.rows - i));
            img_part = src(rect);
            minMaxLoc(img_part, &minVal, &maxVal, NULL, NULL);
            extreme_min.at<float>(row_count, col_count) = (float)minVal;
            extreme_max.at<float>(row_count, col_count) = (float)maxVal;
            col_count++;
        }
        col_count = 0;
        row_count++;
    }
    // Initialize border tiles too. The former interior-only loop read
    // uninitialized thresholds at all four image boundaries.
    erode(extreme_min, extreme_min_final, Mat::ones(3,3,CV_8UC1));
    dilate(extreme_max, extreme_max_final, Mat::ones(3,3,CV_8UC1));

    for (int i = 0; i < src.rows; i++){
        for (int j = 0; j < src.cols; j++){
            const float lo=extreme_min_final.at<float>(i/thresholdWindow,j/thresholdWindow);
            const float hi=extreme_max_final.at<float>(i/thresholdWindow,j/thresholdWindow);
            if (hi-lo >= .05f && src.at<float>(i,j) < (hi+lo)/2){
                dst.at<uchar>(i, j) = 255;
            }
            else{
                dst.at<uchar>(i, j) = 0;
            }
        }
    }
}

void corner_detector::refineQuads(const Mat& gray, vector<vector<Point2f>>& quads, int searchRadius) {
    auto sample=[&](Point2f p) -> float {
        if(p.x<0||p.y<0||p.x>=gray.cols-1||p.y>=gray.rows-1) return NAN;
        const int x=int(p.x),y=int(p.y);
        const float a=p.x-x,b=p.y-y;
        return (1-b)*((1-a)*gray.at<float>(y,x)+a*gray.at<float>(y,x+1))+
                b*((1-a)*gray.at<float>(y+1,x)+a*gray.at<float>(y+1,x+1));
    };
    for(auto& quad : quads) {
        array<Vec4f,4> lines;
        bool valid=true;
        float shortest=1e9;
        for(int i=0;i<4;++i) shortest=std::min(shortest,float(norm(quad[i]-quad[(i+1)%4])));
        const float range=std::min(float(searchRadius),std::max(1.f,shortest*.25f));
        for(int side=0;side<4;++side) {
            const Point2f a=quad[side],b=quad[(side+1)%4],delta=b-a;
            const float length=norm(delta);
            const Point2f normal(delta.y/length,-delta.x/length);
            vector<Point2f> edgePoints;
            const int samples=std::clamp(int(length),8,80);
            for(int k=0;k<samples;++k) {
                const float t=.15f+.7f*(k+.5f)/samples;
                const Point2f center=a+t*delta;
                float best=-1,offset=0;
                for(float shift=-range;shift<=range;shift+=.25f) {
                    const float gradient=sample(center+(shift+.65f)*normal)-sample(center+(shift-.65f)*normal);
                    if(std::isfinite(gradient)&&gradient>best) {best=gradient;offset=shift;}
                }
                if(best<.06f) continue;
                float weighted=0,weight=0;
                for(float shift=offset-.5f;shift<=offset+.5f;shift+=.125f) {
                    const float gradient=sample(center+(shift+.65f)*normal)-sample(center+(shift-.65f)*normal);
                    if(std::isfinite(gradient)&&gradient>0) {weighted+=shift*gradient;weight+=gradient;}
                }
                if(weight>0) edgePoints.push_back(center+normal*(weighted/weight));
            }
            if(edgePoints.size()<4) {valid=false;break;}
            fitLine(edgePoints,lines[side],DIST_WELSCH,0,.01,.01);
        }
        if(!valid) continue;
        auto next=quad;
        for(int k=0;k<4;++k) {
            const auto& a=lines[(k+3)%4];const auto& b=lines[k];
            const double determinant=a[0]*b[1]-a[1]*b[0];
            if(abs(determinant)<.05) {valid=false;break;}
            const double t=((b[2]-a[2])*b[1]-(b[3]-a[3])*b[0])/determinant;
            next[k]={float(a[2]+t*a[0]),float(a[3]+t*a[1])};
            if(!std::isfinite(next[k].x)||!std::isfinite(next[k].y)||norm(next[k]-quad[k])>range*2) {valid=false;break;}
        }
        if(valid&&isContourConvex(next)) quad=next;
    }
}

void corner_detector::contourQuads(const Mat& binary, vector<vector<Point2f>>& quads) {
    vector<vector<Point>> contours;
    findContours(binary,contours,RETR_EXTERNAL,CHAIN_APPROX_NONE);
    quads.clear();
    for(const auto& contour : contours) {
        const double area=abs(contourArea(contour));
        if(area<20||area>binary.total()*.04) continue;
        const double perimeter=arcLength(contour,true);
        vector<Point> poly;
        for(double accuracy : {.015,.025,.04}) {
            approxPolyDP(contour,poly,accuracy*perimeter,true);
            if(poly.size()==4) break;
        }
        if(poly.size()!=4||!isContourConvex(poly)) continue;
        bool valid=true;
        for(int k=0;k<4;++k) {
            valid &= norm(poly[k]-poly[(k+1)%4])>=3;
            valid &= poly[k].x>1&&poly[k].y>1&&poly[k].x<binary.cols-2&&poly[k].y<binary.rows-2;
        }
        if(!valid) continue;
        vector<Point2f> points(poly.begin(),poly.end());
        // Pairing expects the same winding as the legacy edge extractor.
        if(contourArea(points,true)<0) reverse(points.begin(),points.end());
        quads.push_back(points);
    }
}

void corner_detector::featureRecoveryRobust(vector<vector<Point2f>>& corners_refined, vector<featureInfo>& features){
    vector<bool> isVisited(corners_refined.size(), false);
    struct PairCandidate {int first,second;float score;featureInfo feature;};
    vector<PairCandidate> candidates;

    corner_dist.resize(corners_refined.size());
    corner_centers.clear();
    corner_angles_1.clear();
    corner_angles_2.clear();

    for (int i = 0; i < corners_refined.size(); i++){
        Point2f center_ = Point2f((corners_refined[i][0].x + corners_refined[i][1].x + corners_refined[i][2].x + corners_refined[i][3].x) / 4, (corners_refined[i][0].y + corners_refined[i][1].y + corners_refined[i][2].y + corners_refined[i][3].y) / 4);
        corner_centers.push_back(center_);
        for (int j = 0; j < 4; j++){
            corner_dist[i][j] = sqrt((corners_refined[i][j].x - corners_refined[i][(j + 1) % 4].x) * (corners_refined[i][j].x - corners_refined[i][(j + 1) % 4].x) + (corners_refined[i][j].y - corners_refined[i][(j + 1) % 4].y) * (corners_refined[i][j].y - corners_refined[i][(j + 1) % 4].y));
        }
        corner_angles_1.push_back((atan2(corners_refined[i][0].y - corners_refined[i][1].y, corners_refined[i][0].x - corners_refined[i][1].x) * 180 / CV_PI + atan2(corners_refined[i][3].y - corners_refined[i][2].y, corners_refined[i][3].x - corners_refined[i][2].x) * 180 / CV_PI) / 2);
        corner_angles_2.push_back((atan2(corners_refined[i][1].y - corners_refined[i][2].y, corners_refined[i][1].x - corners_refined[i][2].x) * 180 / CV_PI + atan2(corners_refined[i][0].y - corners_refined[i][3].y, corners_refined[i][0].x - corners_refined[i][3].x) * 180 / CV_PI) / 2);
    }
    for (int i = 0; i < corners_refined.size() - 1; i++){
        if (isVisited[i]) continue;
        for (int j = i + 1; j < corners_refined.size(); j++) {
            if (!isVisited[j]){
                tag1 = false;
                tag2 = false;
                feature_angle = atan2(corner_centers[i].y - corner_centers[j].y, corner_centers[i].x - corner_centers[j].x) * 180 / CV_PI;

                if (abs(feature_angle - corner_angles_1[i]) < threshold_angle || abs(abs(feature_angle - corner_angles_1[i]) - 180) < threshold_angle || abs(abs(feature_angle - corner_angles_1[i]) - 360) < threshold_angle){
                    tag1 = true;
                    dist1_long = (corner_dist[i][0] + corner_dist[i][2]) / 2;
                    dist1_short = min(corner_dist[i][1], corner_dist[i][3]);
                    if (corner_dist[i][1] < corner_dist[i][3]) {
                        edge_angle1 = atan2(corners_refined[i][0].y - corners_refined[i][3].y, corners_refined[i][0].x - corners_refined[i][3].x) * 180 / CV_PI;
                        feature_point1 = (corners_refined[i][1] + corners_refined[i][2]) / 2;
                    }
                    else {
                        edge_angle1 = atan2(corners_refined[i][1].y - corners_refined[i][2].y, corners_refined[i][1].x - corners_refined[i][2].x) * 180 / CV_PI;
                        feature_point1 = (corners_refined[i][0] + corners_refined[i][3]) / 2;
                    }
                }
                if (abs(feature_angle - corner_angles_2[i]) < threshold_angle || abs(abs(feature_angle - corner_angles_2[i]) - 180) < threshold_angle || abs(abs(feature_angle - corner_angles_2[i]) - 360) < threshold_angle){
                    tag1 = true;
                    dist1_short = min(corner_dist[i][0], corner_dist[i][2]);
                    dist1_long = (corner_dist[i][1] + corner_dist[i][3]) / 2;
                    if (corner_dist[i][0] > corner_dist[i][2]) {
                        edge_angle1 = atan2(corners_refined[i][0].y - corners_refined[i][1].y, corners_refined[i][0].x - corners_refined[i][1].x) * 180 / CV_PI;
                        feature_point1 = (corners_refined[i][2] + corners_refined[i][3]) / 2;
                    }
                    else {
                        edge_angle1 = atan2(corners_refined[i][2].y - corners_refined[i][3].y, corners_refined[i][2].x - corners_refined[i][3].x) * 180 / CV_PI;
                        feature_point1 = (corners_refined[i][0] + corners_refined[i][1]) / 2;
                    }
                }
                if (abs(feature_angle - corner_angles_1[j]) < threshold_angle || abs(abs(feature_angle - corner_angles_1[j]) - 180) < threshold_angle || abs(abs(feature_angle - corner_angles_1[j]) - 360) < threshold_angle){
                    tag2 = true;
                    dist2_long = (corner_dist[j][0] + corner_dist[j][2]) / 2;
                    dist2_short = min(corner_dist[j][1], corner_dist[j][3]);
                    if (corner_dist[j][1] < corner_dist[j][3]) {
                        edge_angle2 = atan2(corners_refined[j][0].y - corners_refined[j][3].y, corners_refined[j][0].x - corners_refined[j][3].x) * 180 / CV_PI;
                        feature_point2 = (corners_refined[j][1] + corners_refined[j][2]) / 2;
                    }
                    else {
                        edge_angle2 = atan2(corners_refined[j][1].y - corners_refined[j][2].y, corners_refined[j][1].x - corners_refined[j][2].x) * 180 / CV_PI;
                        feature_point2 = (corners_refined[j][0] + corners_refined[j][3]) / 2;
                    }
                }
                if (abs(feature_angle - corner_angles_2[j]) < threshold_angle || abs(abs(feature_angle - corner_angles_2[j]) - 180) < threshold_angle || abs(abs(feature_angle - corner_angles_2[j]) - 360) < threshold_angle){
                    tag2 = true;
                    dist2_short = min(corner_dist[j][0], corner_dist[j][2]);
                    dist2_long = (corner_dist[j][1] + corner_dist[j][3]) / 2;
                    if (corner_dist[j][0] > corner_dist[j][2]) {
                        edge_angle2 = atan2(corners_refined[j][0].y - corners_refined[j][1].y, corners_refined[j][0].x - corners_refined[j][1].x) * 180 / CV_PI;
                        feature_point2 = (corners_refined[j][2] + corners_refined[j][3]) / 2;
                    }
                    else {
                        edge_angle2 = atan2(corners_refined[j][2].y - corners_refined[j][3].y, corners_refined[j][2].x - corners_refined[j][3].x) * 180 / CV_PI;
                        feature_point2 = (corners_refined[j][0] + corners_refined[j][1]) / 2;
                    }
                }
                feature_length = robustDistance(corner_centers[i], corner_centers[j]);
                if ((tag1 && tag2) && (dist1_long > dist1_short || dist2_long > dist2_short)
                    && (abs(edge_angle1 - edge_angle2) < threshold_angle * 10 || abs(abs(edge_angle1 - edge_angle2) - 180) < threshold_angle * 10 || abs(abs(edge_angle1 - edge_angle2) - 360) < threshold_angle * 10)
                    && (abs(dist1_short - dist2_short) < min(dist1_short, dist2_short) * 0.33)
                    && ((dist1_long + dist2_long) > (dist1_short + dist2_short))
                    && ((dist1_long + dist2_long) < 15 * (dist1_short + dist2_short))
                    && (feature_length - (dist1_long + dist2_long) / 2 < 0.3 * (feature_length + (dist1_long + dist2_long) / 2)))
                {
                    const auto feature=featureOrganization(corners_refined[i], corners_refined[j], corner_centers[i], corner_centers[j], feature_angle);
                    const auto& p=feature.corners;
                    const float width=(norm(p[0]-p[1])+norm(p[4]-p[5]))/2;
                    float rms=0;
                    for(const vector<int>& indices : {vector<int>{0,3,6,5},vector<int>{1,2,7,4}}) {
                        vector<Point2f> points;
                        for(int k:indices) points.push_back(p[k]);
                        Vec4f line; fitLine(points,line,DIST_L2,0,.01,.01);
                        for(const auto& q:points) {
                            const float d=(q.x-line[2])*line[1]-(q.y-line[3])*line[0];
                            rms+=d*d;
                        }
                    }
                    rms=std::sqrt(rms/8);
                    // The corresponding long edges lie on cylinder
                    // generatrices. Neighboring bars fail this collinearity
                    // check even when their centroids happen to align.
                    if(rms<=std::max(.8f,.1f*width))
                        candidates.push_back({i,j,rms/std::max(1.f,width),feature});
                }
            }
        }
    }

    std::stable_sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b){return a.score<b.score;});
    for(const auto& candidate:candidates) {
        if(isVisited[candidate.first]||isVisited[candidate.second]) continue;
        isVisited[candidate.first]=isVisited[candidate.second]=true;
        features.push_back(candidate.feature);
    }

}

void corner_detector::markerOrganizationRobust(vector<featureInfo> feature, vector<MarkerInfo>& markers){
    markers.clear();
    if (feature.empty()) return;
    vector<int> father(feature.size());
    auto findRoot = [&](int index) {
        int root = index;
        while (father[root] != root) root = father[root];
        while (father[index] != index) {
            const int next = father[index];
            father[index] = root;
            index = next;
        }
        return root;
    };
    for (int i = 0; i < feature.size(); i++){
        father[i] = i;
    }
    for (int i = 0; i < feature.size() - 1; i++){
        for (int j = i + 1; j < feature.size(); j++){
            vector_center = feature[i].feature_center - feature[j].feature_center;
            vector_longedge = feature[i].corners[0] - feature[i].corners[5];
            center_angle = (vector_center.x * vector_longedge.x + vector_center.y * vector_longedge.y) / sqrt((vector_center.x * vector_center.x + vector_center.y * vector_center.y) * (vector_longedge.x * vector_longedge.x + vector_longedge.y * vector_longedge.y));
            if ((abs(feature[i].feature_angle - feature[j].feature_angle) < threshold_angle * 2 || abs(180 - abs(feature[i].feature_angle - feature[j].feature_angle)) < threshold_angle) && (robustDistance(feature[i].feature_center, feature[j].feature_center) < 0.3 * robustDistance(feature[i].corners[0], feature[i].corners[5])) && (abs(center_angle) < threshold_vertical)){
                int father_i = findRoot(i), father_j = findRoot(j);
                if (father_i != father_j)
                    father[father_j] = father_i;
            }
        }
    }

    father_database.clear();
    cnt = 0;
    vector<vector<int>> marker_ID(feature.size());
    father_database.push_back(father[0]);
    marker_ID[cnt++].push_back(0);

    for (int i = 1; i < feature.size(); i++) {
        int father_now = father[i];
        while (father_now != father[father_now])
            father_now = findRoot(father_now);
        father[i] = father_now;
    }

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
        marker_angle = 0;
        for (int j = 0; j < marker_ID[i].size(); j++){
            mark.cornerLists.push_back(feature[marker_ID[i][j]].corners);
            mark.feature_center.push_back(feature[marker_ID[i][j]].feature_center);
            mark.edge_length.push_back((robustDistance(feature[marker_ID[i][j]].corners[0], feature[marker_ID[i][j]].corners[1]) + robustDistance(feature[marker_ID[i][j]].corners[4], feature[marker_ID[i][j]].corners[5])) / 2);
            double angle_now = fastAtan2(mark.cornerLists[j][0].y - mark.cornerLists[j][5].y, mark.cornerLists[j][0].x - mark.cornerLists[j][5].x);
            if (angle_now > 180) angle_now -= 180;
            marker_angle += angle_now;
        }
        marker_angle /= mark.cornerLists.size();

        if (abs(marker_angle) < 45 || abs(marker_angle) > 135) {
            auto p = sort_permutation(mark.feature_center,
                [](const auto& a, const auto& b) { return a.y > b.y; });
            mark.feature_center = apply_permutation(mark.feature_center, p);
            mark.cornerLists = apply_permutation(mark.cornerLists, p);
            mark.edge_length = apply_permutation(mark.edge_length, p);
            featureExtractionRobust(mark, mark, 0);
        }
        else {
            auto p = sort_permutation(mark.feature_center,
                [](const auto& a, const auto& b) { return a.x < b.x; });
            mark.feature_center = apply_permutation(mark.feature_center, p);
            mark.cornerLists = apply_permutation(mark.cornerLists, p);
            mark.edge_length = apply_permutation(mark.edge_length, p);
            featureExtractionRobust(mark, mark, 1);
        }
        markers.push_back(mark);
    }
}

void corner_detector::featureExtractionRobust(MarkerInfo& marker_src, MarkerInfo& marker_dst, int direction) {
    marker_dst = marker_src;
    marker_dst.feature_ID.clear();
    marker_dst.feature_ID_left.clear();
    marker_dst.feature_ID_right.clear();
    marker_dst.cr_left.clear();
    marker_dst.cr_right.clear();
    for (auto& points : marker_dst.cornerLists) {
        // Preserve the public corner ordering expected by the pose estimator.
        if (!direction && points[0].x > points[4].x) {
            for (int k=0; k<4; ++k) swap(points[k], points[k+4]);
        }
        const float gapLeft = norm(points[3]-points[6]);
        const float gapRight = norm(points[2]-points[7]);
        const float heightLeft = norm(points[0]-points[5]);
        const float heightRight = norm(points[1]-points[4]);
        const float crLeft = (norm(points[0]-points[6])*norm(points[3]-points[5])) /
                             std::max(1e-6f,gapLeft*heightLeft);
        const float crRight = (norm(points[1]-points[7])*norm(points[2]-points[4])) /
                              std::max(1e-6f,gapRight*heightRight);
        marker_dst.cr_left.push_back(crLeft);
        marker_dst.cr_right.push_back(crRight);
        // Rectification avoids explicitly intersecting nearly parallel lines
        // to infer which half contains the white gap (the old right-hand side
        // also incorrectly used the LEFT midpoint).
        const vector<Point2f> outside = {points[0],points[1],points[4],points[5]};
        const vector<Point2f> square = {{0,0},{1,0},{1,1},{0,1}};
        const Mat homography = getPerspectiveTransform(outside,square);
        vector<Point2f> inside;
        perspectiveTransform(vector<Point2f>{points[3],points[6],points[2],points[7]},inside,homography);
        int left=-1, right=-1;
        auto symbol = [&](float cr, bool lower) {
            if (!std::isfinite(cr) || cr<1.30f || cr>1.85f) return -1;
            int nearest=0;
            for (int j=1;j<4;++j)
                if(abs(cr-ID_cr_correspond[j])<abs(cr-ID_cr_correspond[nearest])) nearest=j;
            return lower ? 7-nearest : nearest;
        };
        // Pixel uncertainty matters for a 10 px gap; a pure percentage gate
        // rejected legitimate columns after quantization of the two edges.
        const bool geometry = gapLeft>2 && gapRight>2 &&
            abs(gapLeft-gapRight) <= 2.0f + .10f*(gapLeft+gapRight) &&
            inside.size()==4 && std::all_of(inside.begin(),inside.end(),[](Point2f q){
                return std::isfinite(q.x)&&std::isfinite(q.y)&&q.y>-.05&&q.y<1.05;
            });
        if (geometry) {
            left=symbol(crLeft,inside[0].y+inside[1].y>1);
            right=symbol(crRight,inside[2].y+inside[3].y>1);
        }
        marker_dst.feature_ID_left.push_back(left);
        marker_dst.feature_ID_right.push_back(right);
        marker_dst.feature_ID.push_back(left>=0&&right>=0 ? left*8+right : -2);
    }
}

void corner_detector::markerDecoderRobust(vector<MarkerInfo> markers_src, vector<MarkerInfo>& markers_dst, Mat1i& state, int featureSize){
    markers_dst.clear();
    struct Hypothesis {
        double error;
        int id, start, direction, support;
        bool mirror;
    };
    constexpr float crValues[8]={1.47f,1.54f,1.61f,1.68f,1.68f,1.61f,1.54f,1.47f};
    for (const auto& group : markers_src) {
        const int count=group.cornerLists.size();
        // A single matched column is not an ID. Require multiple independent
        // columns, and compare against EVERY ID, phase and orientation.
        if(count<std::max(4,featureSize)) continue;
        vector<int> offsets(count,0);
        bool validSpacing=true;
        for(int k=1;k<count;++k) {
            const float pitch=.75f*(group.edge_length[k]+group.edge_length[k-1]);
            const float ratio=norm(group.feature_center[k]-group.feature_center[k-1])/std::max(pitch,1e-6f);
            const int step=std::max(1,int(std::round(ratio)));
            if(!std::isfinite(ratio)||ratio<.35f||step>3) { validSpacing=false; break; }
            offsets[k]=offsets[k-1]+step;
        }
        if(!validSpacing||offsets.back()>=state.cols) continue;
        vector<Hypothesis> hypotheses;
        for(int id=0;id<state.rows;++id) for(int phase=0;phase<state.cols;++phase)
        for(int direction : {1,-1}) for(bool mirror : {false,true}) {
            vector<double> residuals;
            int support=0;
            for(int k=0;k<count;++k) {
                const int left=group.feature_ID_left[k],right=group.feature_ID_right[k];
                if(left<0||right<0) continue;
                const int pos=(phase+direction*offsets[k]+state.cols)%state.cols;
                int code=state(id,pos);
                if(mirror) code=63-code;
                if(direction<0) code=(7-code%8)*8+(7-code/8);
                const int a=code/8,b=code%8;
                const double dl=group.cr_left[k]-crValues[a];
                const double dr=group.cr_right[k]-crValues[b];
                const bool sameHalf=(a>=4)==(left>=4)&&(b>=4)==(right>=4);
                const double cost=(dl*dl+dr*dr)/2 + (sameHalf ? 0.0 : .25);
                residuals.push_back(cost);
                support += sameHalf && abs(dl)<.09 && abs(dr)<.09;
            }
            if(residuals.size()<4) continue;
            std::sort(residuals.begin(),residuals.end());
            // Bound one damaged column's influence, but retain a penalty.
            // Dropping it entirely makes a wrong 4-of-5 hypothesis tie an
            // exact 5-of-5 match, discarding useful dictionary evidence.
            if(residuals.size()>=5) residuals.back()=std::min(.01,residuals.back())*.25;
            const double error=std::sqrt(std::accumulate(residuals.begin(),residuals.end(),0.0)/residuals.size());
            hypotheses.push_back({error,id,phase,direction,support,mirror});
        }
        if(hypotheses.size()<2) continue;
        std::sort(hypotheses.begin(),hypotheses.end(),[](const auto&a,const auto&b){return a.error<b.error;});
        const auto& best=hypotheses[0];
        const double margin=hypotheses[1].error-best.error;
        if(best.support<std::max(4,featureSize)||best.error>.040||margin<.012) continue;
        MarkerInfo result=group;
        result.markerID=best.id;
        result.mirrored=best.mirror;
        result.decodeError=best.error;
        result.decodeMargin=margin;
        result.supportColumns=best.support;
        result.featurePos.clear();
        for(int k=0;k<count;++k) {
            result.featurePos.push_back((best.start+best.direction*offsets[k]+state.cols)%state.cols);
            // Reflection flags allow the corner order to be mapped back to
            // canonical model coordinates without inventing a camera pose.
            if(best.mirror) {
                const auto previous=result.cornerLists[k];
                const int reflection[8]={5,4,7,6,1,0,3,2};
                for(int j=0;j<8;++j) result.cornerLists[k][j]=previous[reflection[j]];
            }
            if(best.direction<0) {
                for(int j=0;j<4;++j) swap(result.cornerLists[k][j],result.cornerLists[k][j+4]);
            }
        }
        markers_dst.push_back(result);
    }
}
