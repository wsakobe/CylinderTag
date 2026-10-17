#include "header/CylinderTag.h"
#include <chrono>
#include <set>

// Usage: input dictionary output-prefix [scale=1] [threshold=5] [subpix=1] [snapshot-frames] [multi=1]
// Output coordinates always refer to the original decoded frame resolution.
int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    if (argc < 4) {
        std::cerr << "Usage: cylindertag_video input dictionary output-prefix [scale] [threshold] [subpix] [snapshot-frames] [multi]\n";
        return 2;
    }
    try {
        const double scale = argc > 4 ? std::stod(argv[4]) : 1.0;
        const int threshold = argc > 5 ? std::stoi(argv[5]) : 5;
        const bool subpix = argc > 6 ? std::stoi(argv[6]) != 0 : true;
        if (!(scale > 0) || threshold < 1) return 2;
        cv::setNumThreads(1);
        cv::VideoCapture input(argv[1]);
        if (!input.isOpened()) throw std::runtime_error("Cannot open input video");
        const double fps = input.get(cv::CAP_PROP_FPS);
        if (!(fps > 0)) throw std::runtime_error("Invalid video FPS");
        const int width = input.get(cv::CAP_PROP_FRAME_WIDTH);
        const int height = input.get(cv::CAP_PROP_FRAME_HEIGHT);
        const std::string prefix(argv[3]);
        cv::VideoWriter output(prefix + ".avi", cv::VideoWriter::fourcc('M','J','P','G'),
                               fps, cv::Size(width, height));
        std::ofstream csv(prefix + ".csv"), json(prefix + ".jsonl");
        std::ofstream stages(prefix + ".stages.jsonl");
        if (!output.isOpened() || !csv || !json) throw std::runtime_error("Cannot open outputs");
        csv << "frame,time_seconds,marker_count,id0_detected,ids,id0_columns,processing_ms,components,quads,features,groups,valid_columns,selected_window\n";
        CylinderTag detector(argv[2]);
        detector.useMultiThreshold(argc>8 ? std::stoi(argv[8])!=0 : true);
        std::set<int> snapshotFrames;
        if(argc>7) {
            std::istringstream list(argv[7]);std::string item;
            while(std::getline(list,item,',')) if(!item.empty()) snapshotFrames.insert(std::stoi(item));
        }
        cv::Mat frame, gray, working;
        int frameIndex = 0, id0Frames = 0, anyFrames = 0, otherFrames = 0;
        while (input.read(frame)) {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            if (scale != 1.0) cv::resize(gray, working, cv::Size(), scale, scale, cv::INTER_CUBIC);
            else working = gray;
            std::vector<MarkerInfo> markers;
            const auto start = std::chrono::steady_clock::now();
            detector.captureDebugImages(snapshotFrames.count(frameIndex)>0);
            detector.detectRobust(working, markers, threshold, subpix, 5);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
            for(const auto& stage : detector.debugImages()) {
                std::ostringstream name;
                name << prefix << "_frame_" << std::setw(6) << std::setfill('0') << frameIndex << '_' << stage.first << ".png";
                cv::imwrite(name.str(),stage.second);
            }
            const auto& stats = detector.detectionStats();
            stages << "{\"frame\":" << frameIndex << ",\"selected_window\":" << stats.selectedWindow << ",\"groups\":[";
            bool firstGroup = true;
            for (const auto& group : detector.detectionCandidates()) {
                if (!firstGroup) stages << ',';
                firstGroup = false;
                stages << "{\"codes\":[";
                for(size_t i=0; i<group.feature_ID.size(); ++i) {
                    if(i) stages << ',';
                    stages << group.feature_ID[i];
                }
                stages << "],\"cross_ratios\":[";
                for(size_t i=0; i<group.cr_left.size(); ++i) {
                    if(i) stages << ',';
                    // Null represents invalid geometry, keeping JSON valid.
                    stages << '[';
                    if(std::isfinite(group.cr_left[i])) stages << group.cr_left[i]; else stages << "null";
                    stages << ',';
                    if(std::isfinite(group.cr_right[i])) stages << group.cr_right[i]; else stages << "null";
                    stages << ']';
                }
                stages << "]}";
            }
            stages << "]}\n";
            // Account for integer rounding when resize dimensions are chosen.
            const double sx = static_cast<double>(width)/working.cols;
            const double sy = static_cast<double>(height)/working.rows;
            bool id0 = false, other = false;
            std::set<int> id0Columns;
            std::ostringstream ids;
            json << "{\"frame\":" << frameIndex << ",\"time_seconds\":" << frameIndex/fps << ",\"markers\":[";
            for (size_t m = 0; m < markers.size(); ++m) {
                const auto& marker = markers[m];
                const bool expected = marker.markerID == 0;
                id0 |= expected; other |= !expected;
                if (m) { ids << ';'; json << ','; }
                ids << marker.markerID;
                json << "{\"id\":" << marker.markerID
                     << ",\"mirrored\":" << (marker.mirrored ? "true" : "false")
                     << ",\"decode_error\":" << marker.decodeError
                     << ",\"decode_margin\":" << marker.decodeMargin
                     << ",\"support_columns\":" << marker.supportColumns << ",\"columns\":[";
                for (size_t c=0; c<marker.featurePos.size(); ++c) {
                    if(c) json << ',';
                    json << marker.featurePos[c];
                    if(expected) id0Columns.insert(marker.featurePos[c]);
                }
                json << "],\"corners\":[";
                const cv::Scalar color = expected ? cv::Scalar(40,230,40) : cv::Scalar(0,150,255);
                std::vector<cv::Point> points;
                for(size_t c=0; c<marker.cornerLists.size(); ++c) {
                    if(c) json << ',';
                    json << '[';
                    std::vector<cv::Point> corners;
                    for(size_t k=0; k<marker.cornerLists[c].size(); ++k) {
                        const auto p=marker.cornerLists[c][k];
                        const cv::Point2f q(p.x*sx,p.y*sy);
                        corners.push_back(q); points.push_back(q);
                        if(k) json << ',';
                        json << '[' << q.x << ',' << q.y << ']';
                        cv::circle(frame,q,2,cv::Scalar(255,220,30),-1,cv::LINE_AA);
                    }
                    json << ']';
                    if(corners.size()==8) {
                        // Separate outlines of the two black quadrilaterals.
                        for(const std::vector<int>& order : {std::vector<int>{0,1,2,3},std::vector<int>{4,5,6,7}}) {
                            std::vector<cv::Point> poly;
                            for(int k:order) poly.push_back(corners[k]);
                            cv::polylines(frame,poly,true,color,2,cv::LINE_AA);
                        }
                    }
                }
                json << "]}";
                if(!points.empty()) {
                    const auto box=cv::boundingRect(points);
                    cv::rectangle(frame,box,color,1,cv::LINE_AA);
                    const std::string label="ID "+std::to_string(marker.markerID)+" | "+std::to_string(marker.supportColumns)+" cols"+(marker.mirrored ? " | MIRROR" : "");
                    const cv::Point anchor(std::max(2,box.x),std::max(92,box.y-8));
                    cv::putText(frame,label,anchor,cv::FONT_HERSHEY_SIMPLEX,.5,cv::Scalar(0,0,0),3,cv::LINE_AA);
                    cv::putText(frame,label,anchor,cv::FONT_HERSHEY_SIMPLEX,.5,color,1,cv::LINE_AA);
                }
            }
            json << "]}\n";
            id0Frames += id0; anyFrames += !markers.empty(); otherFrames += other;
            csv << frameIndex << ',' << std::fixed << std::setprecision(6) << frameIndex/fps << ',' << markers.size()
                << ',' << id0 << ',' << ids.str() << ',' << id0Columns.size() << ',' << ms
                << ',' << stats.components << ',' << stats.quads << ',' << stats.features
                << ',' << stats.groups << ',' << stats.validColumns << ',' << stats.selectedWindow << '\n';
            cv::rectangle(frame,{0,0},{width,78},cv::Scalar(15,15,15),-1);
            const cv::Scalar status = id0 ? cv::Scalar(40,230,40) : cv::Scalar(80,180,255);
            cv::putText(frame,id0 ? "CylinderTag ID 0: DETECTED" : "CylinderTag ID 0: NOT DETECTED",
                        {10,23},cv::FONT_HERSHEY_SIMPLEX,.57,status,1,cv::LINE_AA);
            std::ostringstream info;
            info << "Frame " << frameIndex << " | " << std::fixed << std::setprecision(2) << frameIndex/fps << " s | ID0 " << id0Frames << '/' << frameIndex+1;
            cv::putText(frame,info.str(),{10,46},cv::FONT_HERSHEY_SIMPLEX,.48,cv::Scalar(230,230,230),1,cv::LINE_AA);
            cv::putText(frame,"Detection only | no calibrated pose",{10,67},cv::FONT_HERSHEY_SIMPLEX,.44,cv::Scalar(190,190,190),1,cv::LINE_AA);
            output.write(frame);
            ++frameIndex;
            if(frameIndex%50==0) std::cerr << "Processed " << frameIndex << " frames\n";
        }
        if(frameIndex==0) throw std::runtime_error("No decoded frames");
        std::cout << "SUMMARY frames=" << frameIndex << " id0=" << id0Frames << " any=" << anyFrames
                  << " other_id_frames=" << otherFrames << " fps=" << fps << '\n';
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1;
    } catch(const std::string& e) { std::cerr << e << '\n'; return 1; }
    return 0;
}
