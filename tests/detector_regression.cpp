#include "CylinderTag.h"
#include <cmath>
#include <stdexcept>

namespace {
void require(bool condition, const string& message) {
    if (!condition) throw std::runtime_error(message);
}

// Build a column from its analytic cross ratios, then apply a nondegenerate
// homography so the original vanishing-point construction is well defined.
vector<Point2f> column(float leftCR, float rightCR, bool lower = false) {
    auto center = [lower](float cr) {
        return 5.f * (1.f + (lower ? 1.f : -1.f) * std::sqrt(1.f + 4.f * (.11f - .2f * cr)));
    };
    const float l = center(leftCR), r = center(rightCR);
    const vector<Point2f> points = {
        {0, 0}, {1, 0}, {1, r - 1}, {0, l - 1},
        {1, 10}, {0, 10}, {0, l + 1}, {1, r + 1}
    };
    const vector<Point2f> source = {{0, 0}, {1, 0}, {1, 10}, {0, 10}};
    const vector<Point2f> target = {{100, 50}, {110, 53}, {112, 160}, {99, 150}};
    vector<Point2f> result;
    perspectiveTransform(points, result, getPerspectiveTransform(source, target));
    return result;
}

void thresholdBorders() {
    corner_detector detector;
    // Partial tiles and images smaller than the window exercise every border.
    for (const Size size : {Size(17, 13), Size(3, 2), Size(1, 1)}) {
        const int window = 5;
        Mat input(size, CV_32FC1), actual(size, CV_8UC1);
        for (int y = 0; y < input.rows; ++y)
            for (int x = 0; x < input.cols; ++x)
                input.at<float>(y, x) = (x + y) % 2 ? .8f : .1f;
        detector.adaptiveThreshold(input, actual, window);
        for (int y = 0; y < input.rows; ++y) {
            for (int x = 0; x < input.cols; ++x) {
                const int tileX = x / window, tileY = y / window;
                const int x0 = std::max(0, tileX - 1) * window;
                const int y0 = std::max(0, tileY - 1) * window;
                const int x1 = std::min(input.cols, (tileX + 2) * window);
                const int y1 = std::min(input.rows, (tileY + 2) * window);
                double lo, hi;
                minMaxLoc(input(Rect(x0, y0, x1 - x0, y1 - y0)), &lo, &hi);
                const uchar expected = input.at<float>(y, x) < std::min(.3f, float((lo + hi) / 2)) ? 255 : 0;
                require(actual.at<uchar>(y, x) == expected, "threshold differs from clipped tile neighborhood at image border");
            }
        }
    }
}

void rightMidpoint() {
    // A projected valid code 52 (left=6, right=4). Using the left midpoint for
    // the right side incorrectly assigns its upper-half counterpart, code 51.
    MarkerInfo input, output;
    input.cornerLists.push_back({
        {100.f, 100.f}, {210.433762f, 247.973984f},
        {220.092163f, 336.049957f}, {113.056648f, 195.594452f},
        {227.479477f, 403.415680f}, {120.458641f, 249.788239f},
        {117.066109f, 224.949738f}, {223.449875f, 366.669312f}
    });
    corner_detector detector;
    detector.featureExtraction(input, output, 1);
    require(output.feature_ID.size() == 1 && output.feature_ID[0] == 52,
            "right-side half selection must use the right-side projective midpoint");
}

void invalidCodes() {
    MarkerInfo input, output;
    input.cornerLists = {column(1.54f, 1.54f), column(1.10f, 1.10f),
                        column(1.54f, 1.10f), column(1.10f, 1.54f)};
    corner_detector detector;
    detector.featureExtraction(input, output, 1);
    require(output.feature_ID.size() == 4 && output.feature_ID[0] >= 0, "valid seed column");
    for (size_t i = 1; i < output.feature_ID.size(); ++i)
        require(output.feature_ID[i] == -2, "unmatched cross ratios must not inherit a previous column's code");
    require(output.feature_ID_left[1] == -1 && output.feature_ID_right[1] == -1,
            "unmatched sides must be initialized separately for each column");
}

void columnWidth() {
    featureInfo feature;
    feature.corners = {{0, 0}, {8, 0}, {9, 30}, {0, 30},
                       {12, 100}, {0, 100}, {0, 50}, {10, 50}};
    feature.feature_center = {5, 50};
    feature.feature_angle = 90;
    corner_detector detector;
    vector<MarkerInfo> groups;
    detector.markerOrganization({feature}, groups);
    require(groups.size() == 1 && groups[0].edge_length.size() == 1, "one organized column");
    require(std::abs(groups[0].edge_length[0] - 10.f) < 1e-5f,
            "column width must be (8 + 12) / 2, not 8 + 12 / 2");
}

void staleDetections(const char* dictionary) {
    CylinderTag detector(dictionary);
    vector<MarkerInfo> markers(1);
    markers[0].markerID = 17;
    detector.detect(Mat(32, 32, CV_8UC1, Scalar(255)), markers);
    require(markers.empty(), "early detection failure must clear results from the previous frame");
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 3, "expected test name and dictionary path");
        cv::setNumThreads(1);
        const string name(argv[1]);
        if (name == "threshold_borders") thresholdBorders();
        else if (name == "right_midpoint") rightMidpoint();
        else if (name == "invalid_codes") invalidCodes();
        else if (name == "column_width") columnWidth();
        else if (name == "stale_detections") staleDetections(argv[2]);
        else throw std::runtime_error("unknown test case");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
