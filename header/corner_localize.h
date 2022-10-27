#pragma once
#ifndef CORNER_LOCALIZE_H
#define CORNER_LOCALIZE_H

#include "config.h"

using namespace std;
using namespace cv;

struct ModelInfo{
    int MarkerID;
    vector<array<Point2f,8>> corners;
};

class PoseEstimator{
public:

private:

};

#endif