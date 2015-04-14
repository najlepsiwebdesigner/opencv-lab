#ifndef WEBCAM_H
#define WEBCAM_H

#include <vector>
#include <cmath>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "imageoperations.h"
#include "myfreenectdevice.h"

using namespace cv;

class Webcam
{
public:
    Webcam();
    ~Webcam();
    void showRGB();
    void showKinectRGB();
};

#endif // WEBCAM_H
