#ifndef WEBCAM_H
#define WEBCAM_H

#include <vector>
#include <cmath>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "mainwindow.h"
#include "helpers.h"

using namespace cv;

class Webcam
{
public:
    Webcam();
    ~Webcam();
    void showRGB();
};

#endif // WEBCAM_H
