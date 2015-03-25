#ifndef WEBCAM_H
#define WEBCAM_H

#include <vector>
#include <cmath>
#include <pthread.h>
#include "libfreenect.hpp"
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "mainwindow.h"

using namespace cv;

class Webcam
{
public:
    Webcam();
    ~Webcam();
    void showRGB();
    void threshold(Mat & src, Mat & dst);
    void contours(Mat & src, Mat & dst);
};

#endif // WEBCAM_H
