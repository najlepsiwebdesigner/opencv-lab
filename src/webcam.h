#ifndef WEBCAM_H
#define WEBCAM_H

#include <vector>
#include <cmath>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "imageoperations.h"
#include "myfreenectdevice.h"

#include "boost/date_time/posix_time/posix_time.hpp"
#include "boost/date_time/gregorian/greg_month.hpp"

using namespace cv;

class Webcam
{
public:

    Webcam();
    ~Webcam();
    void showRGB();
    void showKinectRGB();

    void setScreenshotPath(std::string);
    std::string getScreenshotPath();

private:
    std::string screenshotPath;
};

#endif // WEBCAM_H
