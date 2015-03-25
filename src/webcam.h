#ifndef WEBCAM_H
#define WEBCAM_H

#include <vector>
#include <cmath>
#include <pthread.h>
#include "libfreenect.hpp"
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"


class Webcam
{
public:
    Webcam();
    ~Webcam();
    void showRGB();
};

#endif // WEBCAM_H