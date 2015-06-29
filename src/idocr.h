#ifndef IDOCR_H
#define IDOCR_H

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <iostream>



class idOCR
{
public:
    idOCR();
    ~idOCR();

    void static fitImage(const cv::Mat & src, cv::Mat & dst, float destWidth, float destHeight);

    void static process(cv::Mat & image);
};

#endif // IDOCR_H
