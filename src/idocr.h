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
    bool static getIntersectionPoint(cv::Point a1, cv::Point a2, cv::Point b1, cv::Point b2, cv::Point2f &intPnt);
    void static process(cv::Mat & image);
    double static cross(cv::Point v1,cv::Point v2);
};

#endif // IDOCR_H