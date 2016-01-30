#ifndef IDOCR_H
#define IDOCR_H

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
//#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/nonfree/features2d.hpp"
#include <iostream>

#include <boost/filesystem.hpp>

class idOCR
{
public:
    idOCR();
    ~idOCR();

    std::string static cutoutPath;

    void static fitImage(const cv::Mat & src, cv::Mat & dst, float destWidth, float destHeight);
    bool static getIntersectionPoint(cv::Point a1, cv::Point a2, cv::Point b1, cv::Point b2, cv::Point2f &intPnt);
    void static sortCorners(std::vector<cv::Point2f>& corners,cv::Point2f center);
    void static process(cv::Mat & image);
    double static cross(cv::Point v1,cv::Point v2);
    void static saveImage(std::string fileName, const cv::Mat & image);
    std::vector<cv::Rect> static getRectanglesFromMask(cv::Mat & mask);

    void static maskCutOut(cv::Mat & image, std::string);
    void static setCutoutPath(std::string path);
    std::string static getCutoutPath();
    void static processField(cv::Mat & image);


};

#endif // IDOCR_H
