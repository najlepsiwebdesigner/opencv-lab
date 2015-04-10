#ifndef IMAGEOPERATIONS_H
#define IMAGEOPERATIONS_H
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "mainwindow.h"
#include "helpers.h"

using namespace cv;
using namespace std;

class ImageOperations
{
public:
    ImageOperations();
    ~ImageOperations();

    void static equalize(Mat & image);
    void static lines(Mat & image);
    void static thresholdGray(Mat & image);
    void static thresholdBinary(Mat & image);
    void static squares(Mat & image);
    void static hsv(Mat & src);
    void static resizedownup(Mat & image);
    void static adaptiveBilateralFilter(Mat & image);
    void static kMeans(Mat & src);
    void static sobel(Mat & src);

    double static cross(Point v1,Point v2);
    bool static getIntersectionPoint(Point a1, Point a2, Point b1, Point b2, Point & intPnt);
    double static angle( Point pt1, Point pt2, Point pt0 );

    void static findSquares( const Mat& image, vector<vector<Point> >& squares );
    void static drawSquares( Mat& image, const vector<vector<Point> >& squares );

    void static filterLines(std::vector<cv::Vec4i>& lines);
    void static sortCorners(std::vector<cv::Point2f>& corners,cv::Point2f center);
    double static vectorLength(cv::Point a, cv::Point b);
    bool static sortByLength(const cv::Vec4i &lineA, const cv::Vec4i &lineB);
};

#endif // IMAGEOPERATIONS_H
