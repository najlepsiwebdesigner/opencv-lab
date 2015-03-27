#ifndef HELPERS
#define HELPERS

#include <vector>
#include <cmath>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

double cross(Point v1,Point v2);
bool getIntersectionPoint(Point a1, Point a2, Point b1, Point b2, Point & intPnt);
double angle( Point pt1, Point pt2, Point pt0 );


void findSquares( const Mat& image, vector<vector<Point> >& squares );
void drawSquares( Mat& image, const vector<vector<Point> >& squares );

void filterLines(std::vector<cv::Vec4i>& lines);
void sortCorners(std::vector<cv::Point2f>& corners,cv::Point2f center);
double vectorLength(cv::Point a, cv::Point b);
bool sortByLength(const cv::Vec4i &lineA, const cv::Vec4i &lineB);





void doLines(Mat & src, Mat & dst);
void doThreshold(Mat & src, Mat & dst);
#endif // HELPERS
