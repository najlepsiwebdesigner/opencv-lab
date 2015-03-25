#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

cv::Point2f computeIntersect(cv::Vec4i a,
                             cv::Vec4i b)
{
    int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3], x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];
    // float denom;

    if (float d = ((float)(x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4)))
    {
        cv::Point2f pt;
        pt.x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d;
        pt.y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d;
        return pt;
    }
    else
        return cv::Point2f(-1, -1);
}



void filterLines(std::vector<cv::Vec4i>& lines) {
    std::vector<cv::Vec4i> output;
    double angleThreshold = 0.005;
    // double xThreshold = 10;
    // double yThreshold = 10;

    bool is_in_output = false;

    for (int i = 0; i < lines.size(); i++)
    {
        is_in_output = false;
        for (int j = 0; j < output.size(); j++)
        {
            double angle1 = atan2((double)lines[i][3] - lines[i][1], (double)lines[i][2] - lines[i][0]);
            double angle2 = atan2((double)output[j][3] - output[j][1], (double)output[j][2] - output[j][0]);

            if (abs(angle1 - angle2) < angleThreshold){

                is_in_output = true;
                break;
            }

        }
        // cout << "Is in output:" << is_in_output << "\n";

        if (!is_in_output){
            output.push_back(lines[i]);
        }
    }
    // cout << "Size:" <<output.size();
    lines = output;
}



void sortCorners(std::vector<cv::Point2f>& corners,
                 cv::Point2f center)
{
    std::vector<cv::Point2f> top, bot;

    for (int i = 0; i < corners.size(); i++)
    {
        if (corners[i].y < center.y)
            top.push_back(corners[i]);
        else
            bot.push_back(corners[i]);
    }
    corners.clear();

    if (top.size() == 2 && bot.size() == 2){
        cv::Point2f tl = top[0].x > top[1].x ? top[1] : top[0];
        cv::Point2f tr = top[0].x > top[1].x ? top[0] : top[1];
        cv::Point2f bl = bot[0].x > bot[1].x ? bot[1] : bot[0];
        cv::Point2f br = bot[0].x > bot[1].x ? bot[0] : bot[1];


        corners.push_back(tl);
        corners.push_back(tr);
        corners.push_back(br);
        corners.push_back(bl);
    }
}

double vectorLength(cv::Point a, cv::Point b) {
    double res = cv::norm(a-b);
    return res;
}

bool sortByLength(const cv::Vec4i &lineA, const cv::Vec4i &lineB) {
    return vectorLength(cv::Point(lineA[0], lineA[1]), cv::Point(lineA[2], lineA[3])) > vectorLength(cv::Point(lineB[0], lineB[1]), cv::Point(lineB[2], lineB[3]));
}

double cross(Point v1,Point v2){
    return v1.x*v2.y - v1.y*v2.x;
}


bool getIntersectionPoint(Point a1, Point a2, Point b1, Point b2, Point & intPnt){
    Point p = a1;
    Point q = b1;
    Point r(a2-a1);
    Point s(b2-b1);

    if(cross(r,s) == 0) {return false;}

    double t = cross(q-p,s)/cross(r,s);

    intPnt = p + t*r;
    return true;
}

