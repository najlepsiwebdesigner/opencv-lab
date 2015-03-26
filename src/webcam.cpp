#include "webcam.h"
#include "myfreenectdevice.h"

using namespace std;
using namespace cv;

Webcam::Webcam()
{

}

Webcam::~Webcam()
{

}


void Webcam::threshold(Mat & src, Mat & dst){
    cv::cvtColor(src,dst , CV_BGR2GRAY);
    cv::GaussianBlur(dst, dst, Size( 7, 7) ,7,7);
    cv::threshold(dst,dst,0,255,THRESH_TOZERO + CV_THRESH_OTSU);
    cv::threshold(dst,dst,0,255,CV_THRESH_BINARY);
//    cv::adaptiveThreshold(dst,dst,255,ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 7, 2);
}


void Webcam::contours(Mat & src, Mat & dst) {
    int largest_area=0;
    int largest_contour_index=0;
    vector<Vec4i> hierarchy;
    vector<vector<Point> > contours;
    vector<Point> contours_approx;
    vector<Point> shape;
//    cv::cvtColor(src,dst , CV_GRAY2RGB);
    cv::Canny(src, dst, 1, 1, 3);

    dilate( dst, dst, Mat(Size(1,1), CV_8UC1));
//    findContours( src, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );

//    cv::cvtColor(src, dst, CV_GRAY2RGB);

//    for( int i = 0; i< contours.size(); i++ )
//    {
//        //  Find the area of contour
//        double a=contourArea( contours[i],false);
//        if(a>largest_area){
//           largest_area=a;
//           largest_contour_index=i;
//           cout << "Largest area : " << a << endl;
//        }


//    }
//    drawContours( dst, contours,largest_contour_index, Scalar ( 0, 255,0), 1,8,hierarchy);


    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(dst, lines, 1, CV_PI/360, 75, 40, 10 );
    std::sort(lines.begin(), lines.end(), sortByLength);
    filterLines(lines);
    cv::cvtColor(dst,dst , CV_GRAY2RGB);

    if (lines.size() < 5){
        // Expand and draw the lines
        for (int i = 0; i < lines.size(); i++)
        {
            cv::Vec4i v = lines[i];
            lines[i][0] = 0;
            lines[i][1] = ((float)v[1] - v[3]) / (v[0] - v[2]) * - v[0] + v[1];
            lines[i][2] = dst.cols;
            lines[i][3] = ((float)v[1] - v[3]) / (v[0] - v[2]) * (dst.cols - v[2]) + v[3];
            cv::line(dst, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), CV_RGB(255,0,0));
        }


        //compute corners
        std::vector<cv::Point2f> corners;
        for (int i = 0; i < lines.size(); i++)
        {
            for (int j = i+1; j < lines.size(); j++)
            {
                // atan2((y1-y2)/(x2-x1))*180/math.pi
                cv::Vec4i v = lines[i];


                Point intersection;

                bool has_intersection = getIntersectionPoint(
                    Point(lines[i][0],lines[i][1]),
                    Point(lines[i][2], lines[i][3]),
                    Point(lines[j][0],lines[j][1]),
                    Point(lines[j][2], lines[j][3]),
                    intersection);

                if (has_intersection
                    && intersection.x > 0
                    && intersection.y > 0
                    && intersection.x < dst.cols
                    && intersection.y < dst.rows){
                    corners.push_back(intersection);
                }


                cv::circle(dst, intersection, 3, CV_RGB(0,0,255), 2);
            }
        }




        // compute and draw center of mass
        cv::Point2f center(0,0);

        for (int i = 0; i < corners.size(); i++)
            center += corners[i];
        center *= (1. / corners.size());

        sortCorners(corners, center);
        std::cout << "The corners were not sorted correctly!" << std::endl;
        cv::circle(dst, center, 3, CV_RGB(255,255,0), 2);
    }
}





void Webcam::showRGB() {
    bool die(false);
    string filename("snapshot");
    string suffix(".png");
    int i_snap(0);

//    Mat depthMat(Size(640,480),CV_16UC1);
//    Mat depthf (Size(640,480),CV_8UC1);
    Mat rgbMat(Size(640,480),CV_8UC3,Scalar(0));
    Mat thresholdedMat(Size(640,480),CV_8UC3,Scalar(0));
    Mat contoursMat(Size(640,480),CV_8UC3,Scalar(0));

    vector<vector<Point> > squares;

    // The next two lines must be changed as Freenect::Freenect
    // isn't a template but the method createDevice:
    // Freenect::Freenect<MyFreenectDevice> freenect;
    // MyFreenectDevice& device = freenect.createDevice(0);
    // by these two lines:

    Freenect::Freenect freenect;
    MyFreenectDevice& device = freenect.createDevice<MyFreenectDevice>(0);

    namedWindow("rgb",CV_WINDOW_AUTOSIZE);
    namedWindow("thresholded",CV_WINDOW_AUTOSIZE);
    namedWindow("processed",CV_WINDOW_AUTOSIZE);
    namedWindow("squares",CV_WINDOW_AUTOSIZE);
    device.startVideo();
    while (!die) {
        device.getVideo(rgbMat);

        findSquares(rgbMat, squares);

        threshold(rgbMat, thresholdedMat);
        cv::imshow("thresholded", thresholdedMat);
        contours(thresholdedMat, contoursMat);
        cv::imshow("processed", contoursMat);

        cv::imshow("rgb", rgbMat);

        drawSquares(rgbMat, squares);


        char k = cvWaitKey(5);
        if( k == 27 ){
            cvDestroyWindow("rgb");
            cvDestroyWindow("processed");
            cvDestroyWindow("thresholded");
            cvDestroyWindow("squares");
            break;
        }
        if( k == 8 ) {
            std::ostringstream file;
            file << filename << i_snap << suffix;
            cv::imwrite(file.str(),rgbMat);
            i_snap++;
        }
    }

    device.stopVideo();
}
