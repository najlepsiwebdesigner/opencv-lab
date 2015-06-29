#include "idocr.h"

using namespace cv;
using namespace std;

idOCR::idOCR()
{

}

idOCR::~idOCR()
{

}


void idOCR::fitImage(const Mat& src,Mat& dst, float destWidth, float destHeight) {
    int srcWidth = src.cols;
    int srcHeight = src.rows;

    float srcRatio = (float) srcWidth / (float) srcHeight;

    float widthRatio = destWidth / srcWidth;
    float heightRatio = destHeight / srcHeight;

    float newWidth = 0;
    float newHeight = 0;

    if (srcWidth > srcHeight) {
        destHeight = destWidth / srcRatio;
    } else {
        destWidth = destHeight * srcRatio;
    }
    cv::resize(src, dst,Size((int)round(destWidth), (int)round(destHeight)),0,0);
}




void idOCR::process(Mat & image) {

    // for id filtering, we need only small version of image
        Mat image_vga(Size(640,480),CV_8UC3,Scalar(0));
        fitImage(image, image_vga, 640, 480);

    // filtering
        cv::GaussianBlur(image_vga, image_vga, Size( 7, 7) ,7,7);

        Mat small;
        cv::resize(image_vga, small,Size(320,240),0,0);
        cv::medianBlur(small, small, 9);
        cv::resize(small, image_vga, Size(image_vga.cols,image_vga.rows));

        cv::resize(image_vga, small,Size(320,240),0,0);
        cv::medianBlur(small, small, 9);
        cv::resize(small, image_vga, Size(image_vga.cols,image_vga.rows));

        cv::Mat const shape = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

        cv::dilate(image_vga, image_vga, Mat(), Point(-1, -1), 3, 1, 1);
        cv::erode(image_vga, image_vga, shape, Point(-1,-1), 1);

        cv::dilate(image_vga, image_vga, Mat(), Point(-1, -1), 1, 1, 1);
        cv::erode(image_vga, image_vga, shape, Point(-1,-1), 1);

        cv::dilate(image_vga, image_vga, Mat(), Point(-1, -1), 1, 1, 1);
        cv::erode(image_vga, image_vga, shape, Point(-1,-1), 1);

        cv::GaussianBlur( image_vga, image_vga, Size(3,3), 0, 0, BORDER_DEFAULT );

    // sobel
        Mat grad, src_gray;
        int scale = 1;
        int delta = 0;
        int ddepth = CV_16S;
        /// Convert it to gray
        cvtColor( image_vga, src_gray, COLOR_RGB2GRAY );

        /// Generate grad_x and grad_y
        Mat grad_x, grad_y;
        Mat abs_grad_x, abs_grad_y;

        /// Gradient X
        //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
        Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_x, abs_grad_x );

        /// Gradient Y
        //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
        Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_y, abs_grad_y );

        /// Total Gradient (approximate)
        addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

        cvtColor( grad, image_vga, COLOR_GRAY2RGB );

    // now threshold
        cv::cvtColor(image_vga,image_vga, CV_RGB2GRAY);
        cv::threshold(image_vga,image_vga,0,255,THRESH_BINARY + CV_THRESH_OTSU);

        cv::dilate(image_vga, image_vga, Mat(), Point(-1, -1), 3, 1, 1);
        cv::erode(image_vga, image_vga, shape, Point(-1,-1), 1);

        cv::dilate(image_vga, image_vga, Mat(), Point(-1, -1), 1, 1, 1);
        cv::erode(image_vga, image_vga, shape, Point(-1,-1), 1);

        cv::dilate(image_vga, image_vga, Mat(), Point(-1, -1), 1, 1, 1);
        cv::erode(image_vga, image_vga, shape, Point(-1,-1), 1);

    // fill biggest contour in the image to get rid of rest of structures inside
        int largest_area=0;
        int largest_contour_index=0;

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;

        findContours( image_vga, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE ); // Find the contours in the image

        for( int i = 0; i< contours.size(); i++ )
        {
            double a=contourArea( contours[i],false);
            if(a>largest_area){
                largest_area=a;
                largest_contour_index=i;
                drawContours( image_vga, contours,i, Scalar(0,0,0), -1, 8, hierarchy, 0);
            }
        }

        drawContours( image_vga, contours,largest_contour_index, Scalar(255,0,0), -1, 8, hierarchy, 0);

        cv::Canny(image_vga, image_vga, 30, 90);
        cv::dilate(image_vga, image_vga, Mat(), Point(1,-1));

    cv::cvtColor(image_vga,image_vga, CV_GRAY2RGB);



    image = image_vga.clone();

//        Mat dst;
//        cv::cvtColor(image,dst , CV_RGB2GRAY);

//        std::vector<cv::Vec4i> lines;
//        cv::HoughLinesP(dst, lines, 1, CV_PI/360,50,50, 10);


//    cv::cvtColor(dst,dst , CV_GRAY2RGB);
//        // draw lines around image
//        std::vector<cv::Vec4i> myLines;

//        cv::Vec4i leftLine;leftLine[0] = 0;leftLine[1] = 0;leftLine[2] = 0;leftLine[3] = dst.rows;
//        myLines.push_back(leftLine);
//        cv::Vec4i rightLine;
//        rightLine[0] = dst.cols;rightLine[1] = 0;rightLine[2] = dst.cols;rightLine[3] = dst.rows;
//        myLines.push_back(rightLine);
//        cv::Vec4i topLine;
//        topLine[0] = dst.cols;topLine[1] = 0;topLine[2] = 0;topLine[3] = 0;
//        myLines.push_back(topLine);
//        cv::Vec4i bottomLine;
//        bottomLine[0] = dst.cols;bottomLine[1] = dst.rows;bottomLine[2] = 0;bottomLine[3] = dst.rows;
//        myLines.push_back(bottomLine);

//        // expand lines to borders of the image - we will get intersections with image borders easily
//        for (int i = 0; i < lines.size(); i++)
//        {
//            std::vector<cv::Point2f> lineIntersections;
//            for (int j = 0; j < myLines.size(); j++)
//            {
//                cv::Vec4i v = lines[i];
//                Point2f intersection;

//                bool has_intersection = ImageOperations::getIntersectionPoint(
//                    Point(lines[i][0],lines[i][1]),
//                    Point(lines[i][2], lines[i][3]),
//                    Point(myLines[j][0],myLines[j][1]),
//                    Point(myLines[j][2], myLines[j][3]),
//                    intersection);

//                if (has_intersection
//                    && intersection.x >= 0
//                    && intersection.y >= 0
//                    && intersection.x <= dst.cols
//                    && intersection.y <= dst.rows){
//                    lineIntersections.push_back(intersection);
//                }
//            }

//            if (lineIntersections.size() > 0) {
//                lines[i][0] = lineIntersections[0].x;
//                lines[i][1] = lineIntersections[0].y;
//                lines[i][2] = lineIntersections[1].x;
//                lines[i][3] = lineIntersections[1].y;
//            }
//        }



//        struct LineCluster {
//            int sumX1;
//            int sumY1;
//            int sumX2;
//            int sumY2;
//            int count;
//        };

//        vector<LineCluster> clusters;

//        int distanceThreshold = 30;
//        double angleThreshold = 0.10;

//            // create first group
//            LineCluster cluster;
//            cluster.sumX1 = lines[0][0];
//            cluster.sumY1 = lines[0][1];
//            cluster.sumX2 = lines[0][2];
//            cluster.sumY2 = lines[0][3];
//            cluster.count = 1;
//            clusters.push_back(cluster);

//            // loop through rest of groups
//            for (int i = 1; i < lines.size(); i++) {
//                bool in_some_cluster = false;

//                for (int j = 0; j < clusters.size(); j++) {

//                     int cluster_x1 = clusters[j].sumX1/clusters[j].count;
//                     int cluster_y1 = clusters[j].sumY1/clusters[j].count;
//                     int cluster_x2 = clusters[j].sumX2/clusters[j].count;
//                     int cluster_y2 = clusters[j].sumY2/clusters[j].count;

//                     double angle1 = atan2((double)lines[i][3] - lines[i][1], (double)lines[i][2] - lines[i][0]);
//                     double angle2 = atan2((double)cluster_y2 - cluster_y1, (double)cluster_x2 - cluster_x1);
//                     float distance_cluster1_to_line1 = sqrt(((cluster_x1 - lines[i][0])*(cluster_x1 - lines[i][0])) + (cluster_y1 - lines[i][1])*(cluster_y1 - lines[i][1]));
//                     float distance_cluster1_to_line2 = sqrt(((cluster_x1 - lines[i][2])*(cluster_x1 - lines[i][2])) + (cluster_y1 - lines[i][3])*(cluster_y1 - lines[i][3]));
//                     float distance_cluster2_to_line1 = sqrt(((cluster_x2 - lines[i][0])*(cluster_x2 - lines[i][0])) + (cluster_y2 - lines[i][1])*(cluster_y2 - lines[i][1]));
//                     float distance_cluster2_to_line2 = sqrt(((cluster_x2 - lines[i][2])*(cluster_x2 - lines[i][2])) + (cluster_y2 - lines[i][3])*(cluster_y2 - lines[i][3]));

//                     if (((distance_cluster1_to_line1 < distanceThreshold) &&
//                             (distance_cluster2_to_line2 < distanceThreshold || abs(angle1 - angle2) < angleThreshold)) ||
//                          ((distance_cluster1_to_line2 < distanceThreshold) &&
//                             (distance_cluster2_to_line1 < distanceThreshold || abs(angle1 - angle2) < angleThreshold))){

//                             clusters[j].sumX1 += lines[i][0];
//                             clusters[j].sumY1 += lines[i][1];
//                             clusters[j].sumX2 += lines[i][2];
//                             clusters[j].sumY2 += lines[i][3];
//                             clusters[j].count += 1;
//                             in_some_cluster = true;
//                    }
//                }
//                // if point doesnt fit, create new group for it
//                if (in_some_cluster == false){
//                    LineCluster cluster;
//                    cluster.sumX1 = lines[i][0];
//                    cluster.sumY1 = lines[i][1];
//                    cluster.sumX2 = lines[i][2];
//                    cluster.sumY2 = lines[i][3];
//                    cluster.count = 1;
//                    clusters.push_back(cluster);
//                }
//            }

//            std::vector<cv::Vec4i> clusteredLines;

//            for (int i = 0; i < clusters.size(); i++){
//                circle(dst, Point(clusters[i].sumX1/clusters[i].count, clusters[i].sumY1/clusters[i].count), 5, Scalar(0,0,255),-1);
//                circle(dst, Point(clusters[i].sumX2/clusters[i].count, clusters[i].sumY2/clusters[i].count), 5, Scalar(0,0,255),-1);

//                cv::line(dst, Point(clusters[i].sumX1/clusters[i].count, clusters[i].sumY1/clusters[i].count), Point(clusters[i].sumX2/clusters[i].count, clusters[i].sumY2/clusters[i].count), CV_RGB(255,0,0), 1);

//                cv::Vec4i line;
//                line[0] = clusters[i].sumX1/clusters[i].count;
//                line[1] = clusters[i].sumY1/clusters[i].count;
//                line[2] = clusters[i].sumX2/clusters[i].count;
//                line[3] = clusters[i].sumY2/clusters[i].count;

//                clusteredLines.push_back(line);
//            }

//        std::vector<cv::Point2f> corners;
//        for (int i = 0; i < clusteredLines.size(); i++)
//        {
//            for (int j = i+1; j < clusteredLines.size(); j++)
//            {
//                cv::Vec4i v = clusteredLines[i];
//                Point2f intersection;

//                bool has_intersection = ImageOperations::getIntersectionPoint(
//                    Point(clusteredLines[i][0],clusteredLines[i][1]),
//                    Point(clusteredLines[i][2], clusteredLines[i][3]),
//                    Point(clusteredLines[j][0],clusteredLines[j][1]),
//                    Point(clusteredLines[j][2], clusteredLines[j][3]),
//                    intersection);

//                if (has_intersection
//                    && intersection.x > 0
//                    && intersection.y > 0
//                    && intersection.x < dst.cols
//                    && intersection.y < dst.rows){
//                    corners.push_back(intersection);
//                }

//                cv::circle(dst, intersection, 3, CV_RGB(0,0,255), 2);
//            }
//        }


//        if (corners.size() == 4) {

//            cv::Point2f center(0,0);

//            for (int i = 0; i < corners.size(); i++)
//                center += corners[i];
//            center *= (1. / corners.size());


//            ImageOperations::sortCorners(corners, center);
//            cv::circle(dst, center, 3, CV_RGB(255,255,0), 2);

//            // Define the destination image
//            cv::Mat quad = cv::Mat::zeros(420, 640, CV_8UC3);

//            // Corners of the destination image
//            std::vector<cv::Point2f> quad_pts;
//            quad_pts.push_back(cv::Point2f(0, 0));
//            quad_pts.push_back(cv::Point2f(quad.cols, 0));
//            quad_pts.push_back(cv::Point2f(quad.cols, quad.rows));
//            quad_pts.push_back(cv::Point2f(0, quad.rows));

//            // Get transformation matrix
//            cv::Mat transmtx = cv::getPerspectiveTransform(corners, quad_pts);

//            // Apply perspective transformation
//            cv::warpPerspective(image,quad, transmtx, quad.size(),INTER_LINEAR,BORDER_TRANSPARENT);

//            cv::Mat newImage = cv::Mat::zeros(480, 640, CV_8UC3);

//            quad.copyTo(newImage.rowRange(0,quad.rows).colRange(0,quad.cols));

//            image = newImage.clone();
//        }
}
