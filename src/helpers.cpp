#include "helpers.h"

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


// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}


// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
void findSquares( const Mat& image, vector<vector<Point> >& squares )
{
    int thresh = 50, N = 11;
    squares.clear();

    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());
    vector<vector<Point> > contours;

    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }

            // find contours and store them all as a list
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 1000 &&
                    isContourConvex(Mat(approx)) )
                {
                    double maxCosine = 0;

                    for( int j = 2; j < 5; j++ )
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.3 )
                        squares.push_back(approx);
                }
            }
        }
    }
}


// the function draws all the squares in the image
void drawSquares( Mat& image, const vector<vector<Point> >& squares )
{
    for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        polylines(image, &p, &n, 1, true, Scalar(0,0,255), 1, CV_AA);
    }

    imshow("squares", image);
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

void doThreshold(Mat & src, Mat & dst){
    cv::cvtColor(src,dst , CV_BGR2GRAY);
    cv::GaussianBlur(dst, dst, Size( 7, 7) ,7,7);
    cv::threshold(dst,dst,0,255,THRESH_TOZERO + CV_THRESH_OTSU);
    cv::threshold(dst,dst,0,255,CV_THRESH_BINARY);
//    cv::adaptiveThreshold(dst,dst,255,ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 7, 2);
}


void doLines(Mat & src, Mat & dst) {
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

    if (lines.size() < 100){
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


Mat equalizeIntensity(const Mat& inputImage)
{
    if(inputImage.channels() >= 3)
    {
        Mat ycrcb;

        cvtColor(inputImage,ycrcb,CV_BGR2YCrCb);

        vector<Mat> channels;
        split(ycrcb,channels);

        equalizeHist(channels[0], channels[0]);

        Mat result;
        merge(channels,ycrcb);
        cout << "equalized" << endl;
        cvtColor(ycrcb,result,CV_YCrCb2BGR);

        return result;
    }

    cout << "not equalized" << endl;
    return inputImage;
}
