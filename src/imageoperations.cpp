#include "imageoperations.h"


ImageOperations::ImageOperations()
{

}

ImageOperations::~ImageOperations()
{

}


double ImageOperations::cross(Point v1,Point v2){

    return v1.x*v2.y - v1.y*v2.x;
}


bool ImageOperations::getIntersectionPoint(Point a1, Point a2, Point b1, Point b2, Point2f & intPnt){
    Point p = a1;
    Point q = b1;
    Point r(a2-a1);
    Point s(b2-b1);

    if(cross(r,s) == 0) {return false;}

    double t = cross(q-p,s)/cross(r,s);

    intPnt = p + t*r;
    return true;
}


Point2f ImageOperations::computeIntersect(cv::Vec4i a, cv::Vec4i b)
{
    int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3], x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];
    float denom;

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


// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
double ImageOperations::angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}


// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
void ImageOperations::findSquares( const Mat& image, vector<vector<Point> >& squares )
{
    int thresh = 80, N = 11;
    squares.clear();

    Mat pyr, timg, gray0(image.size(), CV_8U), gray;
    image.copyTo(timg);
    // down-scale and upscale the image to filter out the noise
//    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
//    pyrUp(pyr, timg, image.size());
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
                dilate(gray, gray, Mat(), Point(1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }

            // find contours and store them all as a list
            findContours(gray, contours, RETR_LIST, CV_CHAIN_APPROX_TC89_L1);




            int largest_area=0;
            int largest_contour_index=0;
            for( int i = 0; i< contours.size(); i++ ) // iterate through each contour.
            {
                double a=contourArea( contours[i],false);  //  Find the area of contour
                if(a>largest_area){
                    largest_area=a;
                    largest_contour_index=i;                //Store the index of largest contour
                }
            }



            cv::drawContours(gray, contours,largest_contour_index,Scalar(255,255,255));
//            cv::imshow("test", gray);
//            break;



            vector<Point> approx;

            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.01, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 1000 && isContourConvex(Mat(approx))
                   )
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
                    if( maxCosine < 0.5)
                        squares.push_back(approx);
                }
            }
        }
    }
}


// the function draws all the squares in the image
void ImageOperations::drawSquares( Mat& image, const vector<vector<Point> >& squares )
{
    for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        polylines(image, &p, &n, 1, true, Scalar(0,0,255), 1, CV_AA);
    }
}





void ImageOperations::filterLines(std::vector<cv::Vec4i>& lines) {
    std::vector<cv::Vec4i> output;
//    double angleThreshold = 0.010;
//    // double xThreshold = 10;
//    // double yThreshold = 10;

//    bool is_in_output = false;

//    for (int i = 0; i < lines.size(); i++)
//    {
//        is_in_output = false;
//        for (int j = 0; j < output.size(); j++)
//        {
//            double angle1 = atan2((double)lines[i][3] - lines[i][1], (double)lines[i][2] - lines[i][0]);
//            double angle2 = atan2((double)output[j][3] - output[j][1], (double)output[j][2] - output[j][0]);

//            if (abs(angle1 - angle2) < angleThreshold){
//                is_in_output = true;
//                break;
//            }

//        }
//        // cout << "Is in output:" << is_in_output << "\n";

//        if (!is_in_output){
//            output.push_back(lines[i]);
//        }
//    }








    // cout << "Size:" <<output.size();
//    lines = output;
}




void ImageOperations::sortCorners(std::vector<cv::Point2f>& corners,
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

double ImageOperations::vectorLength(cv::Point a, cv::Point b) {
    double res = cv::norm(a-b);
    return res;
}

bool ImageOperations::sortByLength(const cv::Vec4i &lineA, const cv::Vec4i &lineB) {
    return vectorLength(cv::Point(lineA[0], lineA[1]), cv::Point(lineA[2], lineA[3])) > vectorLength(cv::Point(lineB[0], lineB[1]), cv::Point(lineB[2], lineB[3]));
}


void ImageOperations::fitImage(const Mat& src,Mat& dst, float destWidth, float destHeight) {
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










void ImageOperations::equalize(Mat & image) {
    Mat ycrcb;
    cvtColor(image,ycrcb,CV_BGR2YCrCb);
    vector<Mat> channels;
    split(ycrcb,channels);
    equalizeHist(channels[0], channels[0]);
    Mat result;
    merge(channels,ycrcb);
    cvtColor(ycrcb,result,CV_YCrCb2BGR);

    image = result;
}

void ImageOperations::lines(Mat & image) {
//    Mat src = image.clone();
    Mat dst;
    cv::cvtColor(image,dst , CV_RGB2GRAY);
//    cv::GaussianBlur(dst, dst, Size( 7, 7) ,7,7);

//    cv::threshold(dst,dst,0,255,CV_THRESH_BINARY);
//    cv::Canny(dst, dst, 1, 1, 3, true);
//    dilate( dst, dst, Mat(Size(1,1), CV_8UC1));



    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(dst, lines, 1, CV_PI/360,50,50, 10);

//    cout << lines.size() << endl;

    std::sort(lines.begin(), lines.end(), sortByLength);
//    filterLines(lines);

cv::cvtColor(dst,dst , CV_GRAY2RGB);


    // draw lines around image
    std::vector<cv::Vec4i> myLines;

    cv::Vec4i leftLine;leftLine[0] = 0;leftLine[1] = 0;leftLine[2] = 0;leftLine[3] = dst.rows;
    myLines.push_back(leftLine);
    cv::Vec4i rightLine;
    rightLine[0] = dst.cols;rightLine[1] = 0;rightLine[2] = dst.cols;rightLine[3] = dst.rows;
    myLines.push_back(rightLine);
    cv::Vec4i topLine;
    topLine[0] = dst.cols;topLine[1] = 0;topLine[2] = 0;topLine[3] = 0;
    myLines.push_back(topLine);
    cv::Vec4i bottomLine;
    bottomLine[0] = dst.cols;bottomLine[1] = dst.rows;bottomLine[2] = 0;bottomLine[3] = dst.rows;
    myLines.push_back(bottomLine);



    std::vector<cv::Point2f> intersections;

    for (int i = 0; i < lines.size(); i++)
    {
        for (int j = 0; j < myLines.size(); j++)
        {
            cv::Vec4i v = lines[i];
            Point2f intersection;

            bool has_intersection = getIntersectionPoint(
                Point(lines[i][0],lines[i][1]),
                Point(lines[i][2], lines[i][3]),
                Point(myLines[j][0],myLines[j][1]),
                Point(myLines[j][2], myLines[j][3]),
                intersection);

            if (has_intersection
                && intersection.x >= 0
                && intersection.y >= 0
                && intersection.x <= dst.cols
                && intersection.y <= dst.rows){
                intersections.push_back(intersection);
                cv::circle(dst, intersection, 3, CV_RGB(255,255,0),3);
            }


        }
    }








//    lines.resize(10);

    // Expand and draw the lines
    for (int i = 0; i < lines.size(); i++)
    {
//        cv::Vec4i v = lines[i];
//        lines[i][0] = 0;
//        lines[i][1] = ((float)v[1] - v[3]) / (v[0] - v[2]) * - v[0] + v[1];
//        lines[i][2] = dst.cols;
//        lines[i][3] = ((float)v[1] - v[3]) / (v[0] - v[2]) * (dst.cols - v[2]) + v[3];
        cv::line(dst, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), CV_RGB(255,0,0), 1);
    }

//    cv::imshow("",dst);


//    std::vector<cv::Point2f> corners;
//    for (int i = 0; i < lines.size(); i++)
//    {
//        for (int j = i+1; j < lines.size(); j++)
//        {
//            cv::Vec4i v = lines[i];
//            Point2f intersection;

//            bool has_intersection = getIntersectionPoint(
//                Point(lines[i][0],lines[i][1]),
//                Point(lines[i][2], lines[i][3]),
//                Point(lines[j][0],lines[j][1]),
//                Point(lines[j][2], lines[j][3]),
//                intersection);

//            if (has_intersection
//                && intersection.x > 0
//                && intersection.y > 0
//                && intersection.x < dst.cols
//                && intersection.y < dst.rows){
//                corners.push_back(intersection);
//            }

//            cv::circle(dst, intersection, 3, CV_RGB(0,0,255), 2);
//        }
//    }

//    if (corners.size() < 1) {
//        return;
//    }

//    vector<vector<cv::Point>> approx;
//    cv::approxPolyDP(cv::Mat(corners), approx[0], cv::arcLength(cv::Mat(corners), true) * 0.03, true);

//    if (approx[0].size() != 4)
//    {
//        std::cout << "The object is not quadrilateral!" << std::endl;
//        return;
//    }







//    for (int i = 0; i < myLines.size(); i++)
//    {
//        cv::line(dst, cv::Point(myLines[i][0], myLines[i][1]), cv::Point(myLines[i][2], myLines[i][3]), CV_RGB(0, 255,0),1);
//    }

//    // init
//    struct Group {
//        int x_sum;
//        int y_sum;
//        int count;
//    };

//    int distanceThreshold = 30;
//    vector<Group> groups;

//    // create first group
//    Group group;
//    group.x_sum = intersections[0].x;
//    group.y_sum = intersections[0].y;
//    group.count = 1;
//    groups.push_back(group);

//    // loop through rest of groups
//    for (int i = 1; i < intersections.size(); i++) {
//        bool in_some_group = false;

//        for (int j = 0; j < groups.size(); j++) {
//            Point groupCenter(groups[j].x_sum/groups[j].count, groups[j].y_sum/groups[j].count);
//            int sum = ((groupCenter.x - intersections[i].x) * (groupCenter.x - intersections[i].x)) + ((groupCenter.y - intersections[i].y) * (groupCenter.y - intersections[i].y));
//            float length = sqrt(sum);

//            // if this point fits into some group, push it in
//            if (length < distanceThreshold) {
//                groups[j].x_sum += intersections[i].x;
//                groups[j].y_sum += intersections[i].y;
//                groups[j].count += 1;
//                in_some_group = true;
//            }
//        }
//        // if point doesnt fit, create new group for it
//        if (in_some_group == false){
//            Group group;
//            group.x_sum = intersections[i].x;
//            group.y_sum = intersections[i].y;
//            group.count = 1;
//            groups.push_back(group);
//        }
//    }

//    for (int i = 0; i < groups.size(); i++){
//        circle(dst, Point(groups[i].x_sum/groups[i].count, groups[i].y_sum/groups[i].count), 5, Scalar(0,0,255),-1);
//    }





//    cv::Point2f center(0,0);

//    for (int i = 0; i < corners.size(); i++)
//        center += corners[i];
//    center *= (1. / corners.size());

//    sortCorners(corners, center);
//    cv::circle(dst, center, 3, CV_RGB(255,255,0), 2);

    image = dst.clone();
}






void ImageOperations::gaussian(Mat & image) {
    cv::GaussianBlur(image, image, Size( 7, 7) ,7,7);
}


void ImageOperations::thresholdAdaptive(Mat & image) {
    Mat dst;
    cv::cvtColor(image,dst , CV_RGB2GRAY);
    cv::adaptiveThreshold(dst, dst, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 7, 0);
    cv::cvtColor(dst,image, CV_GRAY2RGB);
}

void ImageOperations::thresholdGray(Mat & image) {
    Mat dst;
    cv::cvtColor(image,dst , CV_RGB2GRAY);
    cv::threshold(dst,dst,0,255,THRESH_TOZERO + CV_THRESH_OTSU);
    cv::cvtColor(dst,image, CV_GRAY2RGB);
}

void ImageOperations::thresholdBinary(Mat & image) {
    Mat dst;
    cv::cvtColor(image,image, CV_RGB2GRAY);
    cv::threshold(image,image,0,255,THRESH_BINARY + CV_THRESH_OTSU);
    cv::cvtColor(image,image, CV_GRAY2RGB);
}




void ImageOperations::squares(Mat & image) {
    Mat src = image;
    Mat dst;
    vector<vector<Point> > squares;
    cvtColor(src, dst, CV_BGR2RGB);
    findSquares(dst, squares);
    drawSquares(dst, squares);
    cvtColor(dst, dst, CV_RGB2BGR);
    image = dst;
}
void ImageOperations::hsv(Mat & src) {
    cv::Mat hsv;
    cv::cvtColor(src, hsv, CV_BGR2HSV);
    cv::Mat hue(src.size(), CV_8U);
    //the third arguments are two number a pair, (0, 0) means copy the data of channels 0(hsv) to channels 0(hue)
    cv::mixChannels(hsv, hue, {0, 0});
    cv::Mat otsuMat;
    cv::adaptiveThreshold(hue,otsuMat,255, CV_ADAPTIVE_THRESH_MEAN_C , CV_THRESH_BINARY, 3, 0);
    cv::cvtColor(otsuMat,src, CV_GRAY2RGB);
}


void ImageOperations::resizedownup(Mat & image){
    Mat small;
    cv::resize(image, small,Size(320,200),0,0);
    cv::medianBlur(small, small, 9);
    cv::resize(small, image, Size(image.cols,image.rows));
}


void ImageOperations::adaptiveBilateralFilter(Mat & image){
    Mat dst;
//    cv::bilateralFilter ( image, dst, 15, 100, 35 );
    cv::adaptiveBilateralFilter(image, dst, Size(3,3),3);
    image = dst;
}

void ImageOperations::kMeans(Mat & src) {
    cv::Mat samples(src.total(), 3, CV_32F);
       auto samples_ptr = samples.ptr<float>(0);
       for( int row = 0; row != src.rows; ++row){
           auto src_begin = src.ptr<uchar>(row);
           auto src_end = src_begin + src.cols * src.channels();
           //auto samples_ptr = samples.ptr<float>(row * src.cols);
           while(src_begin != src_end){
               samples_ptr[0] = src_begin[0];
               samples_ptr[1] = src_begin[1];
               samples_ptr[2] = src_begin[2];
               samples_ptr += 3; src_begin +=3;
           }
       }

       //step 2 : apply kmeans to find labels and centers
       int clusterCount = 3;
       cv::Mat labels;
       int attempts = 5;
       cv::Mat centers;
       cv::kmeans(samples, clusterCount, labels,
                  cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,
                                   10, 0.01),
                  attempts, cv::KMEANS_PP_CENTERS, centers);

       //step 3 : map the centers to the output
       cv::Mat new_image(src.size(), src.type());
       for( int row = 0; row != src.rows; ++row){
           auto new_image_begin = new_image.ptr<uchar>(row);
           auto new_image_end = new_image_begin + new_image.cols * 3;
           auto labels_ptr = labels.ptr<int>(row * src.cols);

           while(new_image_begin != new_image_end){
               int const cluster_idx = *labels_ptr;
               auto centers_ptr = centers.ptr<float>(cluster_idx);
               new_image_begin[0] = centers_ptr[0];
               new_image_begin[1] = centers_ptr[1];
               new_image_begin[2] = centers_ptr[2];
               new_image_begin += 3; ++labels_ptr;
           }
       }

       src = new_image.clone();
}


void ImageOperations::sobel(Mat & src) {
    GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
    Mat grad, src_gray;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    /// Convert it to gray
    cvtColor( src, src_gray, COLOR_RGB2GRAY );

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

//    cv::dilate(grad, grad, Mat(), Point(1,-1));

    cvtColor( grad, src, COLOR_GRAY2RGB );
}


void ImageOperations::canny(Mat & image) {
    cv::Canny(image, image, 30, 90);
    dilate(image, image, Mat(), Point(1,-1));
    cv::cvtColor(image,image, CV_GRAY2RGB);
}


void ImageOperations::erosion(Mat & image) {
   //get the structuring of rectangle, size is 5 * 5
   cv::Mat const shape = cv::getStructuringElement(
                 cv::MORPH_RECT, cv::Size(5, 5));
   cv::erode(image, image, shape);
}


void ImageOperations::dilation(Mat & image) {
    dilate(image, image, Mat(), Point(-1, -1), 2, 1, 1);
}


void ImageOperations::biggestContour(Mat & image) {

    int largest_area=0;
    int largest_contour_index=0;
//    Rect bounding_rect;

    Mat thr = image.clone();
    cvtColor(image,thr,CV_RGB2GRAY); //Convert to gray
    //     threshold(thr, thr,25, 255,THRESH_BINARY); //Threshold the gray

    vector<vector<Point>> contours; // Vector for storing contour
    vector<Vec4i> hierarchy;


//    imshow("thr", thr);
    findContours( thr, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE ); // Find the contours in the image

    for( int i = 0; i< contours.size(); i++ ) // iterate through each contour.
    {
        double a=contourArea( contours[i],false);  //  Find the area of contour
        if(a>largest_area){
            largest_area=a;
            largest_contour_index=i;                //Store the index of largest contour
//            bounding_rect=boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
            drawContours( image, contours,i, Scalar(0,0,0), -1, 8, hierarchy, 0);
        }

    }

    Scalar color( 255,0,0);
//    image.setTo(cv::Scalar::all(0));
    drawContours( image, contours,largest_contour_index, color, -1, 8, hierarchy, 0); // Draw the largest contour using previously stored index.
//    rectangle(image, bounding_rect,  Scalar(0,255,0),1, 8,0);

}



void ImageOperations::convexH(Mat & image) {
    vector<vector<Point> > contours;
    vector<vector<Point> > hull(1);
    vector<Vec4i> hierarchy;
    int largest_area=0;
    int largest_contour_index=0;

    cvtColor( image, image, CV_RGB2GRAY );
    cv::threshold(image,image,0,255,THRESH_BINARY + CV_THRESH_OTSU);

    findContours( image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    /// Find biggest contour
    for( int i = 0; i < contours.size(); i++ )
    {
        double a=contourArea( contours[i],false);  //  Find the area of contour
        if(a>largest_area){
            largest_area=a;
            largest_contour_index=i;                //Store the index of largest contour
        }
    }

    // Calculate convex hull of largest contour
    convexHull( Mat(contours[largest_contour_index]), hull[0], false );

    cvtColor( image, image, CV_GRAY2RGB);

    drawContours( image, contours, largest_contour_index, Scalar(255,255,0), 1, 8, vector<Vec4i>(), 0, Point() );
    drawContours( image, hull, 0, Scalar(255,255,0), 1, 8, vector<Vec4i>(), 0, Point() );
}