#include "webcam.h"

using namespace std;
using namespace cv;

class myMutex {
    public:
        myMutex() {
            pthread_mutex_init( &m_mutex, NULL );
        }
        void lock() {
            pthread_mutex_lock( &m_mutex );
        }
        void unlock() {
            pthread_mutex_unlock( &m_mutex );
        }
    private:
        pthread_mutex_t m_mutex;
};


class MyFreenectDevice : public Freenect::FreenectDevice {
    public:
        MyFreenectDevice(freenect_context *_ctx, int _index)
            : Freenect::FreenectDevice(_ctx, _index), m_buffer_depth(FREENECT_DEPTH_11BIT),
            m_buffer_rgb(FREENECT_VIDEO_RGB), m_gamma(2048), m_new_rgb_frame(false),
            m_new_depth_frame(false), depthMat(Size(640,480),CV_16UC1),
            rgbMat(Size(640,480), CV_8UC3, Scalar(0)),
            ownMat(Size(640,480),CV_8UC3,Scalar(0)) {

            for( unsigned int i = 0 ; i < 2048 ; i++) {
                float v = i/2048.0;
                v = std::pow(v, 3)* 6;
                m_gamma[i] = v*6*256;
            }
        }

        // Do not call directly even in child
        void VideoCallback(void* _rgb, uint32_t timestamp) {
            std::cout << "RGB callback" << std::endl;
            m_rgb_mutex.lock();
            uint8_t* rgb = static_cast<uint8_t*>(_rgb);
            rgbMat.data = rgb;
            m_new_rgb_frame = true;
            m_rgb_mutex.unlock();
        };

        // Do not call directly even in child
        void DepthCallback(void* _depth, uint32_t timestamp) {
            std::cout << "Depth callback" << std::endl;
            m_depth_mutex.lock();
            uint16_t* depth = static_cast<uint16_t*>(_depth);
            depthMat.data = (uchar*) depth;
            m_new_depth_frame = true;
            m_depth_mutex.unlock();
        }

        bool getVideo(Mat& output) {
            m_rgb_mutex.lock();
            if(m_new_rgb_frame) {
                cv::cvtColor(rgbMat, output, CV_RGB2BGR);
                m_new_rgb_frame = false;
                m_rgb_mutex.unlock();
                return true;
            } else {
                m_rgb_mutex.unlock();
                return false;
            }
        }

        bool getDepth(Mat& output) {
                m_depth_mutex.lock();
                if(m_new_depth_frame) {
                    depthMat.copyTo(output);
                    m_new_depth_frame = false;
                    m_depth_mutex.unlock();
                    return true;
                } else {
                    m_depth_mutex.unlock();
                    return false;
                }
            }
    private:
        std::vector<uint8_t> m_buffer_depth;
        std::vector<uint8_t> m_buffer_rgb;
        std::vector<uint16_t> m_gamma;
        Mat depthMat;
        Mat rgbMat;
        Mat ownMat;
        myMutex m_rgb_mutex;
        myMutex m_depth_mutex;
        bool m_new_rgb_frame;
        bool m_new_depth_frame;
};





Webcam::Webcam()
{

}

Webcam::~Webcam()
{

}





void Webcam::threshold(Mat & src, Mat & dst){
    cv::cvtColor(src,dst , CV_BGR2GRAY);
//    cv::threshold(dst,dst,0,255,CV_THRESH_OTSU);
    cv::adaptiveThreshold(dst,dst,255,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 7, 10);
}


void Webcam::contours(Mat & src, Mat & dst) {
    int largest_area=0;
    int largest_contour_index=0;
    vector<Vec4i> hierarchy;
    vector<vector<Point> > contours;
    vector<Point> contours_approx;
    vector<Point> shape;

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
//    cv::HoughLinesP(dst, lines, 1, CV_PI/720, 75, 40, 10 );
//    std::sort(lines.begin(), lines.end(), sortByLength);



    cv::cvtColor(dst,dst , CV_GRAY2RGB);
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

    // The next two lines must be changed as Freenect::Freenect
    // isn't a template but the method createDevice:
    // Freenect::Freenect<MyFreenectDevice> freenect;
    // MyFreenectDevice& device = freenect.createDevice(0);
    // by these two lines:

    Freenect::Freenect freenect;
    MyFreenectDevice& device = freenect.createDevice<MyFreenectDevice>(0);

    namedWindow("rgb",CV_WINDOW_AUTOSIZE);
    namedWindow("processed",CV_WINDOW_AUTOSIZE);
    device.startVideo();
    while (!die) {
        device.getVideo(rgbMat);

        cv::imshow("rgb", rgbMat);
        threshold(rgbMat, thresholdedMat);
        contours(thresholdedMat, contoursMat);
        cv::imshow("processed", contoursMat);

        char k = cvWaitKey(5);
        if( k == 27 ){
            cvDestroyWindow("rgb");
            cvDestroyWindow("processed");
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
