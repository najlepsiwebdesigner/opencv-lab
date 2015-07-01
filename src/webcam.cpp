#include "webcam.h"
#include "idocr.h"

using namespace std;
using namespace cv;

Webcam::Webcam() {

}


Webcam::~Webcam()
{

}


void Webcam::showKinectRGB() {
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

    try {
        Freenect::Freenect freenect;
        MyFreenectDevice& device = freenect.createDevice<MyFreenectDevice>(0);

        namedWindow("rgb",CV_WINDOW_AUTOSIZE);

        device.startVideo();
        while (!die) {
            device.getVideo(rgbMat);
            cv::imshow("rgb", rgbMat);

            char k = cvWaitKey(5);
            if( k == 27 ){
                cvDestroyWindow("rgb");
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
    catch (exception& e) {
        QMessageBox msgBox;
        msgBox.setText(e.what());
        msgBox.exec();
    }
}


void Webcam::showRGB() {
    namedWindow("Camera_Output", 1);
    int cap = CV_CAP_ANY;
//    if (cap == 0){
//        cap = 1;
//    }
    VideoCapture capture = VideoCapture(cap);
    char key;
    Mat frame;

    while(1){
        string timestamp = boost::posix_time::to_iso_string(boost::posix_time::microsec_clock::local_time()) + ".jpg";
        capture.read(frame);

        string filename = getScreenshotPath() + timestamp;

        idOCR::process(frame);

        imshow("Camera", frame);

        key = cvWaitKey(10);

        if (char(key) == 27){
            break;      // esc
        } else if (char(key) == 32) { // space
            idOCR::saveImage(filename, frame);
            idOCR::maskCutOut(frame, "front.png");
        }

    }
    capture.release();
    cv::destroyAllWindows();
}



void Webcam::setScreenshotPath(std::string path) {
    this->screenshotPath = path;
}

std::string Webcam::getScreenshotPath() {
    if (this->screenshotPath.length() > 0){
       return this->screenshotPath + "/";
    }
    return "";
}
