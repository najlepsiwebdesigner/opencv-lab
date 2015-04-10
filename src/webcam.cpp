#include "webcam.h"


using namespace std;
using namespace cv;

Webcam::Webcam()
{

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
/*        namedWindow("thresholded",CV_WINDOW_AUTOSIZE);
        namedWindow("processed",CV_WINDOW_AUTOSIZE);
        namedWindow("squares",CV_WINDOW_AUTOSIZE/);*/

        device.startVideo();
        while (!die) {
            device.getVideo(rgbMat);
//            findSquares(rgbMat, squares);
            cv::imshow("rgb", rgbMat);

            char k = cvWaitKey(5);
            if( k == 27 ){
                cvDestroyWindow("rgb");
//                cvDestroyWindow("processed");
//                cvDestroyWindow("thresholded");
//                cvDestroyWindow("squares");
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
    namedWindow("Camera_Output", 1);    //Create window
    VideoCapture capture = VideoCapture(CV_CAP_ANY);  //Capture using any camera connected to your system
    char key;

    Mat frame;

    while(1){ //Create infinte loop for live streaming

        capture.read(frame); //Create image frames from capture
        imshow("Camera_Output", frame);   //Show image frames on created window
        key = cvWaitKey(10);     //Capture Keyboard stroke
        if (char(key) == 27){
            break;      //If you hit ESC key loop will break.
        }
    }
    capture.release();//Release capture.
    destroyWindow("Camera_Output"); //Destroy Window
}
