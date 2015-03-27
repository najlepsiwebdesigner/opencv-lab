#ifndef mainwindow_h
#define mainwindow_h

#include <QMainWindow>
#include <QScopedPointer>
#include <QFileDialog>
#include <iostream>
#include "helpers.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "webcam.h"


using namespace cv;
using namespace std;

namespace Ui
{
    class MainWindow;
}

class MainWindow : public QMainWindow {
    Q_OBJECT
    Mat curImage;
    QScopedPointer<Ui::MainWindow> ui;

public:
    MainWindow(QWidget *parent = 0);
    virtual ~MainWindow();
    void drawImage(Mat image);

private:
    void redrawImage();
    void fitImage(const Mat& src,Mat& dst, float destWidth, float destHeight);
    bool thresholdWindow = false;
    bool squaresWindow = false;
    bool linesWindow = false;

private slots:
    void loadImage();
    void saveImage();
    void showWebcam();
    void showThreshold();
    void showSquares();
    void showLines();
};

#endif
