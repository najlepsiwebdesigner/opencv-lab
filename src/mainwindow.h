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


public:
    MainWindow(QWidget *parent = 0);
    virtual ~MainWindow();
    void drawImage(Mat image);


private:
    QScopedPointer<Ui::MainWindow> ui;
    void redrawImage();

private slots:
    void loadImage();
    void saveImage();
    void showWebcam();
};

#endif
