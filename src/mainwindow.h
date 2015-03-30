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
#include "batchwindow.h"
#include <QDropEvent>
#include <QUrl>
#include <QDebug>
#include <QList>
#include <QMimeData>
#include <QMessageBox>
#include <QStringList>
#include <QStringListModel>
#include <QModelIndex>


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

protected:
    void dragEnterEvent(QDragEnterEvent *e);
    void dropEvent(QDropEvent *e);

private:

    QStringListModel *operationsModel;

    void redrawImage();
    void fitImage(const Mat& src,Mat& dst, float destWidth, float destHeight);
    void loadLocalImage(QString fileName);
    batchWindow *batchWin;

    bool thresholdWindow = false;
    bool squaresWindow = false;
    bool linesWindow = false;
    bool equalizedWindow = false;

private slots:
    void loadImage();
    void saveImage();
    void showWebcam();
    void showThreshold();
    void showSquares();
    void showLines();
    void showEqualized();
    void showBatchWindow();
    void executeOperation();
};

#endif
