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

#include <map>
//#include <boost/function.hpp>


using namespace cv;
using namespace std;



namespace Ui
{
    class MainWindow;

}

typedef void (*FunctionPointer)(Mat & image);
typedef map<string,FunctionPointer> FunctionMap;


class MainWindow : public QMainWindow {
    Q_OBJECT
    Mat curImage;
    QScopedPointer<Ui::MainWindow> ui;
    FunctionMap	operationsMap;

    void static equalize(Mat & image);
    void static lines(Mat & image);
    void static threshold(Mat & image);
    void static squares(Mat & image);

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
