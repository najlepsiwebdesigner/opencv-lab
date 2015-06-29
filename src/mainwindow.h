#ifndef mainwindow_h
#define mainwindow_h

// std
#include <iostream>
#include <string.h>
#include <map>

// app
#include "webcam.h"
#include "imageoperations.h"
#include "idocr.h"

// cv
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"

// boost
#include <boost/filesystem.hpp>

// qt
#include <QMainWindow>
#include <QScopedPointer>
#include <QFileDialog>
#include <QDropEvent>
#include <QUrl>
#include <QDebug>
#include <QList>
#include <QMimeData>
#include <QMessageBox>
#include <QStringList>
#include <QStringListModel>
#include <QModelIndex>
#include <QScrollBar>
#include <QLabel>
#include <QMessageBox>
#include <QListWidgetItem>

using namespace cv;
using namespace std;
using namespace boost::filesystem;

namespace Ui
{
    class MainWindow;

}

typedef void (*FunctionPointer)(Mat & image);
typedef map<string,FunctionPointer> FunctionMap;


class MainWindow : public QMainWindow {
    Q_OBJECT
    Mat curImage;

    vector<Mat> loadedImages;

    QScopedPointer<Ui::MainWindow> ui;
    FunctionMap	operationsMap;
    QStringList List;
    QList<QUrl> urls;
    QList<QLabel *> imageLabels;

    string  getSelectedOperation();

public:
    MainWindow(QWidget *parent = 0);
    virtual ~MainWindow();

protected:
    void dragEnterEvent(QDragEnterEvent *e);
    void dropEvent(QDropEvent *e);

private:
    QStringListModel *operationsModel;

    void redrawImages();
    void loadImage(string filename);
    void saveImage(string fileName, const Mat &image);


private slots:
    void openImage();
    void showWebcam();
    void showKinect();
    void clearImages();
    void executeOperation();
    void reloadImages();
    void saveImages();
    void SIFT();
    void itemDblClicked(QModelIndex);
    void filtering();
    void locating();
    void cutAndPerspective();
};


QImage cvMatToQImage(const cv::Mat &inMat);
QPixmap cvMatToQPixmap(const cv::Mat &inMat);


#endif
