#include "mainwindow.h"
#include "ui_mainwindow.h"

using namespace std;
using namespace cv;


MainWindow::MainWindow(QWidget *parent)
: QMainWindow(parent), ui(new Ui::MainWindow)
{
    setAcceptDrops(true);
    ui->setupUi(this);
    connect(ui->loadPicture, SIGNAL(clicked()), this, SLOT(loadImage()));
    connect(ui->savePicture, SIGNAL(clicked()), this, SLOT(saveImage()));
    connect(ui->showWebcam, SIGNAL(clicked()), this, SLOT(showWebcam()));
    connect(ui->thresholdButton, SIGNAL(clicked()), this, SLOT(showThreshold()));
    connect(ui->showSquares, SIGNAL(clicked()), this, SLOT(showSquares()));
    connect(ui->showLines, SIGNAL(clicked()), this, SLOT(showLines()));
    connect(ui->showEqualized, SIGNAL(clicked()), this, SLOT(showEqualized()));
    connect(ui->batchWindow, SIGNAL(clicked()), this, SLOT(showBatchWindow()));
}



MainWindow::~MainWindow()
{
}



void MainWindow::showBatchWindow() {
    // modal window approach
    batchWindow window;
    window.setModal(true);
    window.exec();

//    batchWin = new batchWindow(this);
//    batchWin->show();
}





void MainWindow::dropEvent(QDropEvent *ev)
{
    QList<QUrl> urls = ev->mimeData()->urls();
    QString filename;

    if (urls.length() == 1){
        QUrl url = urls.first();
        filename = QString(url.toString());
        filename.replace(QString("file://"), QString(""));

        loadLocalImage(filename);
    } else {
        QMessageBox msgBox;
        msgBox.setText("Only one file is supported for drag and drop loading!");
        msgBox.exec();

//        foreach(QUrl url, urls)
//        {

//        }

    }
}

void MainWindow::dragEnterEvent(QDragEnterEvent *ev)
{
    ev->accept();
}




void MainWindow::showWebcam() {
    Webcam cam;
    cam.showRGB();
}


void MainWindow::showThreshold() {
    if (this->curImage.empty()) {
        cout << "No image!" << endl;
        return;
    }

    Mat src = this->curImage;
    Mat dst(Size(640,480),CV_8UC3,Scalar(0));
    doThreshold(src,dst);

    if (thresholdWindow){
        cvDestroyWindow("threshold");
        thresholdWindow = false;
    } else {
        namedWindow("threshold",CV_WINDOW_AUTOSIZE);
        imshow("threshold", dst);
        thresholdWindow = true;
    }


//    bool die(false);
//    while (!die) {
//        namedWindow("threshold",CV_WINDOW_AUTOSIZE);
//        imshow("threshold", dst);

//        char k = cvWaitKey(5);
//        if( k == 27 ){
//            cvDestroyWindow("threshold");
//            break;
//        }
////        if( k == 8 ) {
////            std::ostringstream file;
////            file << filename << i_snap << suffix;
////            cv::imwrite(file.str(),dst);
////            i_snap++;
////        }
//    }
}


void MainWindow::showSquares() {
    if (this->curImage.empty()) {
        cout << "No image!" << endl;
        return;
    }

    Mat src = this->curImage;
    Mat dst;
    vector<vector<Point> > squares;
    cvtColor(src, dst, CV_BGR2RGB);
    findSquares(dst, squares);
    drawSquares(dst, squares);

    if (squaresWindow){
        cvDestroyWindow("squares");
        squaresWindow = false;
    } else {
        namedWindow("squares",CV_WINDOW_AUTOSIZE);
        imshow("squares", dst);
        squaresWindow = true;
    }
}




void MainWindow::showLines() {
    if (this->curImage.empty()) {
        cout << "No image!" << endl;
        return;
    }

    Mat src = this->curImage;
    Mat dst;
    vector<vector<Point> > squares;

    cvtColor(src, dst, CV_BGR2RGB);
    doThreshold(dst,src);
    doLines(src,dst);

    if (linesWindow){
        cvDestroyWindow("lines");
        linesWindow = false;
    } else {
        namedWindow("lines",CV_WINDOW_AUTOSIZE);
        imshow("lines", dst);
        linesWindow = true;
    }
}



void MainWindow::showEqualized(){
    if (this->curImage.empty()) {
        cout << "No image!" << endl;
        return;
    }

    Mat src = this->curImage;
    cvtColor(src, src, CV_BGR2RGB);
    src = equalizeIntensity(src);

    if (equalizedWindow){
        cvDestroyWindow("equalized");
        equalizedWindow = false;
    } else {
        namedWindow("equalized",CV_WINDOW_AUTOSIZE);
        imshow("equalized", src);
        equalizedWindow = true;
    }
}








void MainWindow::saveImage(){
    if (this->curImage.empty()) {
        cout << "No image to save!" << endl;
        return;
    }

    QString fileName = QFileDialog::getSaveFileName(this,tr("Save File"),".",tr("Images (*.png *.jpg)"));

    if (fileName.length() < 1) return;

    cvtColor(this->curImage, this->curImage, CV_BGR2RGB);
    imwrite(fileName.toStdString(), this->curImage);
    cout << "File saved!" << endl;
}


void MainWindow::fitImage(const Mat& src,Mat& dst, float destWidth, float destHeight) {
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



void MainWindow::loadImage(){
    // open file dialog
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"),".",tr("Images (*.png *.jpg)"));
    loadLocalImage(fileName);
}


void MainWindow::loadLocalImage(QString fileName) {
    if (fileName.length() < 1) return;

    cout << "Loaded image name: "<< fileName.toStdString() << endl;

    // open image as CV image
    Mat src = imread(fileName.toStdString(),CV_LOAD_IMAGE_COLOR);
    cvtColor(src, src, CV_BGR2RGB);
    Mat dst(Size(640,480),CV_8UC3,Scalar(0));

    fitImage(src, dst, 640, 480);

    // display image
    this->curImage = dst;
    redrawImage();
}




void MainWindow::drawImage(Mat src){
    QPixmap pix = QPixmap::fromImage(QImage((unsigned char*) src.data, src.cols, src.rows, QImage::Format_RGB888));
    ui->imageDisplay->setPixmap(pix);
}


void MainWindow::redrawImage(){
    Mat src = this->curImage;
    // convert color models
    QPixmap pix = QPixmap::fromImage(QImage((unsigned char*) src.data, src.cols, src.rows, QImage::Format_RGB888));
    ui->imageDisplay->setPixmap(pix);
}

