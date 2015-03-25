#include "mainwindow.h"
#include "ui_mainwindow.h"

using namespace std;
using namespace cv;


MainWindow::MainWindow(QWidget *parent)
: QMainWindow(parent), ui(new Ui::MainWindow)
{
//    connect(cam, SIGNAL(processImage(Mat image)), this, SLOT(drawImage(Mat image)));

    ui->setupUi(this);
    connect(ui->loadPicture, SIGNAL(clicked()), this, SLOT(loadImage()));
    connect(ui->savePicture, SIGNAL(clicked()), this, SLOT(saveImage()));
    connect(ui->showWebcam, SIGNAL(clicked()), this, SLOT(showWebcam()));
}



MainWindow::~MainWindow()
{
}


void MainWindow::showWebcam() {
    Webcam cam;
    cam.showRGB();
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


void MainWindow::loadImage(){
    // open file dialog
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"),".",tr("Images (*.png *.jpg)"));
    if (fileName.length() < 1) return;

    cout << "Loaded image name: "<< fileName.toStdString() << endl;

    // open image as CV image
    Mat src = imread(fileName.toStdString(),CV_LOAD_IMAGE_COLOR);

    // display image
    this->curImage = src;
    redrawImage();
}

void MainWindow::drawImage(Mat src){
    cvtColor(src, src, CV_BGR2RGB);
    QPixmap pix = QPixmap::fromImage(QImage((unsigned char*) src.data, src.cols, src.rows, QImage::Format_RGB888));
    ui->imageDisplay->setPixmap(pix);
}


void MainWindow::redrawImage(){
    Mat src = this->curImage;
    // convert color models
    cvtColor(src, src, CV_BGR2RGB);
    QPixmap pix = QPixmap::fromImage(QImage((unsigned char*) src.data, src.cols, src.rows, QImage::Format_RGB888));
    ui->imageDisplay->setPixmap(pix);
}

