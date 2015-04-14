#include "mainwindow.h"
#include "ui_mainwindow.h"

using namespace std;
using namespace cv;



MainWindow::MainWindow(QWidget *parent)
: QMainWindow(parent), ui(new Ui::MainWindow)
{
    setAcceptDrops(true);
    ui->setupUi(this);

    List << "lines"
         << "thresholdGray"
         << "thresholdBinary"
         << "equalize"
         << "squares"
         << "HSV"
         << "resizeDownUp"
         << "adaptive bilateral"
         << "kMeans"
         << "Sobel";

    operationsMap.insert(FunctionMap::value_type("equalize",ImageOperations::equalize));
    operationsMap.insert(FunctionMap::value_type("lines",ImageOperations::lines));
    operationsMap.insert(FunctionMap::value_type("thresholdGray",ImageOperations::thresholdGray));
    operationsMap.insert(FunctionMap::value_type("thresholdBinary",ImageOperations::thresholdBinary));
    operationsMap.insert(FunctionMap::value_type("squares",ImageOperations::squares));
    operationsMap.insert(FunctionMap::value_type("HSV",ImageOperations::hsv));
    operationsMap.insert(FunctionMap::value_type("resizeDownUp",ImageOperations::resizedownup));
    operationsMap.insert(FunctionMap::value_type("adaptive bilateral",ImageOperations::adaptiveBilateralFilter));
    operationsMap.insert(FunctionMap::value_type("kMeans",ImageOperations::kMeans));
    operationsMap.insert(FunctionMap::value_type("Sobel",ImageOperations::sobel));

    operationsModel = new QStringListModel(this);
    operationsModel->setStringList(List);
    ui->operationsList->setModel(operationsModel);
    QModelIndex initialCellIndex = operationsModel->index(0);
    ui->operationsList->setCurrentIndex(initialCellIndex);

    connect(ui->loadPicture, SIGNAL(clicked()), this, SLOT(openImage()));
    connect(ui->showWebcam, SIGNAL(clicked()), this, SLOT(showWebcam()));
    connect(ui->showKinect, SIGNAL(clicked()), this, SLOT(showKinect()));
    connect(ui->executeButton, SIGNAL(clicked()), this, SLOT(executeOperation()));
    connect(ui->clearImages, SIGNAL(clicked()), this, SLOT(clearImages()));
    connect(ui->reload, SIGNAL(clicked()), this, SLOT(reloadImages()));
    connect(ui->savePictures, SIGNAL(clicked()), this, SLOT(saveImages()));
}

MainWindow::~MainWindow(){}

void MainWindow::dragEnterEvent(QDragEnterEvent *ev)
{
    ev->accept();
}

void MainWindow::dropEvent(QDropEvent *ev) {
    urls = ev->mimeData()->urls();
    QString filename;

    foreach(QUrl url, urls) {
        filename = QString(url.toString());
        filename.replace(QString("file://"), QString(""));
        loadImage(filename.toStdString());
    }

    redrawImages();
}

void MainWindow::loadImage(string filename) {
    if (exists(filename)){
        Mat src = imread(filename,CV_LOAD_IMAGE_COLOR);
        cvtColor(src, src, CV_BGR2RGB);
        Mat dst(Size(640,480),CV_8UC3,Scalar(0));
        ImageOperations::fitImage(src, dst, 640, 480);
        loadedImages.push_back(dst);
    }
}

void MainWindow::reloadImages(){
    clearImages();
    QString filename;

    foreach(QUrl url, urls) {
        filename = QString(url.toString());
        filename.replace(QString("file://"), QString(""));
        loadImage(filename.toStdString());
    }

    redrawImages();
}


void MainWindow::showWebcam() {
    Webcam cam;
    cam.showRGB();
}



void MainWindow::showKinect() {
    Webcam cam;
    cam.showKinectRGB();
}



void MainWindow::clearImages() {
    loadedImages.clear();

    while(ui->imagesLayout->count() > 0){
       QLayoutItem *item = ui->imagesLayout->takeAt(0);
       delete item->widget();
       delete item;
    }

    redrawImages();
}

void MainWindow::saveImage(string fileName, const Mat & image){
    if (fileName.length() < 1) return;

    cvtColor(image, image, CV_BGR2RGB);
    imwrite(fileName, image);
    cout << "File saved!" << endl;
}


void MainWindow::openImage(){
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"),".",tr("Images (*.png *.jpg)"));
    loadImage(fileName.toStdString());
}


void MainWindow::redrawImages() {
    QPixmap pix;

    for (int i = 0; i<loadedImages.size(); i=i+1){
        for (int j = 0; j<1; j++){
            QLabel *picLabel = new QLabel();
            pix = QPixmap::fromImage(QImage((unsigned char*) loadedImages[i+j].data, loadedImages[i+j].cols, loadedImages[i+j].rows, QImage::Format_RGB888));
            picLabel->setPixmap(pix);
            picLabel->setFixedWidth(640);
            picLabel->setFixedHeight(480);
            ui->imagesLayout->addWidget(picLabel,i,j);
            imageLabels << picLabel;
        }
    }

//    QLabel *label;
//    for (QLabel *label : imageLabels) {
//        connect(label, SIGNAL(clicked()), this, SLOT(clearImages()));
//    }

}

void MainWindow::saveImages() {
    if (this->loadedImages.size() < 1) {
        QMessageBox msgBox;
        msgBox.setText("No images to process!");
        msgBox.exec();
        return;
    }

    QString dir = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
                                                "",
                                                QFileDialog::ShowDirsOnly
                                                | QFileDialog::DontResolveSymlinks);
    QString filename;
    for (int i =0; i < urls.size(); i++) {
        filename = QString(urls[i].toString());
        filename.replace(QString("file://"), QString(""));

        string name = filename.toStdString();
        unsigned found = name.find_last_of("/\\");
        name = name.substr(found+1);

        name = dir.toStdString() + '/' + name ;
        saveImage(name, this->loadedImages[i]);
    }
}


string MainWindow::getSelectedOperation() {
    int row = ui->operationsList->currentIndex().row();
    FunctionMap::const_iterator call;
    return List[row].toStdString();
}


void MainWindow::executeOperation() {
    if (this->loadedImages.size() < 1) {
        QMessageBox msgBox;
        msgBox.setText("No images to process!");
        msgBox.exec();
        return;
    }

    string functionName = getSelectedOperation();

    FunctionMap::const_iterator call;
    call = operationsMap.find(functionName);

    for (int i = 0; i < loadedImages.size(); i++){
        if (call != operationsMap.end()) {
           (*call).second(loadedImages[i]);
        }
        else {
            QMessageBox msgBox;
            msgBox.setText("Unknown call requested");
            msgBox.exec();
            redrawImages();
            return;
        }
    }

    redrawImages();
}



