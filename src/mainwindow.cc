#include "mainwindow.h"
#include "ui_mainwindow.h"

using namespace std;
using namespace cv;

MainWindow::MainWindow(QWidget *parent)
: QMainWindow(parent), ui(new Ui::MainWindow)
{
    setAcceptDrops(true);
    ui->setupUi(this);

    List << "threshold"
         << "equalize"
         << "squares"
         << "lines"
         << "HSV"
         << "resizeDownUp"
         << "adaptive bilateral"
         << "kMeans";

    operationsMap.insert(FunctionMap::value_type("equalize",MainWindow::equalize));
    operationsMap.insert(FunctionMap::value_type("lines",MainWindow::lines));
    operationsMap.insert(FunctionMap::value_type("threshold",MainWindow::threshold));
    operationsMap.insert(FunctionMap::value_type("squares",MainWindow::squares));
    operationsMap.insert(FunctionMap::value_type("HSV",MainWindow::hsv));
    operationsMap.insert(FunctionMap::value_type("resizeDownUp",MainWindow::resizedownup));
    operationsMap.insert(FunctionMap::value_type("adaptive bilateral",MainWindow::adaptiveBilateralFilter));
    operationsMap.insert(FunctionMap::value_type("kMeans",MainWindow::kMeans));

    operationsModel = new QStringListModel(this);
    operationsModel->setStringList(List);
    ui->operationsList->setModel(operationsModel);
    QModelIndex initialCellIndex = operationsModel->index(0);
    ui->operationsList->setCurrentIndex(initialCellIndex);

    connect(ui->loadPicture, SIGNAL(clicked()), this, SLOT(openImage()));
    connect(ui->showWebcam, SIGNAL(clicked()), this, SLOT(showWebcam()));
    connect(ui->executeButton, SIGNAL(clicked()), this, SLOT(executeOperation()));
    connect(ui->clearImages, SIGNAL(clicked()), this, SLOT(clearImages()));
    connect(ui->reload, SIGNAL(clicked()), this, SLOT(reloadImages()));
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
    Mat src = imread(filename,CV_LOAD_IMAGE_COLOR);
    cvtColor(src, src, CV_BGR2RGB);
    Mat dst(Size(640,480),CV_8UC3,Scalar(0));
    fitImage(src, dst, 640, 480);
    loadedImages.push_back(dst);
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
    cout << "test" << endl;
}


void MainWindow::showWebcam() {
    Webcam cam;
    cam.showRGB();
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

//void MainWindow::saveImage(){
//    if (this->curImage.empty()) {
//        cout << "No image to save!" << endl;
//        return;
//    }

//    QString fileName = QFileDialog::getSaveFileName(this,tr("Save File"),".",tr("Images (*.png *.jpg)"));

//    if (fileName.length() < 1) return;

//    cvtColor(this->curImage, this->curImage, CV_BGR2RGB);
//    imwrite(fileName.toStdString(), this->curImage);
//    cout << "File saved!" << endl;
//}


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







void MainWindow::openImage(){
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"),".",tr("Images (*.png *.jpg)"));
    loadImage(fileName.toStdString());
}


void MainWindow::redrawImages() {

    cout << loadedImages.size() << endl;
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




string MainWindow::getSelectedOperation() {
    int row = ui->operationsList->currentIndex().row();
    FunctionMap::const_iterator call;
    return List[row].toStdString();
}


void MainWindow::executeOperation() {
    if (this->loadedImages.size() < 1) {
        cout << "No images to process!" << endl;
        return;
    }

    string functionName = getSelectedOperation();

    FunctionMap::const_iterator call;
    call = operationsMap.find(functionName);

    for (int i = 0; i < loadedImages.size(); i++){
        if (call != operationsMap.end())
           (*call).second(loadedImages[i]);
        else
           cout << "Unknown call requested" << endl;
    }

    redrawImages();
}





void MainWindow::equalize(Mat & image) {
    image = equalizeIntensity(image);
}

void MainWindow::lines(Mat & image) {
    Mat src = image;
    Mat dst;
    cv::cvtColor(src,dst , CV_BGR2GRAY);
    cv::GaussianBlur(dst, dst, Size( 7, 7) ,7,7);
//    cv::threshold(dst,dst,0,255,THRESH_TOZERO + CV_THRESH_OTSU);
//    cv::threshold(dst,dst,0,255,CV_THRESH_BINARY);

    doLines(dst,src);
//    cv::cvtColor(src,image, CV_GRAY2RGB);
    image = src;
}

void MainWindow::threshold(Mat & image) {
    Mat dst;
    cv::cvtColor(image,dst, CV_RGB2GRAY);
    cv::GaussianBlur(dst, dst, Size( 7, 7) ,7,7);
//    cv::threshold(dst,dst,0,255,THRESH_TOZERO + CV_THRESH_OTSU);
//    cv::threshold(dst,dst,0,255,CV_THRESH_BINARY);
    cv::adaptiveThreshold(dst, dst, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 7, 0);
    cv::cvtColor(dst,image, CV_GRAY2RGB);
}

void MainWindow::squares(Mat & image) {
    Mat src = image;
    Mat dst;
    vector<vector<Point> > squares;
    cvtColor(src, dst, CV_BGR2RGB);
    findSquares(dst, squares);
    drawSquares(dst, squares);
    cvtColor(dst, dst, CV_RGB2BGR);
    image = dst;
}
void MainWindow::hsv(Mat & src) {
    cv::Mat hsv;
    cv::cvtColor(src, hsv, CV_BGR2HSV);
    cv::Mat hue(src.size(), CV_8U);
    //the third arguments are two number a pair, (0, 0) means copy the data of channels 0(hsv) to channels 0(hue)
    cv::mixChannels(hsv, hue, {0, 0});
    cv::Mat otsuMat;
    cv::adaptiveThreshold(hue,otsuMat,255, CV_ADAPTIVE_THRESH_MEAN_C , CV_THRESH_BINARY, 3, 0);
    cv::cvtColor(otsuMat,src, CV_GRAY2RGB);
}


void MainWindow::resizedownup(Mat & image){
    Mat small;
    cv::resize(image, small,Size(320,200),0,0);
    cv::medianBlur(small, small, 9);
    cv::resize(small, image, Size(image.cols,image.rows));
}


void MainWindow::adaptiveBilateralFilter(Mat & image){
    Mat dst;
//    cv::bilateralFilter ( image, dst, 15, 100, 35 );
    cv::adaptiveBilateralFilter(image, dst, Size(3,3),3);
    image = dst;
}

void MainWindow::kMeans(Mat &src) {
    cv::Mat samples(src.total(), 3, CV_32F);
       auto samples_ptr = samples.ptr<float>(0);
       for( int row = 0; row != src.rows; ++row){
           auto src_begin = src.ptr<uchar>(row);
           auto src_end = src_begin + src.cols * src.channels();
           //auto samples_ptr = samples.ptr<float>(row * src.cols);
           while(src_begin != src_end){
               samples_ptr[0] = src_begin[0];
               samples_ptr[1] = src_begin[1];
               samples_ptr[2] = src_begin[2];
               samples_ptr += 3; src_begin +=3;
           }
       }

       //step 2 : apply kmeans to find labels and centers
       int clusterCount = 3;
       cv::Mat labels;
       int attempts = 5;
       cv::Mat centers;
       cv::kmeans(samples, clusterCount, labels,
                  cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,
                                   10, 0.01),
                  attempts, cv::KMEANS_PP_CENTERS, centers);

       //step 3 : map the centers to the output
       cv::Mat new_image(src.size(), src.type());
       for( int row = 0; row != src.rows; ++row){
           auto new_image_begin = new_image.ptr<uchar>(row);
           auto new_image_end = new_image_begin + new_image.cols * 3;
           auto labels_ptr = labels.ptr<int>(row * src.cols);

           while(new_image_begin != new_image_end){
               int const cluster_idx = *labels_ptr;
               auto centers_ptr = centers.ptr<float>(cluster_idx);
               new_image_begin[0] = centers_ptr[0];
               new_image_begin[1] = centers_ptr[1];
               new_image_begin[2] = centers_ptr[2];
               new_image_begin += 3; ++labels_ptr;
           }
       }
       cv::Mat binary;
       cv::Canny(new_image, binary, 30, 90);
       cv::cvtColor(binary,src, CV_GRAY2RGB);
}
