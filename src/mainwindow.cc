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
         << "thresholdAdaptive"
         << "equalize"
         << "squares"
         << "HSV"
         << "resizeDownUp"
         << "adaptive bilateral"
         << "kMeans"
         << "Sobel"
         << "Gaussian"
         << "Canny"
         << "Erode"
         << "Dilate"
         << "BiggestContour"
         << "ConvexHull";

    operationsMap.insert(FunctionMap::value_type("equalize",ImageOperations::equalize));
    operationsMap.insert(FunctionMap::value_type("lines",ImageOperations::lines));
    operationsMap.insert(FunctionMap::value_type("thresholdGray",ImageOperations::thresholdGray));
    operationsMap.insert(FunctionMap::value_type("thresholdBinary",ImageOperations::thresholdBinary));
    operationsMap.insert(FunctionMap::value_type("thresholdAdaptive",ImageOperations::thresholdAdaptive));
    operationsMap.insert(FunctionMap::value_type("squares",ImageOperations::squares));
    operationsMap.insert(FunctionMap::value_type("HSV",ImageOperations::hsv));
    operationsMap.insert(FunctionMap::value_type("resizeDownUp",ImageOperations::resizedownup));
    operationsMap.insert(FunctionMap::value_type("adaptive bilateral",ImageOperations::adaptiveBilateralFilter));
    operationsMap.insert(FunctionMap::value_type("kMeans",ImageOperations::kMeans));
    operationsMap.insert(FunctionMap::value_type("Sobel",ImageOperations::sobel));
    operationsMap.insert(FunctionMap::value_type("Gaussian",ImageOperations::gaussian));
    operationsMap.insert(FunctionMap::value_type("Canny",ImageOperations::canny));
    operationsMap.insert(FunctionMap::value_type("Erode",ImageOperations::erosion));
    operationsMap.insert(FunctionMap::value_type("Dilate",ImageOperations::dilation));
    operationsMap.insert(FunctionMap::value_type("BiggestContour",ImageOperations::biggestContour));
    operationsMap.insert(FunctionMap::value_type("ConvexHull",ImageOperations::convexH));

    operationsModel = new QStringListModel(this);
    operationsModel->setStringList(List);
    ui->operationsList->setModel(operationsModel);
    QModelIndex initialCellIndex = operationsModel->index(0);
    ui->operationsList->setCurrentIndex(initialCellIndex);
    ui->operationsList->setEditTriggers(QAbstractItemView::NoEditTriggers);

    connect(ui->loadPicture, SIGNAL(clicked()), this, SLOT(openImage()));
    connect(ui->showWebcam, SIGNAL(clicked()), this, SLOT(showWebcam()));
    connect(ui->showKinect, SIGNAL(clicked()), this, SLOT(showKinect()));
    connect(ui->executeButton, SIGNAL(clicked()), this, SLOT(executeOperation()));
    connect(ui->clearImages, SIGNAL(clicked()), this, SLOT(clearImages()));
    connect(ui->reload, SIGNAL(clicked()), this, SLOT(reloadImages()));
    connect(ui->savePictures, SIGNAL(clicked()), this, SLOT(saveImages()));

    connect(ui->SIFT , SIGNAL(clicked()), this, SLOT(SIFT()));
    connect(ui->filtering, SIGNAL(clicked()), this, SLOT(filtering()));

    connect(ui->operationsList, SIGNAL(doubleClicked(QModelIndex)),this,SLOT(itemDblClicked(QModelIndex)));
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
//        cvtColor(src, src, CV_BGR2RGB);
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
//            cvtColor(loadedImages[i+j],loadedImages[i+j],CV_BGR2RGB);
            pix = cvMatToQPixmap(loadedImages[i+j]);

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


/*
 * utils from
 * source: http://asmaloney.com/2013/11/code/converting-between-cvmat-and-qimage-or-qpixmap/
*/

QImage cvMatToQImage( const cv::Mat &inMat )
{
    switch ( inMat.type() )
    {
       // 8-bit, 4 channel
       case CV_8UC4:
       {
          QImage image( inMat.data, inMat.cols, inMat.rows, inMat.step, QImage::Format_RGB32 );

          return image;
       }

       // 8-bit, 3 channel
       case CV_8UC3:
       {
          QImage image( inMat.data, inMat.cols, inMat.rows, inMat.step, QImage::Format_RGB888 );

          return image.rgbSwapped();
       }

       // 8-bit, 1 channel
       case CV_8UC1:
       {
          static QVector<QRgb>  sColorTable;

          // only create our color table once
          if ( sColorTable.isEmpty() )
          {
             for ( int i = 0; i < 256; ++i )
                sColorTable.push_back( qRgb( i, i, i ) );
          }

          QImage image( inMat.data, inMat.cols, inMat.rows, inMat.step, QImage::Format_Indexed8 );

          image.setColorTable( sColorTable );

          return image;
       }

       default:
          qWarning() << "ASM::cvMatToQImage() - cv::Mat image type not handled in switch:" << inMat.type();
          break;
    }

    return QImage();
}

QPixmap cvMatToQPixmap( const cv::Mat &inMat )
{
    return QPixmap::fromImage( cvMatToQImage( inMat ) );
}

void MainWindow::SIFT() {

    if (loadedImages.size() == 2) {

    Mat img_1 = loadedImages[0];
    Mat img_2 = loadedImages[1];

//    imshow("1",image1);
//    imshow("2",image2);

    if( !img_1.data || !img_2.data )
      { std::cout<< " --(!) Error reading images " << std::endl;return; }

      //-- Step 1: Detect the keypoints using SURF Detector
      int minHessian = 400;

      SurfFeatureDetector detector( minHessian );

      std::vector<KeyPoint> keypoints_1, keypoints_2;

      detector.detect( img_1, keypoints_1 );
      detector.detect( img_2, keypoints_2 );

      //-- Step 2: Calculate descriptors (feature vectors)
      SurfDescriptorExtractor extractor;

      Mat descriptors_1, descriptors_2;

      extractor.compute( img_1, keypoints_1, descriptors_1 );
      extractor.compute( img_2, keypoints_2, descriptors_2 );

      //-- Step 3: Matching descriptor vectors using FLANN matcher
      FlannBasedMatcher matcher;
      std::vector< DMatch > matches;
      matcher.match( descriptors_1, descriptors_2, matches );

      double max_dist = 0; double min_dist = 100;

      //-- Quick calculation of max and min distances between keypoints
      for( int i = 0; i < descriptors_1.rows; i++ )
      { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
      }

      printf("-- Max dist : %f \n", max_dist );
      printf("-- Min dist : %f \n", min_dist );

      //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
      //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
      //-- small)
      //-- PS.- radiusMatch can also be used here.
      std::vector< DMatch > good_matches;

      for( int i = 0; i < descriptors_1.rows; i++ )
      { if( matches[i].distance <= max(2*min_dist, 0.02) )
        { good_matches.push_back( matches[i]); }
      }

      //-- Draw only "good" matches
      Mat img_matches;
      drawMatches( img_1, keypoints_1, img_2, keypoints_2,
                   good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                   vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


      for( int i = 0; i < (int)good_matches.size(); i++ )
      { printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }

      char key;

      while(1){ //Create infinte loop for live streaming

          //-- Show detected matches
          imshow( "Good Matches", img_matches );

          key = cvWaitKey(10);     //Capture Keyboard stroke
          if (char(key) == 27){
              break;      //If you hit ESC key loop will break.
          }
      }

      destroyAllWindows();

      return;

    }
    else {
        QMessageBox msgBox;
        msgBox.setText("SIFT is supported only for 2 images at once!");
        msgBox.exec();
        return;
    }
}


void MainWindow::itemDblClicked(QModelIndex) {
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



void MainWindow::filtering() {

    if (this->loadedImages.size() < 1) {
        QMessageBox msgBox;
        msgBox.setText("No images to process!");
        msgBox.exec();
        return;
    }


    for (int i = 0; i < loadedImages.size(); i++){
        Mat image = loadedImages[i].clone();

        ImageOperations::gaussian(image);
        ImageOperations::resizedownup(image);
        ImageOperations::resizedownup(image);
        ImageOperations::dilation(image);
        ImageOperations::dilation(image);
        ImageOperations::dilation(image);
        ImageOperations::erosion(image);
        ImageOperations::erosion(image);
        ImageOperations::erosion(image);
        ImageOperations::sobel(image);
        ImageOperations::thresholdBinary(image);
        ImageOperations::dilation(image);
//        ImageOperations::dilation(image);
        ImageOperations::dilation(image);
        ImageOperations::erosion(image);
//        ImageOperations::erosion(image);
        ImageOperations::erosion(image);
        ImageOperations::biggestContour(image);
//        ImageOperations::erosion(image);
        ImageOperations::canny(image);
        ImageOperations::lines(image);

        double alpha = 0.5; double beta;
        beta = ( 1.0 - alpha );
        addWeighted(loadedImages[i] , alpha, image, beta, 0.0, loadedImages[i]);
//        loadedImages[i] = image.clone();
    }

    redrawImages();
}
