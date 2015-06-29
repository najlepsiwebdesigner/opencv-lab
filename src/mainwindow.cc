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
    connect(ui->locating, SIGNAL(clicked()), this, SLOT(locating()));
    connect(ui->cutaAndPerspective, SIGNAL(clicked()), this, SLOT(cutAndPerspective()));

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

        Mat dst(Size(1920,1440),CV_8UC3,Scalar(0));
        ImageOperations::fitImage(src, dst, 1920, 1440);

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

    cv::destroyAllWindows();
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


            Mat img(Size(640,480),CV_8UC3,Scalar(0));
            ImageOperations::fitImage(loadedImages[i+j], img, 640, 480);

            pix = cvMatToQPixmap(img);

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
        ImageOperations::dilation(image);
        ImageOperations::erosion(image);
        ImageOperations::erosion(image);
        ImageOperations::biggestContour(image);
        ImageOperations::canny(image);

        double alpha = 0.5; double beta;
        beta = ( 1.0 - alpha );
        addWeighted(loadedImages[i] , alpha, image, beta, 0.0, loadedImages[i]);
     }

     redrawImages();
}



void MainWindow::locating() {

    if (this->loadedImages.size() < 1) {
        QMessageBox msgBox;
        msgBox.setText("No images to process!");
        msgBox.exec();
        return;
    }


    for (int i = 0; i < loadedImages.size(); i++){
//        Mat image = loadedImages[i].clone();


        Mat image(Size(640,480),CV_8UC3,Scalar(0));
        ImageOperations::fitImage(loadedImages[i], image, 640, 480);

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
        ImageOperations::dilation(image);
        ImageOperations::erosion(image);
        ImageOperations::erosion(image);
        ImageOperations::biggestContour(image);
        ImageOperations::canny(image);


            Mat dst;
            cv::cvtColor(image,dst , CV_RGB2GRAY);

            std::vector<cv::Vec4i> lines;
            cv::HoughLinesP(dst, lines, 1, CV_PI/360,50,50, 10);


        cv::cvtColor(dst,dst , CV_GRAY2RGB);
            // draw lines around image
            std::vector<cv::Vec4i> myLines;

            cv::Vec4i leftLine;leftLine[0] = 0;leftLine[1] = 0;leftLine[2] = 0;leftLine[3] = dst.rows;
            myLines.push_back(leftLine);
            cv::Vec4i rightLine;
            rightLine[0] = dst.cols;rightLine[1] = 0;rightLine[2] = dst.cols;rightLine[3] = dst.rows;
            myLines.push_back(rightLine);
            cv::Vec4i topLine;
            topLine[0] = dst.cols;topLine[1] = 0;topLine[2] = 0;topLine[3] = 0;
            myLines.push_back(topLine);
            cv::Vec4i bottomLine;
            bottomLine[0] = dst.cols;bottomLine[1] = dst.rows;bottomLine[2] = 0;bottomLine[3] = dst.rows;
            myLines.push_back(bottomLine);

            // expand lines to borders of the image - we will get intersections with image borders easily
            for (int i = 0; i < lines.size(); i++)
            {
                std::vector<cv::Point2f> lineIntersections;
                for (int j = 0; j < myLines.size(); j++)
                {
                    cv::Vec4i v = lines[i];
                    Point2f intersection;

                    bool has_intersection = ImageOperations::getIntersectionPoint(
                        Point(lines[i][0],lines[i][1]),
                        Point(lines[i][2], lines[i][3]),
                        Point(myLines[j][0],myLines[j][1]),
                        Point(myLines[j][2], myLines[j][3]),
                        intersection);

                    if (has_intersection
                        && intersection.x >= 0
                        && intersection.y >= 0
                        && intersection.x <= dst.cols
                        && intersection.y <= dst.rows){
                        lineIntersections.push_back(intersection);
                    }
                }

                if (lineIntersections.size() > 0) {
                    lines[i][0] = lineIntersections[0].x;
                    lines[i][1] = lineIntersections[0].y;
                    lines[i][2] = lineIntersections[1].x;
                    lines[i][3] = lineIntersections[1].y;
                }
            }



            struct LineCluster {
                int sumX1;
                int sumY1;
                int sumX2;
                int sumY2;
                int count;
            };

            vector<LineCluster> clusters;

            int distanceThreshold = round(image.cols/10);
            double angleThreshold = 0.80;

                // create first group
                LineCluster cluster;
                cluster.sumX1 = lines[0][0];
                cluster.sumY1 = lines[0][1];
                cluster.sumX2 = lines[0][2];
                cluster.sumY2 = lines[0][3];
                cluster.count = 1;
                clusters.push_back(cluster);

                // loop through rest of groups
                for (int i = 1; i < lines.size(); i++) {
                    bool in_some_cluster = false;

                    for (int j = 0; j < clusters.size(); j++) {

                         int cluster_x1 = clusters[j].sumX1/clusters[j].count;
                         int cluster_y1 = clusters[j].sumY1/clusters[j].count;
                         int cluster_x2 = clusters[j].sumX2/clusters[j].count;
                         int cluster_y2 = clusters[j].sumY2/clusters[j].count;

                         double angle1 = atan2((double)lines[i][3] - lines[i][1], (double)lines[i][2] - lines[i][0]);
                         double angle2 = atan2((double)cluster_y2 - cluster_y1, (double)cluster_x2 - cluster_x1);
                         float distance_cluster1_to_line1 = sqrt(((cluster_x1 - lines[i][0])*(cluster_x1 - lines[i][0])) + (cluster_y1 - lines[i][1])*(cluster_y1 - lines[i][1]));
                         float distance_cluster1_to_line2 = sqrt(((cluster_x1 - lines[i][2])*(cluster_x1 - lines[i][2])) + (cluster_y1 - lines[i][3])*(cluster_y1 - lines[i][3]));
                         float distance_cluster2_to_line1 = sqrt(((cluster_x2 - lines[i][0])*(cluster_x2 - lines[i][0])) + (cluster_y2 - lines[i][1])*(cluster_y2 - lines[i][1]));
                         float distance_cluster2_to_line2 = sqrt(((cluster_x2 - lines[i][2])*(cluster_x2 - lines[i][2])) + (cluster_y2 - lines[i][3])*(cluster_y2 - lines[i][3]));

                         if (((distance_cluster1_to_line1 < distanceThreshold) &&
                                 (distance_cluster2_to_line2 < distanceThreshold || abs(angle1 - angle2) < angleThreshold)) ||
                              ((distance_cluster1_to_line2 < distanceThreshold) &&
                                 (distance_cluster2_to_line1 < distanceThreshold || abs(angle1 - angle2) < angleThreshold))){

                                 clusters[j].sumX1 += lines[i][0];
                                 clusters[j].sumY1 += lines[i][1];
                                 clusters[j].sumX2 += lines[i][2];
                                 clusters[j].sumY2 += lines[i][3];
                                 clusters[j].count += 1;
                                 in_some_cluster = true;
                        }
                    }
                    // if point doesnt fit, create new group for it
                    if (in_some_cluster == false){
                        LineCluster cluster;
                        cluster.sumX1 = lines[i][0];
                        cluster.sumY1 = lines[i][1];
                        cluster.sumX2 = lines[i][2];
                        cluster.sumY2 = lines[i][3];
                        cluster.count = 1;
                        clusters.push_back(cluster);
                    }
                }

                std::vector<cv::Vec4i> clusteredLines;

                for (int i = 0; i < clusters.size(); i++){
                    circle(dst, Point(clusters[i].sumX1/clusters[i].count, clusters[i].sumY1/clusters[i].count), 5, Scalar(0,0,255),-1);
                    circle(dst, Point(clusters[i].sumX2/clusters[i].count, clusters[i].sumY2/clusters[i].count), 5, Scalar(0,0,255),-1);

                    cv::line(dst, Point(clusters[i].sumX1/clusters[i].count, clusters[i].sumY1/clusters[i].count), Point(clusters[i].sumX2/clusters[i].count, clusters[i].sumY2/clusters[i].count), CV_RGB(255,0,0), 1);

                    cv::Vec4i line;
                    line[0] = clusters[i].sumX1/clusters[i].count;
                    line[1] = clusters[i].sumY1/clusters[i].count;
                    line[2] = clusters[i].sumX2/clusters[i].count;
                    line[3] = clusters[i].sumY2/clusters[i].count;

                    clusteredLines.push_back(line);
                }

            std::vector<cv::Point2f> corners;
            for (int i = 0; i < clusteredLines.size(); i++)
            {
                for (int j = i+1; j < clusteredLines.size(); j++)
                {
                    cv::Vec4i v = clusteredLines[i];
                    Point2f intersection;

                    bool has_intersection = ImageOperations::getIntersectionPoint(
                        Point(clusteredLines[i][0],clusteredLines[i][1]),
                        Point(clusteredLines[i][2], clusteredLines[i][3]),
                        Point(clusteredLines[j][0],clusteredLines[j][1]),
                        Point(clusteredLines[j][2], clusteredLines[j][3]),
                        intersection);

                    if (has_intersection
                        && intersection.x > 0
                        && intersection.y > 0
                        && intersection.x < dst.cols
                        && intersection.y < dst.rows){
                        corners.push_back(intersection);
                    }

                    cv::circle(dst, intersection, 3, CV_RGB(0,0,255), 2);
                }
            }

            cv::Point2f center(0,0);

            for (int i = 0; i < corners.size(); i++)
                center += corners[i];
            center *= (1. / corners.size());


            cv::circle(dst, center, 3, CV_RGB(255,255,0), 2);
            loadedImages[i] = dst.clone();
    }

    redrawImages();



}







void MainWindow::cutAndPerspective() {

    if (this->loadedImages.size() < 1) {
        QMessageBox msgBox;
        msgBox.setText("No images to process!");
        msgBox.exec();
        return;
    }


    for (int i = 0; i < loadedImages.size(); i++){
        idOCR::process(loadedImages[i]);
    }

    redrawImages();
}
