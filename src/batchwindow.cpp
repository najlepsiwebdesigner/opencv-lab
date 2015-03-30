#include "batchwindow.h"
#include "ui_batchwindow.h"

#include <string.h>


using namespace std;
using namespace boost::filesystem;

batchWindow::batchWindow(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::batchWindow)
{
    setAcceptDrops(true);
    ui->setupUi(this);
}

batchWindow::~batchWindow()
{
    delete ui;
}


void batchWindow::dropEvent(QDropEvent *ev)
{
    QList<QUrl> urls = ev->mimeData()->urls();
    QString filename;

    foreach(QUrl url, urls)
    {
        filename = QString(url.toString());
        filename.replace(QString("file://"), QString(""));

        String name = filename.toStdString();

        if (is_directory(name)) {
            qDebug() << "Directory: " << filename << endl;
            QMessageBox msgBox;
            msgBox.setText(filename + " is a directory, which is not supported!");
            msgBox.exec();
        } else {
            qDebug() << "File: " << filename << endl;
        }

    }
}


void batchWindow::dragEnterEvent(QDragEnterEvent *ev)
{
    ev->accept();
}



