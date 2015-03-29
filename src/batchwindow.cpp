#include "batchwindow.h"
#include "ui_batchwindow.h"

batchWindow::batchWindow(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::batchWindow)
{
    ui->setupUi(this);
}

batchWindow::~batchWindow()
{
    delete ui;
}
