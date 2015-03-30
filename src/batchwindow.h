#ifndef BATCHWINDOW_H
#define BATCHWINDOW_H

#include <QMainWindow>
#include <QScopedPointer>
#include <QFileDialog>
#include <iostream>
#include "helpers.h"
#include <QDropEvent>
#include <QUrl>
#include <QDebug>
#include <QList>
#include <QMimeData>
#include <QMessageBox>

#include <boost/filesystem.hpp>

namespace Ui {
class batchWindow;
}

class batchWindow : public QDialog
{
    Q_OBJECT

public:
    explicit batchWindow(QWidget *parent = 0);
    ~batchWindow();

protected:
    void dragEnterEvent(QDragEnterEvent *e);
    void dropEvent(QDropEvent *e);

private:
    Ui::batchWindow *ui;
};

#endif // BATCHWINDOW_H
