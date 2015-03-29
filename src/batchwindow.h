#ifndef BATCHWINDOW_H
#define BATCHWINDOW_H

#include <QDialog>

namespace Ui {
class batchWindow;
}

class batchWindow : public QDialog
{
    Q_OBJECT

public:
    explicit batchWindow(QWidget *parent = 0);
    ~batchWindow();

private:
    Ui::batchWindow *ui;
};

#endif // BATCHWINDOW_H
