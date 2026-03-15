#pragma once

#include <QWidget>

class QLineEdit;
class QLabel;
class QPushButton;
class QComboBox;

class Window : public QWidget
{
    Q_OBJECT

public:
    explicit Window(QWidget *parent = nullptr);

private slots:
    void runEngine();

private:

    QLineEdit* s0Box;
    QLineEdit* kBox;
    QLineEdit* tBox;
    QLineEdit* rBox;
    QLineEdit* sigmaBox;
    QLineEdit* nBox;
    QLineEdit* mBox;

    QComboBox* engineBox;

    QLabel* priceLabel;
    QLabel* stderrLabel;
    QLabel* timeLabel;

    QPushButton* runButton;
};

#include "main.moc"
