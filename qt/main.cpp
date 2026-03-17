#include <QApplication>
#include "Window.hpp"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    Window window;
    window.setWindowTitle("Monte Carlo Option Pricer");
    window.show();

    return app.exec();
}
