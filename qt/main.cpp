#include <QApplication>
#include "Window.hpp"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    // Optional: set a default font
    QFont defaultFont("Segoe UI", 10);
    app.setFont(defaultFont);

    Window window;
    window.setWindowTitle("Monte Carlo Option Pricer");
    window.setMinimumSize(400, 600);
    window.show();

    return app.exec();
}
