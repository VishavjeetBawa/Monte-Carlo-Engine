#include "Window.hpp"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QLineEdit>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QFontDatabase>
#include <QGraphicsDropShadowEffect>
#include <chrono>
#include <memory>

#include "OptionParams.hpp"
#include "MCE.hpp"
#include "Payoff.hpp"
#include "RNG.hpp"
#include "CudaQOMCE.hpp"
#include "GAsianPricer.hpp"

using namespace urop;

Window::Window(QWidget *parent)
    : QWidget(parent)
{
    // ----- Load a modern font -----
    QFontDatabase::addApplicationFont(":/fonts/Roboto-Regular.ttf"); // optional, fallback to system
    QFont defaultFont("Segoe UI", 10);
    if (QFontDatabase::families().contains("Roboto"))
        defaultFont.setFamily("Roboto");
    qApp->setFont(defaultFont);

    // ----- Global stylesheet (web‑inspired) -----
    setStyleSheet(R"(
        QWidget {
            background-color: #f8f9fc;
        }
        QGroupBox {
            font-weight: 500;
            border: 1px solid #e0e5ec;
            border-radius: 12px;
            margin-top: 1.2em;
            padding-top: 15px;
            background-color: white;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 20px;
            padding: 0 8px 0 8px;
            color: #2c3e50;
            font-size: 14px;
        }
        QLineEdit, QComboBox {
            border: 1px solid #d0d9e8;
            border-radius: 8px;
            padding: 8px 12px;
            background-color: white;
            color: #1e2b3a;
            selection-background-color: #3f8cff;
            font-size: 13px;
        }
        QLineEdit:focus, QComboBox:focus {
            border-color: #3f8cff;
            outline: none;
        }
        QPushButton {
            background-color: #3f8cff;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 24px;
            font-weight: 600;
            font-size: 15px;
        }
        QPushButton:hover {
            background-color: #2b6ed9;
        }
        QPushButton:pressed {
            background-color: #1a4faa;
        }
        QLabel {
            color: #2c3e50;
            font-size: 13px;
        }
        QLabel#outputLabel {
            font-size: 15px;
            font-weight: 600;
            color: #1e2b3a;
            background-color: #f0f4fa;
            padding: 8px 12px;
            border-radius: 8px;
            margin: 2px 0;
        }
        QComboBox::drop-down {
            border: none;
            width: 24px;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #5a6b7c;
            margin-right: 8px;
        }
    )");

    // ----- Main layout with margins and spacing -----
    auto *mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(20);
    mainLayout->setContentsMargins(30, 30, 30, 30);

    // ----- Header (optional) -----
    QLabel *titleLabel = new QLabel("⚡ Monte Carlo Option Pricer");
    titleLabel->setStyleSheet("font-size: 24px; font-weight: 600; color: #1e2b3a; padding: 10px 0;");
    mainLayout->addWidget(titleLabel);

    // ----- Input Parameters Group (two‑column layout) -----
    auto *inputGroup = new QGroupBox("Parameters");
    auto *inputGrid = new QGridLayout(inputGroup);
    inputGrid->setHorizontalSpacing(20);
    inputGrid->setVerticalSpacing(15);

    // Create input fields
    s0Box = new QLineEdit("100");
    kBox = new QLineEdit("100");
    tBox = new QLineEdit("1");
    rBox = new QLineEdit("0.05");
    sigmaBox = new QLineEdit("0.2");
    nBox = new QLineEdit("100");
    mBox = new QLineEdit("100000");

    // First column
    inputGrid->addWidget(new QLabel("Spot Price (S₀)"), 0, 0);
    inputGrid->addWidget(s0Box, 0, 1);
    inputGrid->addWidget(new QLabel("Strike (K)"), 1, 0);
    inputGrid->addWidget(kBox, 1, 1);
    inputGrid->addWidget(new QLabel("Maturity (T, years)"), 2, 0);
    inputGrid->addWidget(tBox, 2, 1);
    inputGrid->addWidget(new QLabel("Risk‑free Rate (r)"), 3, 0);
    inputGrid->addWidget(rBox, 3, 1);

    // Second column
    inputGrid->addWidget(new QLabel("Volatility (σ)"), 0, 2);
    inputGrid->addWidget(sigmaBox, 0, 3);
    inputGrid->addWidget(new QLabel("Time Steps (N)"), 1, 2);
    inputGrid->addWidget(nBox, 1, 3);
    inputGrid->addWidget(new QLabel("Paths (M)"), 2, 2);
    inputGrid->addWidget(mBox, 2, 3);

    // Column stretch
    inputGrid->setColumnStretch(0, 1);
    inputGrid->setColumnStretch(1, 2);
    inputGrid->setColumnStretch(2, 1);
    inputGrid->setColumnStretch(3, 2);

    mainLayout->addWidget(inputGroup);

    // ----- Engine Selection (styled combo) -----
    engineBox = new QComboBox();
    engineBox->addItem("CPU Crude MC");
    engineBox->addItem("CPU Concurrent QMC");
    engineBox->addItem("GPU QMC (CUDA)");
    engineBox->setMinimumHeight(40);
    mainLayout->addWidget(engineBox);

    // ----- Run Button (with shadow) -----
    runButton = new QPushButton("Run Simulation");
    runButton->setCursor(Qt::PointingHandCursor);
    runButton->setMinimumHeight(50);

    // Add a drop shadow to the button
    auto *shadow = new QGraphicsDropShadowEffect();
    shadow->setBlurRadius(15);
    shadow->setOffset(0, 4);
    shadow->setColor(QColor(0, 0, 0, 50));
    runButton->setGraphicsEffect(shadow);

    mainLayout->addWidget(runButton);

    // ----- Results Group (cards style) -----
    auto *outputGroup = new QGroupBox("Results");
    auto *outputLayout = new QVBoxLayout(outputGroup);
    outputLayout->setSpacing(8);

    priceLabel = new QLabel("Price: ");
    stderrLabel = new QLabel("StdErr: ");
    timeLabel = new QLabel("Time: ");

    priceLabel->setObjectName("outputLabel");
    stderrLabel->setObjectName("outputLabel");
    timeLabel->setObjectName("outputLabel");

    outputLayout->addWidget(priceLabel);
    outputLayout->addWidget(stderrLabel);
    outputLayout->addWidget(timeLabel);

    mainLayout->addWidget(outputGroup);

    // Add stretch to keep everything compact at the top
    mainLayout->addStretch();

    connect(runButton, &QPushButton::clicked, this, &Window::runEngine);
}

void Window::runEngine()
{
    double S0 = s0Box->text().toDouble();
    double K = kBox->text().toDouble();
    double T = tBox->text().toDouble();
    double r = rBox->text().toDouble();
    double sigma = sigmaBox->text().toDouble();
    int N = nBox->text().toInt();
    long long M = mBox->text().toLongLong();

    AOP params(S0, K, T, r, sigma, N, M);

    auto start = std::chrono::high_resolution_clock::now();

    MCResult result;
    QString engine = engineBox->currentText();

    try {
        if (engine == "GPU QMC (CUDA)")
        {
            CudaQOMCE engine(params);
            result = engine.run();
        }
        else if (engine == "CPU Concurrent QMC")
        {
            auto arith_payoff = std::make_unique<AsianCallPayoff>(K);
            auto geo_payoff   = std::make_unique<GeometricAsianPayoff>(K);
            auto rng = std::make_unique<Sobol>(N, T);
            geo_pricer exact_calc(params);
            double geo_exact = exact_calc.price();
            COQMCE engine(params,
                          std::move(arith_payoff),
                          std::move(geo_payoff),
                          std::move(rng),
                          geo_exact);
            result = engine.run();
        }
        else // CPU Crude MC
        {
            auto payoff = std::make_unique<AsianCallPayoff>(K);
            auto rng = std::make_unique<MtRand>();
            CrudeMCE engine(params, std::move(payoff), std::move(rng));
            result = engine.run();
        }
    } catch (const std::exception &e) {
        priceLabel->setText("Error: " + QString(e.what()));
        stderrLabel->setText("StdErr: --");
        timeLabel->setText("Time: --");
        return;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    priceLabel->setText("Price: " + QString::number(result.price, 'f', 6));
    stderrLabel->setText("StdErr: " + QString::number(result.std_error, 'f', 6));
    timeLabel->setText("Time: " + QString::number(elapsed, 'f', 4) + " s");
}
