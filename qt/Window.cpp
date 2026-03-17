#include "Window.hpp"

#include <QVBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QLineEdit>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QFont>
#include <chrono>

#include "OptionParams.hpp"
#include "MCE.hpp"
#include "Payoff.hpp"
#include "RNG.hpp"
#include "CudaQOMCE.hpp"

using namespace urop;

Window::Window(QWidget *parent)
    : QWidget(parent)
{
    // ----- Apply a style sheet for a modern look -----
    setStyleSheet(R"(
        QWidget {
            background-color: #f5f5f5;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #cccccc;
            border-radius: 8px;
            margin-top: 1ex;
            padding-top: 10px;
            background-color: white;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QLineEdit {
            border: 1px solid #aaa;
            border-radius: 4px;
            padding: 4px;
            background-color: white;
        }
        QLineEdit:focus {
            border-color: #3daee9;
        }
        QPushButton {
            background-color: #3daee9;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #1e7ab9;
        }
        QComboBox {
            border: 1px solid #aaa;
            border-radius: 4px;
            padding: 4px;
            background-color: white;
        }
        QLabel#outputLabel {
            font-size: 12pt;
            font-weight: bold;
            color: #333;
        }
    )");

    auto *mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(15);
    mainLayout->setContentsMargins(20, 20, 20, 20);

    // ----- Input Parameters Group -----
    auto *inputGroup = new QGroupBox("Input Parameters");
    auto *formLayout = new QFormLayout(inputGroup);
    formLayout->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    formLayout->setLabelAlignment(Qt::AlignRight);

    s0Box = new QLineEdit("100");
    kBox = new QLineEdit("100");
    tBox = new QLineEdit("1");
    rBox = new QLineEdit("0.05");
    sigmaBox = new QLineEdit("0.2");
    nBox = new QLineEdit("100");
    mBox = new QLineEdit("100000");

    formLayout->addRow("S0:", s0Box);
    formLayout->addRow("K:", kBox);
    formLayout->addRow("T (years):", tBox);
    formLayout->addRow("r:", rBox);
    formLayout->addRow("σ:", sigmaBox);
    formLayout->addRow("N (steps):", nBox);
    formLayout->addRow("M (paths):", mBox);

    mainLayout->addWidget(inputGroup);

    // ----- Engine Selection -----
    engineBox = new QComboBox();
    engineBox->addItem("CPU Crude MC");
    engineBox->addItem("GPU QMC (CUDA)");
    mainLayout->addWidget(engineBox);

    // ----- Run Button -----
    runButton = new QPushButton("Run Engine");
    runButton->setCursor(Qt::PointingHandCursor);
    mainLayout->addWidget(runButton);

    // ----- Output Group -----
    auto *outputGroup = new QGroupBox("Results");
    auto *outputLayout = new QVBoxLayout(outputGroup);

    priceLabel = new QLabel("Price: ");
    stderrLabel = new QLabel("StdErr: ");
    timeLabel = new QLabel("Time: ");

    // Give these labels an object name for styling
    priceLabel->setObjectName("outputLabel");
    stderrLabel->setObjectName("outputLabel");
    timeLabel->setObjectName("outputLabel");

    outputLayout->addWidget(priceLabel);
    outputLayout->addWidget(stderrLabel);
    outputLayout->addWidget(timeLabel);

    mainLayout->addWidget(outputGroup);

    // Stretch to push everything up
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

    if (engineBox->currentText() == "GPU QMC (CUDA)")
    {
        CudaQOMCE engine(params);
        result = engine.run();
    }
    else // CPU Crude MC
    {
        auto payoff = std::make_unique<AsianCallPayoff>(K);
        auto rng = std::make_unique<MtRand>();
        CrudeMCE engine(params, std::move(payoff), std::move(rng));
        result = engine.run();
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    priceLabel->setText("Price: " + QString::number(result.price, 'f', 6));
    stderrLabel->setText("StdErr: " + QString::number(result.std_error, 'f', 6));
    timeLabel->setText("Time: " + QString::number(elapsed, 'f', 4) + " s");
}
