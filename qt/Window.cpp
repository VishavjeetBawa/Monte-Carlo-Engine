#include "Window.hpp"

#include <QVBoxLayout>
#include <QGridLayout>
#include <QLineEdit>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>

#include "OptionParams.hpp"
#include "Timer.hpp"
#include "MCE.hpp"          // for CrudeMCE
#include "Payoff.hpp"       // for AsianCallPayoff
#include "RNG.hpp"          // for MtRand
#include "CudaQOMCE.hpp"

using namespace urop;

Window::Window(QWidget *parent)
    : QWidget(parent)
{
    auto *layout = new QVBoxLayout(this);

    auto *grid = new QGridLayout();

    s0Box = new QLineEdit("100");
    kBox = new QLineEdit("100");
    tBox = new QLineEdit("1");
    rBox = new QLineEdit("0.05");
    sigmaBox = new QLineEdit("0.2");
    nBox = new QLineEdit("100");
    mBox = new QLineEdit("100000");

    grid->addWidget(new QLabel("S0"), 0, 0);
    grid->addWidget(s0Box, 0, 1);
    grid->addWidget(new QLabel("K"), 1, 0);
    grid->addWidget(kBox, 1, 1);
    grid->addWidget(new QLabel("T"), 2, 0);
    grid->addWidget(tBox, 2, 1);
    grid->addWidget(new QLabel("r"), 3, 0);
    grid->addWidget(rBox, 3, 1);
    grid->addWidget(new QLabel("sigma"), 4, 0);
    grid->addWidget(sigmaBox, 4, 1);
    grid->addWidget(new QLabel("N"), 5, 0);
    grid->addWidget(nBox, 5, 1);
    grid->addWidget(new QLabel("M"), 6, 0);
    grid->addWidget(mBox, 6, 1);

    layout->addLayout(grid);

    engineBox = new QComboBox();
    engineBox->addItem("CPU Crude MC");
    engineBox->addItem("GPU QMC (CUDA)");
    layout->addWidget(engineBox);

    runButton = new QPushButton("Run Engine");
    layout->addWidget(runButton);

    priceLabel = new QLabel("Price: ");
    stderrLabel = new QLabel("StdErr: ");
    timeLabel = new QLabel("Time: ");
    layout->addWidget(priceLabel);
    layout->addWidget(stderrLabel);
    layout->addWidget(timeLabel);

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

    Timer timer;
    timer.start();

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

    double elapsed = timer.elapsed();

    priceLabel->setText("Price: " + QString::number(result.price, 'f', 6));
    stderrLabel->setText("StdErr: " + QString::number(result.std_error, 'f', 6));
    timeLabel->setText("Time: " + QString::number(elapsed, 'f', 4) + " s");
}
