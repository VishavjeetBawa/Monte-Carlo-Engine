#pragma once

#include<vector>
#include <algorithm>
#include<numeric>

namespace urop{

class Payoff{
public:
    virtual ~Payoff() = default;
    virtual double calculate(const std::vector<double>& path_prices) const = 0;   
};


class AsianCallPayoff: public Payoff{
public:
    explicit AsianCallPayoff(double k):K_(k){}
    double calculate(const std::vector<double>& path_prices) const override;

private:
    double K_;
};

class GeometricAsianPayoff final : public Payoff{
public:
    explicit GeometricAsianPayoff(double k): K_(k){}
    double calculate(const std::vector<double>& path_prices) const override;
private:
    double K_;
};

};
