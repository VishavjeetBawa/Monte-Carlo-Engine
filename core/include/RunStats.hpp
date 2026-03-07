#pragma once 

#include<cmath>

namespace urop{

class RunStats{
public:
    RunStats () = default;
    void update(double x);
    double get_mean() const { return mean_; }
    long long get_count() const { return count_; }
    double get_M2() const { return M2_; }
    double get_std_dev() const;
    double get_std_error() const;
    void merge(const RunStats& other);
private:
    long long count_ = 0;
    double mean_ = 0.0;
    double M2_ = 0.0;
};


//BiRunStats for covariance in beta term and CONTROL VARIATES!!
class BiRunStats{
public:
    BiRunStats () = default;
    void update(double x , double y);
    double get_mean_x() const { return mean_x_; }
    double get_mean_y() const { return mean_y_; }

    double beta() const{ return covariance()/variance_y(); }

private:
    double covariance() const{ return (count_ > 1) ? cov_xy_ / (count_ - 1) : 0.0; }
    double variance_y() const{ return (count_ > 1) ? var_y_ / (count_ - 1) : 0.0; }
    double x_ = 0.0;
    double y_ = 0.0;
    long long count_ = 0;
    double mean_x_ = 0.0;
    double mean_y_ = 0.0;
    double cov_xy_ = 0.0;
    double var_y_ = 0.0; 
};

};
