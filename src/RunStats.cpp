#include "RunStats.hpp"
#include<cmath>

namespace urop{

void RunStats::update(double x){
    count_++;
    double del_1 = x - mean_ ;
    mean_ += del_1/count_ ;
    double del_2 = x - mean_;
    M2_ += del_1 * del_2;
}

double RunStats::get_std_dev() const{
    if (count_ < 2){
        return 0.0;
    }
    double variance = M2_ / (count_ - 1);
    return std::sqrt(variance);
}

double RunStats::get_std_error() const{
    if (count_ < 2){
        return 0.0;
    }
    return this->get_std_dev() / std::sqrt(static_cast<double>(count_));
}

void BiRunStats::update(double x , double y){
    count_++;
    double dx = x - mean_x_;
    double dy = y - mean_y_;

    mean_x_ += dx/count_;
    mean_y_ += dy/count_;

    cov_xy_ += dx*(y-mean_y_);
    var_y_ += dy*(y-mean_y_);
}



};
