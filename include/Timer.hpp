#pragma once

#include <chrono>
#include <iostream>
#include <string>

namespace urop{

class Timer{
public:
    inline void start(){ start_time_ = Clock::now(); is_running_ = true; }
    inline void stop(){ stop_time_=Clock::now(); is_running_ = false; }
    void print_report(const std::string& Label , long long n_paths) const;

private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    TimePoint start_time_;
    TimePoint stop_time_;
    bool is_running_ = false;

    double elapsed_seconds() const;
    inline double elapsed_milliseconds() const{ return elapsed_seconds()*1000.0; }
}; 

}

