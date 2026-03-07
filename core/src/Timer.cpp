#include"Timer.hpp"
#include<iostream>
#include <chrono>
#include <string>
#include <iomanip>

namespace urop{

double Timer::elapsed_seconds() const{
    std::chrono::duration<double> elapsed;
    if(is_running_){
        elapsed = Clock::now() - start_time_;
    }else{
        elapsed = stop_time_ - start_time_;
    }
    return elapsed.count();
}

void Timer::print_report(const std::string& Label , long long n_paths) const{
    double seconds = elapsed_seconds();
    double throughput = static_cast<double>(n_paths)/seconds;

    std::cout<<std::string(30,'=')<<"\n";
    std::cout<<"Benchmark: "<<Label<<"\n";
    std::cout<<"Time(s):"<<std::setprecision(4)<<seconds<<"s\n";
    std::cout<<"Throughput:"<<std::scientific<<throughput<<"path/sec\n";
    std::cout<<std::string(30 , '=')<<std::endl;
}

}
