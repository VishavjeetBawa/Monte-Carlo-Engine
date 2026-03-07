#include "MCE.hpp"

#include "Payoff.hpp"
#include "RNG.hpp"
#include "RunStats.hpp"
#include <cmath>
#include<vector>
#include <thread>

namespace urop{


MCResult CrudeMCE::run(){
    RunStats stats;
    const long long num_of_paths = params_.M_;
    const long long time_steps = params_.N_;
    const double discount_factor = params_.dF_;

    std::vector<double> z_plus;
    z_plus.reserve(time_steps);

    std::vector<double> buffer;//path_prices inside calculate_path_payoff
    buffer.reserve(time_steps);

    for(long long i = 0 ; i < num_of_paths ; ++i){
        rng_->generate_deviates(time_steps , z_plus);
        double payoff = calculate_path_payoff(z_plus , buffer);

        stats.update(payoff);
    }

    return {
        stats.get_mean()*discount_factor , stats.get_std_error()*discount_factor
    };
}


double CrudeMCE::calculate_path_payoff(const std::vector<double>& deviates , std::vector<double>& path_prices) const {
    const double dt = params_.dT_;
    const double sigma = params_.sigma_;
    const double r = params_.R_;

    const double drift_term = ( r - 0.5*sigma*sigma)*dt;
    const double volatility_term = sigma * std::sqrt(dt);

    double log_price = std::log(params_.S0_);//for numerical stability for large numbers
    const long long time_steps = params_.N_;

    path_prices.clear();

    for(const double Z : deviates){
        log_price += drift_term + volatility_term*Z;//for numerical stability
        path_prices.push_back(std::exp(log_price));
    }

    return payoff_->calculate(path_prices);
}


//Antithetic-Variate version
MCResult AVMCE::run(){
    RunStats stats;
    const long long num_of_paths = params_.M_ / 2;
    const long long time_steps = params_.N_;
    const double discount_factor = params_.dF_;

    std::vector<double> z_plus;
    z_plus.reserve(time_steps);

    std::vector<double> buffer;//path_prices inside calculate_path_payoff
    buffer.reserve(time_steps);

    for(long long i = 0 ; i < num_of_paths ; ++i){
        rng_->generate_deviates(time_steps , z_plus);
        double payoff_a = calculate_path_payoff(z_plus , buffer);

        for(auto& z : z_plus ) z = -z ;

        double payoff_b = calculate_path_payoff(z_plus , buffer);
        double average_payoff = 0.5*(payoff_a + payoff_b);

        stats.update(average_payoff);
    }

    return {
        stats.get_mean()*discount_factor , stats.get_std_error()*discount_factor
    };

}

double AVMCE::calculate_path_payoff(const std::vector<double>& deviates , std::vector<double>& path_prices ) const{
    const double dt = params_.dT_;
    const double sigma = params_.sigma_;
    const double r = params_.R_;

    const double drift_term = ( r - 0.5*sigma*sigma)*dt;
    const double volatility_term = sigma * std::sqrt(dt);

    double log_price = std::log(params_.S0_);
    const long long time_steps = params_.N_;

    path_prices.clear();

    for(const double Z : deviates){
        log_price += drift_term + volatility_term*Z;//for numerical stability
        path_prices.push_back(std::exp(log_price));
    }

    return payoff_->calculate(path_prices);

}


//Optimised version (CV + AV)

MCResult OMCE::run(){
    RunStats stats;
    BiRunStats cv;
    const long long num_of_paths = params_.M_ / 2;
    const long long time_steps = params_.N_;
    const double discount_factor = params_.dF_;

    std::vector<double> buffer;//path_prices inside calculate_path_payoff
    buffer.reserve(time_steps);

    std::vector<double> z_plus;
    z_plus.reserve(time_steps);

/*
    std::vector<double> X , Y;
    X.reserve(num_of_paths);
    Y.reserve(num_of_paths);
*/

    for(long long i = 0 ; i < num_of_paths ; ++i){
        rng_->generate_deviates(time_steps , z_plus);

        double arith_payoff_a = calculate_path_payoff(z_plus , buffer , *arith_payoff_);
        double geo_payoff_a = calculate_path_payoff(z_plus , buffer , *geo_payoff_);

        for(auto& z : z_plus ) z = -z ;

        double arith_payoff_b = calculate_path_payoff(z_plus , buffer, *arith_payoff_);
        double geo_payoff_b = calculate_path_payoff(z_plus , buffer , *geo_payoff_);

        double geo_average_payoff = 0.5*(geo_payoff_a + geo_payoff_b);
        double arith_average_payoff = 0.5*(arith_payoff_a + arith_payoff_b);

    /*
        X.push_back(arith_average_payoff);
        Y.push_back(geo_average_payoff);
    */

        cv.update(arith_average_payoff , geo_average_payoff);
    }

    const double beta = cv.beta();
    const double geo_exact = geo_exact_;

    /*
    for(long long i = 0 ; i<num_of_paths ; ++i){
        double corrected = X[i] - beta*(Y[i] - geo_exact);
        stats.update(corrected);
    }
    */

    for (long long i = 0 ; i<num_of_paths ; ++i){
        rng_->generate_deviates(time_steps , z_plus);

        double arith_payoff_a = calculate_path_payoff(z_plus , buffer , *arith_payoff_);
        double geo_payoff_a = calculate_path_payoff(z_plus , buffer , *geo_payoff_);

        for(auto& z : z_plus ) z = -z ;

        double arith_payoff_b = calculate_path_payoff(z_plus , buffer, *arith_payoff_);
        double geo_payoff_b = calculate_path_payoff(z_plus , buffer , *geo_payoff_);

        double geo_average_payoff = 0.5*(geo_payoff_a + geo_payoff_b);
        double arith_average_payoff = 0.5*(arith_payoff_a + arith_payoff_b);

        double corrected = arith_average_payoff - beta*(geo_average_payoff - geo_exact);
        stats.update(corrected);

    }

    return {
        stats.get_mean()*discount_factor , stats.get_std_error()*discount_factor
    };

}

double OMCE::calculate_path_payoff(const std::vector<double>& deviates , std::vector<double>& path_prices , Payoff& payoff) const{
    const double dt = params_.dT_;
    const double sigma = params_.sigma_;
    const double r = params_.R_;

    const double drift_term = ( r - 0.5*sigma*sigma)*dt;
    const double volatility_term = sigma * std::sqrt(dt);

    double log_price = std::log(params_.S0_);
    const long long time_steps = params_.N_;

    path_prices.clear();

    for(const double Z : deviates){
        log_price += drift_term + volatility_term*Z;//for numerical stability
        path_prices.push_back(std::exp(log_price));
    }

    return payoff.calculate(path_prices);

}

/*
 *
 *QUASI OPTIMISED MONTE-CARLO ENGINE:-
 * 
 */

double QOMCE::calculate_path_payoff(const std::vector<double>& deviates,
                                    std::vector<double>& path_prices,
                                    Payoff& payoff) const
{
    const double dt = params_.dT_;
    const double sigma = params_.sigma_;
    const double r = params_.R_;

    const double drift_term = (r - 0.5*sigma*sigma) * dt;
    const double volatility_term = sigma * std::sqrt(dt);

    double log_price = std::log(params_.S0_);

    path_prices.clear();

    for (const double Z : deviates) {
        log_price += drift_term + volatility_term * Z;
        path_prices.push_back(std::exp(log_price));
    }

    return payoff.calculate(path_prices);
}

MCResult QOMCE::run()
{
    RunStats stats;   // now over batches

    const long long total_paths = params_.M_;
    const long long time_steps  = params_.N_;
    const double discount_factor = params_.dF_;

    const long long Nbatches = total_paths / batch_size_;
    const long long paths_per_batch = batch_size_ / 2; // because AV

    std::vector<double> buffer;
    buffer.reserve(time_steps);

    std::vector<double> z_plus;
    z_plus.reserve(time_steps);
// QMC requires Sobol RNG
    auto* sobol = dynamic_cast<Sobol*>(rng_.get());
    if (!sobol)
        throw std::runtime_error("QOMCE requires Sobol RNG");

    // Reset Sobol sequence once
    sobol->reset();


    for (long long b = 0; b < Nbatches; ++b)
    {
      // New digital shift per batch
        sobol->randomize_shift();
      

        // ================= PASS 1: estimate beta =================
        BiRunStats cv;

        for (long long i = 0; i < paths_per_batch; ++i)
        {
            rng_->generate_deviates(time_steps, z_plus);

            double arith_a = calculate_path_payoff(z_plus, buffer, *arith_payoff_);
            double geo_a   = calculate_path_payoff(z_plus, buffer, *geo_payoff_);

            for (auto& z : z_plus) z = -z;

            double arith_b = calculate_path_payoff(z_plus, buffer, *arith_payoff_);
            double geo_b   = calculate_path_payoff(z_plus, buffer, *geo_payoff_);

            double arith_avg = 0.5 * (arith_a + arith_b);
            double geo_avg   = 0.5 * (geo_a   + geo_b);

            cv.update(arith_avg, geo_avg);
        }

        const double beta = cv.beta();
        const double geo_exact = geo_exact_;

        // ================= PASS 2: corrected estimator =================
        double batch_sum = 0.0;

        for (long long i = 0; i < paths_per_batch; ++i)
        {
            rng_->generate_deviates(time_steps, z_plus);

            double arith_a = calculate_path_payoff(z_plus, buffer, *arith_payoff_);
            double geo_a   = calculate_path_payoff(z_plus, buffer, *geo_payoff_);

            for (auto& z : z_plus) z = -z;

            double arith_b = calculate_path_payoff(z_plus, buffer, *arith_payoff_);
            double geo_b   = calculate_path_payoff(z_plus, buffer, *geo_payoff_);

            double arith_avg = 0.5 * (arith_a + arith_b);
            double geo_avg   = 0.5 * (geo_a   + geo_b);

            double corrected =
                arith_avg - beta * (geo_avg - geo_exact);

            batch_sum += corrected;
        }

        double batch_mean = batch_sum / paths_per_batch;

        // this is the key difference vs OMCE
        stats.update(batch_mean);
    }

    return {
        stats.get_mean() * discount_factor,
        stats.get_std_error() * discount_factor
    };
}

/*
*
* Concurrent QOMCE
*
*/

MCResult COQMCE::run()
{
    const long long M = params_.M_ / 2;   // because AV
    const long long N = params_.N_;
    const double discount = params_.dF_;

    const long long batch_size = 64;

    const unsigned num_threads = std::min(4u, std::thread::hardware_concurrency());

    // Per-thread stats
    struct alignas(64) ThreadData {
        RunStats stats;
    };

    std::vector<ThreadData> thread_stats(num_threads);
    std::vector<std::thread> threads;

    // Work splitter
    auto worker = [&](unsigned tid)
    {
        // Each thread gets its own Sobol
        auto rng = prototype_rng_->clone();
        auto* sobol = dynamic_cast<Sobol*>(rng.get());
        
        if (!sobol)
            throw std::runtime_error("COQMCE requires Sobol RNG");

        sobol->reset();
        sobol->randomize_shift();   // unique per thread


        std::vector<double> z_plus(N);
        std::vector<double> buffer(N);

        long long paths_per_thread = M / num_threads;
        long long start = tid * paths_per_thread;
        long long end   = (tid == num_threads - 1)? M: start + paths_per_thread;

        BiRunStats cv;

        // ---------- First pass: estimate beta ----------
        for (long long i = start; i < end; ++i)
        {
            rng->generate_deviates(N, z_plus);

            double arith_a = calculate_path_payoff(z_plus, buffer, *arith_payoff_);
            double geo_a   = calculate_path_payoff(z_plus, buffer, *geo_payoff_);

            for (auto& z : z_plus) z = -z;

            double arith_b = calculate_path_payoff(z_plus, buffer, *arith_payoff_);
            double geo_b   = calculate_path_payoff(z_plus, buffer, *geo_payoff_);

            double arith_avg = 0.5 * (arith_a + arith_b);
            double geo_avg   = 0.5 * (geo_a   + geo_b);

            cv.update(arith_avg, geo_avg);
        }

        const double beta = cv.beta();
        const double geo_exact = geo_exact_;

        // Reset Sobol for second pass
        sobol->reset();

        // ---------- Second pass: corrected estimator ----------
        for (long long i = start; i < end; ++i)
        {
            rng->generate_deviates(N, z_plus);

            double arith_a = calculate_path_payoff(z_plus, buffer, *arith_payoff_);
            double geo_a   = calculate_path_payoff(z_plus, buffer, *geo_payoff_);

            for (auto& z : z_plus) z = -z;

            double arith_b = calculate_path_payoff(z_plus, buffer, *arith_payoff_);
            double geo_b   = calculate_path_payoff(z_plus, buffer, *geo_payoff_);

            double arith_avg = 0.5 * (arith_a + arith_b);
            double geo_avg   = 0.5 * (geo_a   + geo_b);

            double corrected = arith_avg - beta * (geo_avg - geo_exact);
            thread_stats[tid].stats.update(corrected);
        }
    };

    // Launch threads
    for (unsigned t = 0; t < num_threads; ++t)
        threads.emplace_back(worker, t);

    for (auto& th : threads)
        th.join();

    // Merge stats

    RunStats final_stats;
    for (unsigned t = 0; t < num_threads; ++t)
    {
        final_stats.merge(thread_stats[t].stats);
    }


    return {
        final_stats.get_mean() * discount,
        final_stats.get_std_error() * discount
    };
}


double COQMCE::calculate_path_payoff(const std::vector<double>& deviates,
                                    std::vector<double>& path_prices,
                                    Payoff& payoff) const
{
    const double dt = params_.dT_;
    const double sigma = params_.sigma_;
    const double r = params_.R_;

    const double drift_term = (r - 0.5*sigma*sigma) * dt;
    const double volatility_term = sigma * std::sqrt(dt);

    double log_price = std::log(params_.S0_);

    path_prices.clear();

    for (const double Z : deviates) {
        log_price += drift_term + volatility_term * Z;
        path_prices.push_back(std::exp(log_price));
    }

    return payoff.calculate(path_prices);
}

};
