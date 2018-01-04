#ifndef COV_HPP
#define COV_HPP

// squared exponential covariance function

#include <armadillo>

using arma::vec;

double cov(int n, int m, vec x, vec y, vec hypers);


#endif
