#ifndef OPTIM_HPP
#define OPTIM_HPP

#include "gp.hpp"

double nlog_lik(unsigned int, const double *, double *, void *);
double log_lik_optim(GP& gp, vec lbounds, vec ubounds, int niter, double ftol_rel);

#endif
