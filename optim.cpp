#include <nlopt.h>
#include "optim.hpp"
#include "gp.hpp"

double nlog_lik(unsigned int n, const double *hypers, double *, void *work)
{
	GP &gp(*(GP *)work);
	vec hypers1(hypers, n);
	gp.set_hypers(hypers1);
	double ll = gp.log_lik();
	
	std::cout << "nlog_lik: " << ll << " " << hypers1.t();

	return ll;
}

static void check_error_code(int err)
{
	if (err <= 0) {
		std::cerr << "NLopt failed with error code" << err << std::endl;
		if (err == -4) {
			std::cerr << "Halted because roundoff errors limited progress.  Continuing with best result.\n";
		}
		else {
			exit(1);
		}
	}
}


double log_lik_optim(GP& gp, vec lbounds, vec ubounds,
		   int niter, double ftol_rel)
{
	int err;
	nlopt_opt opt;
	
	vec hypers(gp.get_hypers());
	
	opt = nlopt_create(NLOPT_LN_BOBYQA, gp.Nhypers());

	err = nlopt_set_lower_bounds(opt, lbounds.memptr());
	  check_error_code(err);
	err = nlopt_set_upper_bounds(opt, ubounds.memptr());
	  check_error_code(err);
	err = nlopt_set_max_objective(opt, nlog_lik, &gp);
	  check_error_code(err);
	err = nlopt_set_ftol_rel(opt, ftol_rel);
	  check_error_code(err);
	nlopt_set_maxeval(opt, niter);

	double maxf;
	std::cout << "# log(liklihood) hyperparameters" << std::endl;
	err = nlopt_optimize(opt, hypers.memptr(), &maxf);

	nlopt_destroy(opt);

	return maxf;
}
