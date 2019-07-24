#ifndef EOS_HPP
#define EOS_HPP

#include <iostream>

#include "state.hpp"
#include "matrix-util.hpp"

class Eos {
public:
	// Helper derivatives.  In these (LaTeX) expressions, the matrix
	// indices are i and j.

	//\frac{1}{\rho} \deriv{\sigma_{ni}}{F_{j\beta}}
  	virtual arma::mat33 A(double S, arma::mat33 F, int n, int beta) const =0;

	// \frac{1}{\rho} \deriv{\sigma_{ij}}{S}
	virtual arma::mat33 B(double S, arma::mat33 F) const =0;


	virtual double rho0(void) const =0;
	virtual double e_internal(double entropy, arma::mat33 G) const =0;
	virtual double entropy(double e_internal, arma::mat33 G) const =0;
	virtual arma::mat33 sigma(double entropy, arma::mat33 G) const =0;

	friend state_mat_t Jacobian(const Eos& e, PrimState W, int dirn);
};

arma::mat33 dE_dG(const Eos& eos, double S, arma::mat33 G);

PrimState cons_to_prim(const Eos& eos, ConsState c);

ConsState prim_to_cons(const Eos& eos, PrimState p);

state_mat_t Jacobian(const Eos& eos, PrimState W, int dirn);

#endif
