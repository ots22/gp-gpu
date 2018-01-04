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
  	virtual mat3 A(double S, mat3 F, int n, int beta) const =0;

	// \frac{1}{\rho} \deriv{\sigma_{ij}}{S}
	virtual mat3 B(double S, mat3 F) const =0;


	virtual double rho0(void) const =0;
	virtual double e_internal(double entropy, mat3 G) const =0;
	virtual double entropy(double e_internal, mat3 G) const =0;
	virtual mat3 sigma(double entropy, mat3 G) const =0;

	friend state_mat_t Jacobian(const Eos& e, PrimState W, int dirn);
};

mat3 dE_dG(const Eos& eos, double S, mat3 G);

PrimState cons_to_prim(const Eos& eos, ConsState c);

ConsState prim_to_cons(const Eos& eos, PrimState p);

state_mat_t Jacobian(const Eos& eos, PrimState W, int dirn);

#endif
