#ifndef STATE_HPP
#define STATE_HPP

#include <numeric>
#include <cmath>
#include <cassert>
#include <armadillo>

#include "matrix-util.hpp"

const int state_len = 13;
const int problem_dims = 3;

typedef arma::vec::fixed<state_len> state_vec_t;
typedef arma::mat::fixed<state_len, state_len> state_mat_t;

arma::vec6 matrix_to_voigt(arma::mat33);
arma::mat33 voigt_to_matrix(arma::vec6);

// Compute Finger tensor G from deformation gradient F
arma::mat33 FingerG(arma::mat33 F);

// Derivative of the matrix G by a component of F, namely F(p,q).
// Notice that the argument is inv(F).
arma::mat33 dG_dF(int p, int q, arma::mat33 inv_F);

// Simple wrapper for state vectors
class State {
	state_vec_t data;

protected:
	arma::vec3 get_vector(void) const 
	{   
		state_vec_t::const_iterator start = data.begin();
		return arma::vec3(&start[idx_beg::vector]);
	}
	arma::mat33 get_matrix(void) const
	{ 
		state_vec_t::const_iterator start = data.begin();
		return arma::mat33(&start[idx_beg::matrix]);
	}
	double get_scalar(void) const { return data[idx_beg::scalar]; }

public:
	struct idx_beg { enum { vector= 0, matrix= 3, scalar=12 }; };
	struct idx_end { enum { vector= 3, matrix=12, scalar=13 }; };

	state_vec_t repr(void) const { return data; } 

	State(state_vec_t s) : data(s) { }
	State(void) { data.fill(0.0); }
	State(arma::vec3 vector, arma::mat33 matrix, double scalar) 
	{
		state_vec_t::iterator it = data.begin();
		for (auto &vector_elt : vector) *it++ = vector_elt;
		for (auto &matrix_elt : matrix) *it++ = matrix_elt;
		*it++ = scalar;
		assert(it == data.end());
	}
};

class ConsState: public State {
public:
	arma::vec3 mom(void) const { return get_vector(); }
	arma::mat33 rhoF(void) const { return get_matrix(); }
	double rhoE(void) const { return get_scalar(); }

	double density(double rho0) const { return sqrt(det(rhoF()) / rho0); }

	template <typename... Ts>
	ConsState(Ts... params) : State(params...) { };
};

class PrimState: public State {
public:
	arma::vec3 u(void) const { return get_vector(); }
	arma::mat33 F(void) const { return get_matrix(); }
	double S(void) const { return get_scalar(); }

	double density(double rho0) const { return rho0 / det(F()); }

	template <typename... Ts>
	PrimState(Ts... params) : State(params...) { };
};

#endif
