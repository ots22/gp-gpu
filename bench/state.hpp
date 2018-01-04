#ifndef STATE_HPP
#define STATE_HPP

#include <numeric>

#ifdef TVMET_DEBUG
#include <iostream>
#endif
#include <cmath>
using std::isnan;
using std::isinf;
#include <tvmet/Vector.h>
#include <tvmet/Matrix.h>

#include "matrix-util.hpp"

const int state_len = 13;
const int problem_dims = 3;

typedef tvmet::Vector<double, 3> vec3;
typedef tvmet::Vector<double, 6> vec6;
typedef tvmet::Matrix<double, 3, 3> mat3;
typedef tvmet::Vector<double, state_len> state_vec_t;
typedef tvmet::Matrix<double, state_len, state_len> state_mat_t;

using namespace tvmet;

vec6 matrix_to_voigt(mat3);
mat3 voigt_to_matrix(vec6);

// Compute Finger tensor G from deformation gradient F
mat3 FingerG(mat3 F);

// Derivative of the matrix G by a component of F, namely F(p,q).
// Notice that the argument is inv(F).
mat3 dG_dF(int p, int q, mat3 inv_F);

// Simple wrapper for state vectors
class State {
	state_vec_t data;

protected:
	vec3 get_vector(void) const 
	{   
		state_vec_t::const_iterator start = data.begin();
		return vec3(&start[idx_beg::vector],
			    &start[idx_end::vector]);
	}
	mat3 get_matrix(void) const
	{ 
		state_vec_t::const_iterator start = data.begin();
		return mat3(&start[idx_beg::matrix],
			    &start[idx_end::matrix]);
	}
	double get_scalar(void) const { return data[idx_beg::scalar]; }

public:
	struct idx_beg { enum { vector= 0, matrix= 3, scalar=12 }; };
	struct idx_end { enum { vector= 3, matrix=12, scalar=13 }; };

	state_vec_t repr(void) const { return data; } 

	State(state_vec_t s) : data(s) { }
	State(void) : data(0) { }
	State(vec3 vector, mat3 matrix, double scalar) 
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
	vec3 mom(void) const { return get_vector(); }
	mat3 rhoF(void) const { return get_matrix(); }
	double rhoE(void) const { return get_scalar(); }

	double density(double rho0) const { return sqrt(det(rhoF()) / rho0); }

	template <typename... Ts>
	ConsState(Ts... params) : State(params...) { };
};

class PrimState: public State {
public:
	vec3 u(void) const { return get_vector(); }
	mat3 F(void) const { return get_matrix(); }
	double S(void) const { return get_scalar(); }

	double density(double rho0) const { return rho0 / det(F()); }

	template <typename... Ts>
	PrimState(Ts... params) : State(params...) { };
};

#endif
