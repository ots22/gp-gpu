#include "state.hpp"
#include <iostream>

mat3 FingerG(mat3 F)
{
	mat3 F_inv, G;
	F_inv = inv(F);
	G = F_inv * trans(F_inv);
	return G;
}

// dG_ij/dF_pq, where i and j are the indices of the returned matrix
mat3 dG_dF(int p, int q, mat3 F_inv)
{
	mat3 G (F_inv * trans(F_inv));
	mat3 result;
	for (int i=0; i<3; i++)
		for (int j=0; j<3; j++)
			result(i,j) = - G(i,q) * F_inv(j,p) - G(j,q) * F_inv(i,p);
	return result;
}

vec6 matrix_to_voigt(mat3 M)
{
	vec6 result;
	result[0] = M(0,0);
	result[1] = M(1,1);
	result[2] = M(2,2);
	result[3] = M(1,2);
	result[4] = M(0,2);
	result[5] = M(0,1);
	return result;
}

mat3 voigt_to_matrix(vec6 v)
{
	mat3 result;
	result(0,0) = v[0];
	result(1,1) = v[1];
	result(2,2) = v[2];
	result(1,2) = result(2,1) = v[3];
	result(0,2) = result(2,0) = v[4];
	result(0,1) = result(1,0) = v[5];
	return result;
}
