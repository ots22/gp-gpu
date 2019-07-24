#ifndef MATRIX_UTIL_HPP
#define MATRIX_UTIL_HPP

#include <iostream>
#include <cmath>

#include <armadillo>

#include <math.h>
#include <utility> // pair
#include <algorithm> // max

template <typename T>
bool approx_equal(T a, T b, double tol=1.0e-10)
{
	return std::inner_product(a.begin(), a.end(),
				  b.begin(),
				  true,
				  std::logical_and<bool>(),
				  [tol](double x, double y){return approx_equal(x,y,tol);}
		);
}

template <> bool approx_equal(double a, double b, double tol);

void write_voigt(std::ostream& logfile, arma::mat33 G);

bool is_symmetric(arma::mat33 G);

bool is_approx_symmetric(arma::mat33 G);

// Declarations for LAPACK
extern "C" {
	void dgeev_(const int *jobvl, const int *jobvr, const int *n, double *a,
		    const int *lda, double *wr, double *wi, double *vl,
		    const int *ldvl, double *vr, const int *ldvr, double *work,
		    const int *lwork, int *info);
	
	void dgesv_(const int *N, const int *nrhs, double *A, const int *lda,
		    int *ipiv, double *b, const int *ldb, int *info);

}

// Returns the eigendecomposition (both left and right eigenvectors),
// sorted by the real part of the eivenvalue.
template <int N>
void eigen_decompose(arma::mat::fixed<N, N> J,
		     arma::vec::fixed<N>& lambda_real,
		     arma::vec::fixed<N>& lambda_imag,
		     arma::mat::fixed<N, N>& eig_left,
		     arma::mat::fixed<N, N>& eig_right)
{
	arma::mat::fixed<N, N> eig_left_unsorted, eig_right_unsorted;
	arma::vec::fixed<N> lambda_real_unsorted, lambda_imag_unsorted;

	const int n = N; // need to pass n by reference.
	const int lwork_dgeev = 12 * N;
	const int jobvl = 'V';
	const int jobvr = 'N';
	double work_dgeev[lwork_dgeev];
	int err;

	// Note: left and right eigenvectors passed in the opposite
	// order to those documented in DGEEV, since this assumes
	// column-major order, so here the matrix is passed as its
	// transpose.
	dgeev_(&jobvl, &jobvr, &n, J.data(), &n,
	       lambda_real_unsorted.data(), lambda_imag_unsorted.data(),
	       eig_right_unsorted.data(), &n, eig_left_unsorted.data(), &n,
	       work_dgeev, &lwork_dgeev, &err);

	if (err) std::cerr << "LAPACK dgeev failed with error code " << err << std::endl;

	// Sort the eigenvalues and eigenvectors using an index
	std::array<int, N> idx;
	for (int i=0; i<N; i++) idx[i] = i;
	std::sort(idx.begin(), idx.end(),
		  [&lambda_real_unsorted](int p, int q)
		  {
			  return lambda_real_unsorted[p] < lambda_real_unsorted[q];
		  }
		);

	// Idea: This algorithm works for a general matrix, but in our
	// case, instead of sorting the whole array as above (with
	// many degenerate waves), use std::partial_sort to find just
	// the smallest three (put these in order at the beginning),
	// and then the largest three (put these in order at the end).


	for (int i=0; i<N; i++) {
		lambda_real[i] = lambda_real_unsorted[idx[i]];
		lambda_imag[i] = lambda_imag_unsorted[idx[i]];
		for (int j=0; j<N; j++) {
			// eig_left(i,j) = eig_left_unsorted(idx[i],j);
			eig_right(i,j) = eig_right_unsorted(idx[i],j);
		}
	}
}

// solves xA = b, for row vectors x and b.  Solution overwrites b.
template <int N>
void lin_solve(arma::mat::fixed<N, N> A,
	       arma::vec::fixed<N>& b) // right hand side in, solution out.
{
	const int n = N, nrhs = 1;
	int ipiv[n];
	int err = -999;
	dgesv_(&n, &nrhs, A.data(), &n, ipiv, b.data(), &n, &err);
}

#endif
