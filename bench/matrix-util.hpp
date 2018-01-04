#ifndef MATRIX_UTIL_HPP
#define MATRIX_UTIL_HPP

#include <iostream>
#include <cmath>
using std::isnan;
using std::isinf;
#include <tvmet/Vector.h>
#include <tvmet/Matrix.h>
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

void write_voigt(std::ostream& logfile, tvmet::Matrix<double, 3, 3> G);

// tvmet provides no matrix determinant or inverse, provide one for 3x3:
template <typename T> T det(tvmet::Matrix<T, 3, 3> M)
{
	return M(0,0) * (M(1,1) * M(2,2) - M(1,2) * M(2,1))
		- M(0,1) * (M(1,0) * M(2,2) - M(1,2) * M(2,0))
	        + M(0,2) * (M(1,0) * M(2,1) - M(1,1) * M(2,0));
}

template <typename T> tvmet::Matrix<T, 3, 3> inv(tvmet::Matrix<T, 3, 3> A)
{
	tvmet::Matrix<T, 3, 3> result;
	double determinant = det(A);
	double invdet = 1.0/determinant;
	result(0,0) =  (A(1,1)*A(2,2)-A(2,1)*A(1,2))*invdet;
	result(0,1) = -(A(0,1)*A(2,2)-A(0,2)*A(2,1))*invdet;
	result(0,2) =  (A(0,1)*A(1,2)-A(0,2)*A(1,1))*invdet;
	result(1,0) = -(A(1,0)*A(2,2)-A(1,2)*A(2,0))*invdet;
	result(1,1) =  (A(0,0)*A(2,2)-A(0,2)*A(2,0))*invdet;
	result(1,2) = -(A(0,0)*A(1,2)-A(1,0)*A(0,2))*invdet;
	result(2,0) =  (A(1,0)*A(2,1)-A(2,0)*A(1,1))*invdet;
	result(2,1) = -(A(0,0)*A(2,1)-A(2,0)*A(0,1))*invdet;
	result(2,2) =  (A(0,0)*A(1,1)-A(1,0)*A(0,1))*invdet;
	return result;
}

template <typename T> bool is_symmetric(tvmet::Matrix<T, 3, 3> G)
{
	return (G(0,1) == G(1,0)) && (G(0,2) == G(2,0)) && (G(1,2) == G(2,1));
}

template <typename T> bool is_approx_symmetric(tvmet::Matrix<T, 3, 3> G)
{
	return approx_equal(G(0,1), G(1,0))
		&& approx_equal(G(0,2), G(2,0))
		&& approx_equal(G(1,2), G(2,1));
}

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
template <typename T, int N>
void eigen_decompose(tvmet::Matrix<T, N, N> J,
		     tvmet::Vector<T, N>& lambda_real,
		     tvmet::Vector<T, N>& lambda_imag,
		     tvmet::Matrix<T, N, N>& eig_left,
		     tvmet::Matrix<T, N, N>& eig_right)
{
	tvmet::Matrix<T, N, N> eig_left_unsorted, eig_right_unsorted;
	tvmet::Vector<T, N> lambda_real_unsorted, lambda_imag_unsorted;

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
template <typename T, int N>
void lin_solve(tvmet::Matrix<T, N, N> A,
	       tvmet::Vector<T, N>& b) // right hand side in, solution out.
{
	const int n = N, nrhs = 1;
	int ipiv[n];
	int err = -999;
	dgesv_(&n, &nrhs, A.data(), &n, ipiv, b.data(), &n, &err);
}


// Find Givens rotation (cos, sin) pair
template <typename T> std::pair<T, T> approx_givens_angle(T a11, T a12, T a22)
{
	bool b = a12*a12 < (a11-a22)*(a11-a22);
	T omega = 1.0/sqrt(a12*a12 + (a11-a22)*(a11-a22));
	T s = b?(omega*a12):(sqrt(0.5));
	T c = b?(omega*(a11-a22)):(sqrt(0.5));
	return std::make_pair(c,s);
}

template <typename T> T jacobi_eig_sym(tvmet::Matrix<T,3,3> A,
				    tvmet::Vector<T,3>& lambda,
				    tvmet::Matrix<T,3,3>& right)
{
	const int num_sweeps = 3;

	right = tvmet::identity<T,3,3>();

	tvmet::Matrix<T,3,3> Q;
	std::pair<double, double> givens;

	for (int n=0; n<num_sweeps; n++) {
		//////////////
		// p=0; q=1;
		givens = approx_givens_angle(A(0,0), A(0,1), A(1,1));
		double c = givens.first;
		double s = givens.second;

		Q = c,-s, 0,
		    s, c, 0,
		    0, 0, 1;

		// Works around a bug with
		// A = tvmet::trans(Q) * A * Q
		// in tvmet, which updates A to the incorrect value
		tvmet::Matrix<T,3,3> A_next;
		A_next = tvmet::trans(Q) * A * Q;
		A = A_next;

		tvmet::Matrix<T,3,3> right_next;
		right_next = right * Q;
		right = right_next;

		//////////////
		// p=0; q=2;
		givens = approx_givens_angle(A(0,0), A(0,2), A(2,2));
		c = givens.first;
		s = givens.second;

		Q = c, 0,-s,
		    0, 1, 0,
		    s, 0, c;

		A_next = tvmet::trans(Q) * A * Q;
		A = A_next;
		right_next = right * Q;
		right = right_next;

		//////////////
		// p=1; q=2;
		givens = approx_givens_angle(A(1,1), A(1,2), A(2,2));
		c = givens.first;
		s = givens.second;

		Q = 1, 0, 0,
	 	    0, c,-s,
		    0, s, c;

		A_next = tvmet::trans(Q) * A * Q;
		A = A_next;
		right_next = right * Q;
		right = right_next;

	}
	lambda[0] = A(0,0);
	lambda[1] = A(1,1);
	lambda[2] = A(2,2);

	return std::max(A(0,1),std::max(A(1,0),A(1,2))); // return the maximum off diagonal element;
}


#endif
