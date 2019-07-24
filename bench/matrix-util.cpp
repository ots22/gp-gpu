#include "matrix-util.hpp"
#include <iostream>
#include <iomanip>

template<> bool approx_equal(double a, double b, double tol)
{
	return fabs(a - b) < tol;
}

void write_voigt(std::ostream& logfile, arma::mat33 G)
{
	logfile << std::setw(12) <<  G(0,0) <<  " "
		<< std::setw(12) <<  G(1,1) <<  " "
		<< std::setw(12) <<  G(2,2) <<  " "
		<< std::setw(12) <<  G(1,2) <<  " "
		<< std::setw(12) <<  G(0,2) <<  " "
		<< std::setw(12) <<  G(0,1);		
}

bool is_symmetric(arma::mat33 G)
{
	return (G(0,1) == G(1,0)) && (G(0,2) == G(2,0)) && (G(1,2) == G(2,1));
}

bool is_approx_symmetric(arma::mat33 G)
{
	return approx_equal(G(0,1), G(1,0))
		&& approx_equal(G(0,2), G(2,0))
		&& approx_equal(G(1,2), G(2,1));
}
