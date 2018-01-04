#include "matrix-util.hpp"

template<> bool approx_equal(double a, double b, double tol)
{
	return fabs(a - b) < tol;
}

void write_voigt(std::ostream& logfile, tvmet::Matrix<double,3,3> G)
{
	logfile << std::setw(12) <<  G(0,0) <<  " "
		<< std::setw(12) <<  G(1,1) <<  " "
		<< std::setw(12) <<  G(2,2) <<  " "
		<< std::setw(12) <<  G(1,2) <<  " "
		<< std::setw(12) <<  G(0,2) <<  " "
		<< std::setw(12) <<  G(0,1);		
}
