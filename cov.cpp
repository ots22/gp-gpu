#include "cov.hpp"

static double cov_val(vec x, vec y, vec hypers)
{
	double scale = hypers(0);
	//vec r(hypers.rows(1,hypers.n_rows-1));

	// bit of a hack.  Assumes a size on the vectors
	vec r{hypers(1), hypers(1), hypers(1), hypers(2), hypers(2), hypers(2), hypers(3)};

	double s = 0.0;
	for (unsigned i=0; i<x.n_rows; i++)
	{
		s+=pow(x(i)-y(i),2)/(r(i)*r(i));
	}
	return scale * exp(-0.5*s);
}

static double dcov_x1(int n, vec x, vec y, vec hypers)
{
//	vec r(hypers.rows(1,hypers.n_rows-1));
	vec r{hypers(1), hypers(1), hypers(1), hypers(2), hypers(2), hypers(2), hypers(3)};

	return -(x(n)-y(n))/(r(n)*r(n)) * cov_val(x,y,hypers);
}

static double dcov_x2(int n, vec x, vec y, vec hypers)
{
	return -dcov_x1(n,x,y,hypers);
}

static double d2cov_xx(int n, int m, vec x, vec y, vec hypers)
{
//	vec r(hypers.rows(1,hypers.n_rows-1));
	vec r{hypers(1), hypers(1), hypers(1), hypers(2), hypers(2), hypers(2), hypers(3)};

	return ((n==m)?(cov_val(x,y,hypers)/(r(n)*r(n))):0.0) 
		- (x(n)-y(n))/(r(n)*r(n)) * dcov_x2(m,x,y,hypers);
}

double cov(int n, int m, vec x, vec y, vec hypers)
{
	if (n == 0 && m == 0) return cov_val(x,y,hypers);
	else if (n==0) return dcov_x2(m-1,x,y,hypers);
	else if (m==0) return dcov_x1(n-1,x,y,hypers);
	else return d2cov_xx(n-1,m-1,x,y,hypers);
}
