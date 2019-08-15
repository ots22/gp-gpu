#include "cov.hpp"

static double cov_val(vec x, vec y, vec hypers)
{
	double scale = hypers(hypers.n_rows-1);
	double s = 0.0;
	for (unsigned i=0; i<x.n_rows; i++)
	{
		double d = (x(i)-y(i))/hypers[i];
		s += d * d;
	}
	return scale * exp(-0.5*s);
}

static double dcov_x1(int n, vec x, vec y, vec hypers)
{
	return -(x(n)-y(n))/(hypers(n)*hypers(n)) * cov_val(x,y,hypers);
}

static double dcov_x2(int n, vec x, vec y, vec hypers)
{
	return -dcov_x1(n,x,y,hypers);
}

static double d2cov_xx(int n, int m, vec x, vec y, vec hypers)
{
	return ((n==m)?(cov_val(x,y,hypers)/(hypers(n)*hypers(n))):0.0) 
		- (x(n)-y(n))/(hypers(n)*hypers(n)) * dcov_x2(m,x,y,hypers);
}

double cov(int n, int m, vec x, vec y, vec hypers)
{
	if (n == 0 && m == 0) return cov_val(x,y,hypers);
	else if (n==0) return dcov_x2(m-1,x,y,hypers);
	else if (m==0) return dcov_x1(n-1,x,y,hypers);
	else return d2cov_xx(n-1,m-1,x,y,hypers);
}
