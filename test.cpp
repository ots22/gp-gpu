#include "gp.hpp"
#include "optim.hpp"
#include <iostream>

static double f(double x)
{
	return x*x;
}

static double Df(double x)
{
	return 2*x + 1e-6;
}

int main(void)
{
	const int N=5;
	vec xs(2*N),ts(2*N);
	ivec Ts(2*N);

	for (int i=0; i<N; i++)
	{
		xs(i) = 5.0 * (double)i/(N-1);
		ts(i) = f(xs(i));
		Ts(i) = 0;
		xs(i+N) = 5.0 * (double)i/(N-1);
		ts(i+N) = Df(xs(i));
		Ts(i+N) = 1;
		
	}

	vec hypers = {1e-3, 1.0, 1.0};

	DenseGP gp(hypers,xs,ts,Ts);

	gp.print();
	
	vec work(2*N);

	log_lik_optim(gp, vec{1e-3, 0.1,  0.1}, vec{1.0, 1000.0, 10.0}, 
		      1000, 1e-6);

	gp.save("out.gp");
	
	DenseGP gp2("out.gp");
	
	for (int i=0;i<1000;i++) {
		double x = 5.0*(double)i/1000;
		std::cout << x << " " 
			  << gp.predict(vec{x}, 0, work) << " "
			  << gp2.predict(vec{x}, 0, work) << std::endl; // should be identical
	}
}

