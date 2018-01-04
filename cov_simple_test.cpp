#include "cov.hpp"
#include <iostream>

int main(void)
{
	const int Ninput = 7;

	vec x(Ninput); x.fill(0.0);
	vec y(Ninput); y.fill(0.1);
	vec hypers(4); hypers.fill(1.0);
	std::cout << cov(0, 0, x, y, hypers) << std::endl;
}
