#include "gp_gpu.hpp"

#include <iostream>

int main(void) {
	DenseGP gp("eos-example.gp");
	DenseGP_GPU gp_gpu("eos-example.gp");

	//vec in{0.95, 1.0, 1.0, 0.0, 0.0, 0.0,  0.0};
	
	mat in(7,2);
	in.col(0) = vec{0.95, 1.0, 1.0, 0.0, 0.0, 0.0,  0.0};
	in.col(1) = vec{1.0, 1.0, 1.0, 0.0, 0.0, 0.0,  0.0};

	Mat<REAL> Rin = arma::conv_to<Mat<REAL> >::from(in);

	vec result{-1.0,-1.0};
	Col<REAL> Rresult(2);

	std::cout << "CPU: " << gp.predict(in.col(0), 0) << std::endl;
	std::cout << "CPU: " << gp.predict(in.col(1), 0) << std::endl;

	gp_gpu.predict_batch(Rresult, Rin, 0);
	std::cout << "GPU: " <<  Rresult.t() << std::endl;

	return 0;
}

