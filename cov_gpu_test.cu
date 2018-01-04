#include "cov.hpp"
#include "cov_gpu.hpp"
#include <iostream>

// __device__ double cov_val_d(int n_dim, double *x, double *y, double *hypers)
// {
// 	double scale = hypers[0];
// 	//vec r(hypers.rows(1,hypers.n_rows-1));

// 	// bit of a hack.  Assumes a size on the vectors
// 	//vec r{hypers(1), hypers(1), hypers(1), hypers(2), hypers(2), hypers(2), hypers(3)};
// 	double r[] = {hypers[1], hypers[1], hypers[1], hypers[2], hypers[2], hypers[2], hypers[3]};

// 	double s = 0.0;
// 	for (unsigned i=0; i<n_dim; i++)
// 	{
// 		s+=pow(x[i]-y[i],2.0)/(r[i]*r[i]);
// 	}
// 	return scale * exp(-0.5*s);
// }


int main(void)
{
	const int Ninput = 7;

	vec x(Ninput); x.fill(0.0);
	vec y(Ninput); y.fill(0.1);
	vec hypers(4); hypers.fill(1.0);
	std::cout << cov(0, 0, x, y, hypers) << std::endl;
	
	double *x_d;
	double *y_d;
	double *hypers_d;
	double *result_d;

	if (cudaMalloc((void**)(&x_d), Ninput*sizeof(double)) != cudaSuccess) {
		throw std::runtime_error("Device allocation failure (x_d)");
	}
	if (cudaMalloc((void**)(&y_d), Ninput*sizeof(double)) != cudaSuccess) {
		throw std::runtime_error("Device allocation failure (y_d)");
	}
	if (cudaMalloc((void**)(&hypers_d), hypers.n_rows*sizeof(double)) != cudaSuccess) {
		throw std::runtime_error("Device allocation failure (hypers_d)");
	}
	if (cudaMalloc((void**)(&result_d), sizeof(double)) != cudaSuccess) {
		throw std::runtime_error("Device allocation failure (result_d)");
	}

	cudaMemcpy(x_d, x.memptr(), Ninput*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y.memptr(), Ninput*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(hypers_d, hypers.memptr(), hypers.n_rows*sizeof(double), cudaMemcpyHostToDevice);
	cov_val_wrapper(result_d, Ninput, x_d, y_d, hypers_d);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
		printf("Error: %s\n", cudaGetErrorString(err));

	double result;
	cudaMemcpy(&result, result_d, sizeof(double), cudaMemcpyDeviceToHost);
	std::cout << result << std::endl;
}

